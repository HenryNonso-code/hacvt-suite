# hacvt/calibration.py
"""
HAC-VT calibration: learn neutral-band tau on a development set.

Core idea:
- For each text, HAC-VT produces a scalar decision value "d"
  (e.g., log-likelihood difference: pos_ll - neg_ll).
- With tau >= 0:
    if d >  tau  -> positive
    if d < -tau  -> negative
    else         -> neutral

This module learns tau on a dev set by maximising a metric (default: macro-F1),
WITH a product-oriented constraint to avoid "neutral inflation".

NEW (Tier-1 + Tier-3 support):
- Learn delta_bias and delta_scale (robustly) and calibrate tau on standardized score:
      z = (d' - delta_bias) / delta_scale
- Optional token-drift adapter trained on dev set:
      d' = d + sum(token_weights[token] for token in tokens(text))
  (learned from pos vs neg; neutral ignored)
- All new functionality is additive: existing calibrate_tau() remains usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

import math
from collections import Counter

try:
    from sklearn.metrics import f1_score
except Exception:  # pragma: no cover
    f1_score = None


# ----------------------------
# DEFAULT POLICY (PRODUCT RULE)
# ----------------------------
# 1) Prevent tau exploding on imbalanced dev sets (neutral inflation).
DEFAULT_MAX_TAU: float = 3.0
# 2) Prevent tau collapsing to 0 (forces a minimal neutral band).
DEFAULT_MIN_TAU: float = 0.2

# 3) Product rule: keep neutral as low as possible (decision-friendly).
#    Option A (hard cap): disallow taus that produce neutral above this rate.
DEFAULT_MAX_NEUTRAL_RATE: float = 0.20  # 20%
#    Option B (soft penalty): among valid taus, prefer smaller neutral.
DEFAULT_NEUTRAL_PENALTY: float = 0.50  # score = macro_f1 - penalty * neutral_rate


Label = str
DecisionFn = Callable[[str], float]
TokenizeFn = Callable[[str], List[str]]


@dataclass(frozen=True)
class TauCalibrationResult:
    tau: float
    metric_name: str
    metric_value: float
    grid: List[Tuple[float, float]]  # list of (tau, metric_value)
    n_dev: int
    min_tau_used: float
    max_tau_used: float
    max_neutral_rate_used: Optional[float]
    neutral_penalty_used: float
    neutral_rate_at_best: float


# ----------------------------
# NEW: Tier-1 + Tier-3 result
# ----------------------------
@dataclass(frozen=True)
class Tier1Tier3CalibrationResult:
    tau: float
    delta_bias: float
    delta_scale: float
    adapter: Dict[str, Any]          # JSON-serialisable (token_weights, stats)
    metric_name: str
    metric_value: float
    neutral_rate_at_best: float
    n_dev: int
    grid: List[Tuple[float, float]]  # tau vs score on standardized space


def apply_tau(decision_value: float, tau: float) -> Label:
    """
    Convert a decision scalar into a 3-class label using tau.
    """
    if decision_value > tau:
        return "pos"
    if decision_value < -tau:
        return "neg"
    return "neu"


def _validate_labels(y: Sequence[Label]) -> None:
    allowed = {"neg", "neu", "pos"}
    bad = sorted({v for v in y if v not in allowed})
    if bad:
        raise ValueError(f"Invalid labels in y_dev: {bad}. Expected only {sorted(allowed)}.")


def _macro_f1(y_true: Sequence[Label], y_pred: Sequence[Label]) -> float:
    """
    Macro-F1 across the three labels in fixed order.
    Uses sklearn if available; otherwise a small fallback.
    """
    labels = ["neg", "neu", "pos"]

    if f1_score is not None:
        return float(f1_score(y_true, y_pred, labels=labels, average="macro"))

    # Fallback macro-F1 (no sklearn)
    def prf_for(lbl: Label) -> float:
        tp = sum((yt == lbl and yp == lbl) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != lbl and yp == lbl) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == lbl and yp != lbl) for yt, yp in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    return (prf_for("neg") + prf_for("neu") + prf_for("pos")) / 3.0


def _neutral_rate(y_pred: Sequence[Label]) -> float:
    if not y_pred:
        return 0.0
    return sum(1 for y in y_pred if y == "neu") / len(y_pred)


def _make_grid_from_decisions(
    decisions: Sequence[float],
    *,
    n_grid: int = 101,
    max_tau: Optional[float] = None,
    min_tau: Optional[float] = None,
) -> Tuple[List[float], float, float]:
    """
    Create a tau grid from [min_tau_used .. max_tau_used].

    Policy:
      - If max_tau is None -> DEFAULT_MAX_TAU
      - If min_tau is None -> DEFAULT_MIN_TAU
      - Ensures max_tau_used >= min_tau_used
      - Returns: (tau_grid, min_tau_used, max_tau_used)
    """
    abs_vals = [abs(d) for d in decisions if d is not None and not math.isnan(d)]
    if not abs_vals:
        # Degenerate case: all NaN/None decisions; fallback to min tau policy if possible
        min_used = float(DEFAULT_MIN_TAU if min_tau is None else min_tau)
        min_used = max(0.0, min_used)
        return [min_used], min_used, min_used

    min_tau_used = float(DEFAULT_MIN_TAU if min_tau is None else min_tau)
    max_tau_used = float(DEFAULT_MAX_TAU if max_tau is None else max_tau)

    min_tau_used = max(0.0, min_tau_used)
    max_tau_used = max(0.0, max_tau_used)

    if max_tau_used < min_tau_used:
        max_tau_used = min_tau_used

    if n_grid <= 1 or math.isclose(max_tau_used, min_tau_used):
        return [min_tau_used], min_tau_used, max_tau_used

    step = (max_tau_used - min_tau_used) / (n_grid - 1)
    grid = [min_tau_used + i * step for i in range(n_grid)]
    return grid, min_tau_used, max_tau_used


# ============================================================
# EXISTING: tau-only calibration (kept stable)
# ============================================================

def calibrate_tau(
    x_dev: Sequence[str],
    y_dev: Sequence[Label],
    decision_fn: DecisionFn,
    *,
    metric: str = "macro_f1",
    n_grid: int = 101,
    max_tau: Optional[float] = None,
    min_tau: Optional[float] = None,
    # Neutral control (decision-friendly)
    max_neutral_rate: Optional[float] = DEFAULT_MAX_NEUTRAL_RATE,
    neutral_penalty: float = DEFAULT_NEUTRAL_PENALTY,
    return_grid: bool = True,
) -> TauCalibrationResult:
    """
    Learn tau on a dev set by grid-searching tau values and maximising a metric.

    Returns TauCalibrationResult with best tau and metric (the penalised score).
    """
    if len(x_dev) != len(y_dev):
        raise ValueError("x_dev and y_dev must have the same length.")
    if len(x_dev) == 0:
        raise ValueError("Dev set is empty; cannot calibrate tau.")

    _validate_labels(y_dev)

    if metric.lower() != "macro_f1":
        raise ValueError(f"Unsupported metric: {metric}. Use metric='macro_f1'.")

    # Compute decisions once
    decisions: List[float] = []
    for t in x_dev:
        decisions.append(float(decision_fn(t)))

    # Prepare tau grid with enforced floor and cap
    tau_grid, min_tau_used, max_tau_used = _make_grid_from_decisions(
        decisions,
        n_grid=n_grid,
        max_tau=max_tau,
        min_tau=min_tau,
    )

    # Safety: ensure valid neutral settings
    max_neutral_rate_used: Optional[float]
    if max_neutral_rate is None:
        max_neutral_rate_used = None
    else:
        max_neutral_rate_used = float(max(0.0, min(1.0, max_neutral_rate)))

    neutral_penalty_used = float(max(0.0, neutral_penalty))

    best_tau = float(tau_grid[0])
    best_score = -1e18
    best_neu_rate = 1.0
    curve: List[Tuple[float, float]] = []

    for tau in tau_grid:
        y_pred = [apply_tau(d, tau) for d in decisions]
        mf1 = _macro_f1(y_dev, y_pred)
        neu_rate = _neutral_rate(y_pred)

        # Hard cap on neutral
        if max_neutral_rate_used is not None and neu_rate > max_neutral_rate_used:
            score = -1e18
        else:
            score = mf1 - neutral_penalty_used * neu_rate

        if return_grid:
            curve.append((float(tau), float(score)))

        # Select best score; tie-breaker prefers smaller tau (less neutral band)
        if (score > best_score) or (math.isclose(score, best_score) and tau < best_tau):
            best_score = float(score)
            best_tau = float(tau)
            best_neu_rate = float(neu_rate)

    metric_name = "macro_f1"
    if neutral_penalty_used > 0.0:
        metric_name = "macro_f1_minus_neutral_penalty"
    if max_neutral_rate_used is not None:
        metric_name = metric_name + "_with_neutral_cap"

    return TauCalibrationResult(
        tau=best_tau,
        metric_name=metric_name,
        metric_value=float(best_score),
        grid=curve if return_grid else [],
        n_dev=len(x_dev),
        min_tau_used=float(min_tau_used),
        max_tau_used=float(max_tau_used),
        max_neutral_rate_used=max_neutral_rate_used,
        neutral_penalty_used=float(neutral_penalty_used),
        neutral_rate_at_best=float(best_neu_rate),
    )


def fit_profile_tau(
    profile: Any,
    x_dev: Sequence[str],
    y_dev: Sequence[Label],
    decision_fn: DecisionFn,
    *,
    n_grid: int = 101,
    max_tau: Optional[float] = None,
    min_tau: Optional[float] = None,
    max_neutral_rate: Optional[float] = DEFAULT_MAX_NEUTRAL_RATE,
    neutral_penalty: float = DEFAULT_NEUTRAL_PENALTY,
) -> Tuple[Any, TauCalibrationResult]:
    """
    Convenience wrapper that learns tau and writes it into the provided profile object.

    Supports:
      - dict-like profile (writes profile["tau"])
      - attribute profile (writes profile.tau)

    Returns (profile, calibration_result).
    """
    result = calibrate_tau(
        x_dev=x_dev,
        y_dev=y_dev,
        decision_fn=decision_fn,
        n_grid=n_grid,
        max_tau=max_tau,
        min_tau=min_tau,
        max_neutral_rate=max_neutral_rate,
        neutral_penalty=neutral_penalty,
        metric="macro_f1",
        return_grid=True,
    )

    if isinstance(profile, dict):
        profile["tau"] = float(result.tau)
    else:
        try:
            setattr(profile, "tau", float(result.tau))
        except Exception as e:
            raise TypeError("Profile object does not support setting 'tau' (dict key or attribute).") from e

    return profile, result


# ============================================================
# NEW: Tier-1 (bias/scale) + Tier-3 (adapter) calibration
# ============================================================

def _mad(vals: List[float]) -> float:
    """
    Median absolute deviation (MAD).
    """
    if not vals:
        return 0.0
    m = _median(vals)
    devs = [abs(v - m) for v in vals]
    return _median(devs)


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float(0.5 * (s[mid - 1] + s[mid]))


def _robust_bias_scale(decisions: List[float], *, min_scale: float = 1e-6) -> Tuple[float, float]:
    """
    Robust bias/scale:
      bias = median(d)
      scale = 1.4826 * MAD(d)  (approx std if normal)
    Fallback to std if MAD is ~0.
    """
    if not decisions:
        return 0.0, 1.0

    bias = _median(decisions)
    mad = _mad(decisions)
    scale = 1.4826 * mad

    if scale < min_scale:
        # fallback: classic std
        mu = sum(decisions) / len(decisions)
        var = sum((d - mu) ** 2 for d in decisions) / max(1, (len(decisions) - 1))
        scale = math.sqrt(var)
        if scale < min_scale:
            scale = 1.0

    return float(bias), float(scale)


def _fit_token_drift_adapter(
    x_dev: Sequence[str],
    y_dev: Sequence[Label],
    tokenize_fn: TokenizeFn,
    *,
    min_token_count: int = 5,
    max_abs_weight: float = 1.5,
) -> Dict[str, Any]:
    """
    Learn token weights using smoothed log-odds ratio between pos and neg.
    Neutral items are ignored.
    Output is JSON-serialisable:
      {
        "token_weights": {...},
        "min_token_count": ...,
        "max_abs_weight": ...,
        "n_pos": ...,
        "n_neg": ...,
        "source": "supervised"
      }
    """
    pos_counts = Counter()
    neg_counts = Counter()
    n_pos = 0
    n_neg = 0

    for t, y in zip(x_dev, y_dev):
        if y not in {"neg", "neu", "pos"}:
            continue
        if y == "neu":
            continue

        toks = tokenize_fn(t)
        if y == "pos":
            pos_counts.update(toks)
            n_pos += 1
        elif y == "neg":
            neg_counts.update(toks)
            n_neg += 1

    vocab = set(pos_counts) | set(neg_counts)
    if n_pos == 0 or n_neg == 0 or not vocab:
        return {
            "token_weights": {},
            "min_token_count": int(min_token_count),
            "max_abs_weight": float(max_abs_weight),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "source": "insufficient_signal",
        }

    alpha = 0.5  # smoothing
    pos_total = sum(pos_counts.values()) + alpha * len(vocab)
    neg_total = sum(neg_counts.values()) + alpha * len(vocab)

    weights: Dict[str, float] = {}
    for tok in vocab:
        c_pos = pos_counts.get(tok, 0)
        c_neg = neg_counts.get(tok, 0)
        if (c_pos + c_neg) < int(min_token_count):
            continue

        p_pos = (c_pos + alpha) / pos_total
        p_neg = (c_neg + alpha) / neg_total
        w = math.log(p_pos / p_neg)

        if w > max_abs_weight:
            w = max_abs_weight
        if w < -max_abs_weight:
            w = -max_abs_weight

        weights[str(tok)] = float(w)

    return {
        "token_weights": weights,
        "min_token_count": int(min_token_count),
        "max_abs_weight": float(max_abs_weight),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "source": "supervised",
    }


def _apply_adapter_to_decision(
    text: str,
    decision_value: float,
    adapter: Dict[str, Any],
    tokenize_fn: TokenizeFn,
) -> float:
    token_weights = adapter.get("token_weights", {})
    if not isinstance(token_weights, dict) or not token_weights:
        return float(decision_value)

    toks = tokenize_fn(text)
    s = 0.0
    for tok in toks:
        w = token_weights.get(tok)
        if isinstance(w, (int, float)):
            s += float(w)
    return float(decision_value) + float(s)


def calibrate_tier1_tier3(
    x_dev: Sequence[str],
    y_dev: Sequence[Label],
    decision_fn: DecisionFn,
    *,
    tokenize_fn: Optional[TokenizeFn] = None,
    # tau grid in standardized space
    metric: str = "macro_f1",
    n_grid: int = 101,
    max_tau: Optional[float] = None,
    min_tau: Optional[float] = None,
    # Neutral control (decision-friendly)
    max_neutral_rate: Optional[float] = DEFAULT_MAX_NEUTRAL_RATE,
    neutral_penalty: float = DEFAULT_NEUTRAL_PENALTY,
    # adapter settings
    use_adapter: bool = True,
    min_token_count: int = 5,
    max_abs_weight: float = 1.5,
    return_grid: bool = True,
) -> Tier1Tier3CalibrationResult:
    """
    Full calibration for predict_one_gated_v2:

      1) base decisions d = decision_fn(text)
      2) optional adapter => d' = d + sum(w_tok)
      3) robust bias/scale on d' => delta_bias, delta_scale
      4) standardize z = (d' - bias) / scale
      5) grid-search tau on z (same neutral cap/penalty logic)

    Returns tau + (bias/scale) + adapter dict.
    """
    if len(x_dev) != len(y_dev):
        raise ValueError("x_dev and y_dev must have the same length.")
    if len(x_dev) == 0:
        raise ValueError("Dev set is empty; cannot calibrate.")

    _validate_labels(y_dev)

    if metric.lower() != "macro_f1":
        raise ValueError(f"Unsupported metric: {metric}. Use metric='macro_f1'.")

    if tokenize_fn is None:
        # Local import to avoid circular imports
        from hacvt.model import haac_tokenize as _tok
        tokenize_fn = _tok

    # 1) base decisions
    base_decisions: List[float] = [float(decision_fn(t)) for t in x_dev]

    # 2) adapter (optional)
    if use_adapter:
        adapter = _fit_token_drift_adapter(
            x_dev=x_dev,
            y_dev=y_dev,
            tokenize_fn=tokenize_fn,
            min_token_count=min_token_count,
            max_abs_weight=max_abs_weight,
        )
        adapted_decisions = [
            _apply_adapter_to_decision(t, d, adapter, tokenize_fn) for t, d in zip(x_dev, base_decisions)
        ]
    else:
        adapter = {
            "token_weights": {},
            "min_token_count": int(min_token_count),
            "max_abs_weight": float(max_abs_weight),
            "n_pos": int(sum(1 for y in y_dev if y == "pos")),
            "n_neg": int(sum(1 for y in y_dev if y == "neg")),
            "source": "disabled",
        }
        adapted_decisions = list(base_decisions)

    # 3) robust bias/scale
    delta_bias, delta_scale = _robust_bias_scale(adapted_decisions)

    # 4) standardized z
    z_vals: List[float] = [float((d - delta_bias) / delta_scale) for d in adapted_decisions]

    # 5) tau grid in standardized space
    tau_grid, min_tau_used, max_tau_used = _make_grid_from_decisions(
        z_vals,
        n_grid=n_grid,
        max_tau=max_tau,
        min_tau=min_tau,
    )

    # neutral settings
    if max_neutral_rate is None:
        max_neutral_rate_used = None
    else:
        max_neutral_rate_used = float(max(0.0, min(1.0, max_neutral_rate)))
    neutral_penalty_used = float(max(0.0, neutral_penalty))

    best_tau = float(tau_grid[0])
    best_score = -1e18
    best_neu_rate = 1.0
    curve: List[Tuple[float, float]] = []

    for tau in tau_grid:
        y_pred = [apply_tau(z, tau) for z in z_vals]
        mf1 = _macro_f1(y_dev, y_pred)
        neu_rate = _neutral_rate(y_pred)

        if max_neutral_rate_used is not None and neu_rate > max_neutral_rate_used:
            score = -1e18
        else:
            score = mf1 - neutral_penalty_used * neu_rate

        if return_grid:
            curve.append((float(tau), float(score)))

        if (score > best_score) or (math.isclose(score, best_score) and tau < best_tau):
            best_score = float(score)
            best_tau = float(tau)
            best_neu_rate = float(neu_rate)

    metric_name = "macro_f1"
    if neutral_penalty_used > 0.0:
        metric_name = "macro_f1_minus_neutral_penalty"
    if max_neutral_rate_used is not None:
        metric_name = metric_name + "_with_neutral_cap"
    metric_name = metric_name + "_tier1_tier3"

    # pack adapter with bias/scale for easier downstream usage
    adapter_out = dict(adapter)
    adapter_out["delta_bias"] = float(delta_bias)
    adapter_out["delta_scale"] = float(delta_scale)
    adapter_out["tau_space"] = "standardized_z"

    return Tier1Tier3CalibrationResult(
        tau=float(best_tau),
        delta_bias=float(delta_bias),
        delta_scale=float(delta_scale),
        adapter=adapter_out,
        metric_name=metric_name,
        metric_value=float(best_score),
        neutral_rate_at_best=float(best_neu_rate),
        n_dev=int(len(x_dev)),
        grid=curve if return_grid else [],
    )


def fit_profile_tier1_tier3(
    profile: Any,
    x_dev: Sequence[str],
    y_dev: Sequence[Label],
    decision_fn: DecisionFn,
    *,
    tokenize_fn: Optional[TokenizeFn] = None,
    n_grid: int = 101,
    max_tau: Optional[float] = None,
    min_tau: Optional[float] = None,
    max_neutral_rate: Optional[float] = DEFAULT_MAX_NEUTRAL_RATE,
    neutral_penalty: float = DEFAULT_NEUTRAL_PENALTY,
    use_adapter: bool = True,
    min_token_count: int = 5,
    max_abs_weight: float = 1.5,
) -> Tuple[Any, Tier1Tier3CalibrationResult]:
    """
    Learns tau + bias/scale + adapter and writes into profile.

    Writes:
      profile["tau"] (or profile.tau)
      profile["calibration_report"]["delta_bias"]
      profile["calibration_report"]["delta_scale"]
      profile["calibration_report"]["adapter"]
      profile["calibration_report"]["metric_name"]
      profile["calibration_report"]["metric_value"]
      profile["calibration_report"]["neutral_rate_at_best"]
      profile["calibration_report"]["tau_space"] = "standardized_z"
    """
    result = calibrate_tier1_tier3(
        x_dev=x_dev,
        y_dev=y_dev,
        decision_fn=decision_fn,
        tokenize_fn=tokenize_fn,
        n_grid=n_grid,
        max_tau=max_tau,
        min_tau=min_tau,
        max_neutral_rate=max_neutral_rate,
        neutral_penalty=neutral_penalty,
        use_adapter=use_adapter,
        min_token_count=min_token_count,
        max_abs_weight=max_abs_weight,
        return_grid=True,
    )

    # ensure calibration_report exists
    if isinstance(profile, dict):
        profile["tau"] = float(result.tau)
        cr = profile.get("calibration_report")
        if not isinstance(cr, dict):
            cr = {}
            profile["calibration_report"] = cr

        cr["tau_space"] = "standardized_z"
        cr["delta_bias"] = float(result.delta_bias)
        cr["delta_scale"] = float(result.delta_scale)
        cr["adapter"] = result.adapter
        cr["metric_name"] = result.metric_name
        cr["metric_value"] = float(result.metric_value)
        cr["neutral_rate_at_best"] = float(result.neutral_rate_at_best)

    else:
        # object style
        try:
            setattr(profile, "tau", float(result.tau))
        except Exception as e:
            raise TypeError("Profile object does not support setting 'tau'.") from e

        cr = getattr(profile, "calibration_report", None)
        if not isinstance(cr, dict):
            cr = {}
            setattr(profile, "calibration_report", cr)

        cr["tau_space"] = "standardized_z"
        cr["delta_bias"] = float(result.delta_bias)
        cr["delta_scale"] = float(result.delta_scale)
        cr["adapter"] = result.adapter
        cr["metric_name"] = result.metric_name
        cr["metric_value"] = float(result.metric_value)
        cr["neutral_rate_at_best"] = float(result.neutral_rate_at_best)

    return profile, result
