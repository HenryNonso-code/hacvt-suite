# hacvt/model.py
"""
hacvt.model — Lightweight HAC-VT sentiment model (with confidence gating)

Public API (module-level) expected by dashboard + external users:
- log_likelihoods(text): returns total LL for neg/neu/pos
- best_label_and_margin(text): returns (best_label, margin = best - second_best)
- predict_one_gated(text, tau, delta_mean=0.0, kappa=0.0): legacy gating on centered delta
- predict_one_gated_v2(text, tau, delta_mean=0.0, kappa=0.0, delta_bias=0.0, delta_scale=1.0, adapter_path=None):
    Tier-1 standardization + adapter confidence boost (no polarity shifting)

Notes:
- Default model weights are loaded from hacvt/default_model.json (must be packaged).
- Adapter is a CONFIDENCE modifier only: it boosts margin inside neutral band.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Set

from importlib import resources


# ============================================================
# Tokenisation & Negation Handling
# ============================================================

WORD_RE = re.compile(r"[A-Za-z']+")

NEGATION_WORDS: Set[str] = {
    "not", "no", "never", "hardly", "scarcely", "cannot", "can't",
    "isn't", "dont", "don't", "doesnt", "doesn't", "won't", "wont",
    "wouldn't", "shouldn't", "couldn't", "didn't", "aint", "ain't",
    "neither", "nor"
}

_SENT_SPLIT_RE = re.compile(r"[.!?;]+")


def haac_tokenize(text: str) -> List[str]:
    """
    Negation-aware tokeniser used by HAC-VT.

    Example:
        "not good at all" -> ["NOT_good", "at", "all"]
        "I am not happy. great car" -> negation does not leak past the sentence break
    """
    if not text:
        return []

    output: List[str] = []
    segments = [seg.strip() for seg in _SENT_SPLIT_RE.split(text.lower()) if seg.strip()]

    for seg in segments:
        tokens = WORD_RE.findall(seg)
        negate = False

        for tok in tokens:
            if tok in NEGATION_WORDS:
                negate = True
                continue

            output.append(f"NOT_{tok}" if negate else tok)

            # Attach negation to the next token only
            if negate:
                negate = False

    return output


# ============================================================
# Likelihoods and Scoring
# ============================================================

def compute_counts(
    texts: List[str],
    labels: List[str],
    classes: Tuple[str, str, str] = ("neg", "neu", "pos"),
) -> Tuple[Dict[str, Counter], Dict[str, int]]:
    counts: Dict[str, Counter] = {c: Counter() for c in classes}
    totals: Dict[str, int] = {c: 0 for c in classes}

    for text, label in zip(texts, labels):
        toks = haac_tokenize(text)
        counts[label].update(toks)
        totals[label] += len(toks)

    return counts, totals


def compute_log_likelihoods(
    counts: Dict[str, Counter],
    totals: Dict[str, int],
    alpha: float = 1.0,
) -> Tuple[Dict[str, Dict[str, float]], Set[str]]:
    vocab: Set[str] = set()
    for c in counts:
        vocab.update(counts[c].keys())

    V = len(vocab)
    ll: Dict[str, Dict[str, float]] = {c: {} for c in counts}

    for c in counts:
        total_c = totals[c] + alpha * V
        for tok in vocab:
            ll[c][tok] = math.log((counts[c][tok] + alpha) / total_c)

    return ll, vocab


def delta_for_tokens(tokens: List[str], log_likelihoods: Dict[str, Dict[str, float]]) -> float:
    ll_pos = sum(log_likelihoods["pos"].get(t, 0.0) for t in tokens)
    ll_neg = sum(log_likelihoods["neg"].get(t, 0.0) for t in tokens)
    return ll_pos - ll_neg


def classify_delta(delta: float, tau_low: float, tau_high: float) -> str:
    if delta < tau_low:
        return "neg"
    if delta > tau_high:
        return "pos"
    return "neu"


# ============================================================
# Evaluation Helpers
# ============================================================

def macro_f1(
    true_labels: List[str],
    pred_labels: List[str],
    classes: Tuple[str, str, str] = ("neg", "neu", "pos"),
) -> float:
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}

    for y, yp in zip(true_labels, pred_labels):
        if y == yp:
            per_class[y]["tp"] += 1
        else:
            per_class[y]["fn"] += 1
            per_class[yp]["fp"] += 1

    f1_scores: List[float] = []
    for c in classes:
        tp = per_class[c]["tp"]
        fp = per_class[c]["fp"]
        fn = per_class[c]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0.0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / len(f1_scores)


def train_dev_split(
    texts: List[str],
    labels: List[str],
    dev_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    indices = list(range(len(texts)))
    rnd = random.Random(seed)
    rnd.shuffle(indices)

    split = int(len(indices) * (1.0 - dev_ratio))
    train_idx = indices[:split]
    dev_idx = indices[split:]

    def subset(idxs: List[int], arr: List[Any]) -> List[Any]:
        return [arr[i] for i in idxs]

    return (
        subset(train_idx, texts),
        subset(train_idx, labels),
        subset(dev_idx, texts),
        subset(dev_idx, labels),
    )


def tune_tau(
    deltas: List[float],
    labels: List[str],
    classes: Tuple[str, str, str] = ("neg", "neu", "pos"),
    max_abs: float = 10.0,
    step: float = 0.1,
) -> Tuple[float, float, float]:
    best_f1 = -1.0
    best_t = 0.0

    steps = int(max_abs / step)
    for i in range(steps + 1):
        t = i * step
        tau_low, tau_high = -t, t
        preds = [classify_delta(d, tau_low, tau_high) for d in deltas]
        f1 = macro_f1(labels, preds, classes)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return -best_t, best_t, best_f1


# ============================================================
# HAC-VT Model Class
# ============================================================

class HACVT:
    """
    HAC-VT sentiment model.

    Labels can be:
        * numbers 1–5  (1–2=neg, 3=neu, 4–5=pos)
        * strings 'neg', 'neu', 'pos'
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_tau: float = 10.0,
        tau_step: float = 0.1,
        dev_ratio: float = 0.2,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.max_tau = max_tau
        self.tau_step = tau_step
        self.dev_ratio = dev_ratio
        self.seed = seed

        self.classes: Tuple[str, str, str] = ("neg", "neu", "pos")
        self.log_likelihoods_: Optional[Dict[str, Dict[str, float]]] = None
        self.vocab_: Optional[Set[str]] = None

        self.tau_low_: float = 0.0
        self.tau_high_: float = 0.0
        self.dev_macro_f1_: Optional[float] = None

    @classmethod
    def load_default(cls) -> "HACVT":
        """
        Loads a pre-trained/default model from hacvt/default_model.json inside the package.
        """
        try:
            default_path = resources.files("hacvt").joinpath("default_model.json")
            data = json.loads(default_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(
                "Unable to load default_model.json from the hacvt package. "
                "Confirm it exists at hacvt/default_model.json and is included in the wheel."
            ) from e

        return cls.from_dict(data)

    @staticmethod
    def _map_label(y: Any) -> str:
        if isinstance(y, str):
            ys = y.strip().lower()
            if ys in {"neg", "negative"}:
                return "neg"
            if ys in {"neu", "neutral"}:
                return "neu"
            if ys in {"pos", "positive"}:
                return "pos"
            raise ValueError(f"Unknown label string: {y}")

        if isinstance(y, (int, float)):
            if y <= 2:
                return "neg"
            if int(round(y)) == 3:
                return "neu"
            return "pos"

        raise ValueError(f"Unsupported label type: {type(y)}")

    # -------------------------
    # Training
    # -------------------------
    def fit(self, texts: List[str], labels: List[Any]) -> "HACVT":
        mapped_labels = [self._map_label(y) for y in labels]

        tr_x, tr_y, dev_x, dev_y = train_dev_split(
            texts, mapped_labels, dev_ratio=self.dev_ratio, seed=self.seed
        )

        counts, totals = compute_counts(tr_x, tr_y, self.classes)
        ll, vocab = compute_log_likelihoods(counts, totals, alpha=self.alpha)
        self.log_likelihoods_ = ll
        self.vocab_ = vocab

        dev_deltas = [delta_for_tokens(haac_tokenize(t), self.log_likelihoods_) for t in dev_x]
        tau_low, tau_high, best_f1 = tune_tau(
            dev_deltas, dev_y, classes=self.classes, max_abs=self.max_tau, step=self.tau_step
        )

        self.tau_low_ = tau_low
        self.tau_high_ = tau_high
        self.dev_macro_f1_ = best_f1
        return self

    # -------------------------
    # Core numeric score
    # -------------------------
    def delta(self, text: str) -> float:
        if self.log_likelihoods_ is None:
            raise RuntimeError("Model not fitted/loaded. Use fit(), from_dict(), or load_default().")
        toks = haac_tokenize(text)
        return float(delta_for_tokens(toks, self.log_likelihoods_))

    def decision_value(self, text: str) -> float:
        return self.delta(text)

    # -------------------------
    # Confidence primitives
    # -------------------------
    def log_likelihoods(self, text: str) -> Dict[str, float]:
        """
        Returns total log-likelihood per class for a text.
        """
        if self.log_likelihoods_ is None:
            raise RuntimeError("Model not fitted/loaded. Use fit(), from_dict(), or load_default().")

        toks = haac_tokenize(text)
        out: Dict[str, float] = {}
        for c in ("neg", "neu", "pos"):
            out[c] = float(sum(self.log_likelihoods_[c].get(t, 0.0) for t in toks))
        return out

    def best_label_and_margin(self, text: str) -> Tuple[str, float]:
        """
        Returns (best_label, margin = best_ll - second_best_ll).
        """
        lls = self.log_likelihoods(text)
        items = sorted(lls.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_ll = items[0]
        second_ll = items[1][1]
        return best_label, float(best_ll - second_ll)

    # -------------------------
    # Prediction (original)
    # -------------------------
    def predict_one(self, text: str) -> str:
        d = self.delta(text)
        return classify_delta(d, self.tau_low_, self.tau_high_)

    def predict(self, texts: List[str]) -> List[str]:
        return [self.predict_one(t) for t in texts]

    def predict_one_gated(
        self,
        text: str,
        *,
        tau: float,
        delta_mean: float = 0.0,
        kappa: float = 0.0,
    ) -> str:
        """
        Legacy gating on centered delta (not standardized).

        1) d = delta(text) - delta_mean
        2) if d outside +/-tau => pos/neg
        3) if inside => neu unless margin >= kappa, then best_label
        """
        d = float(self.delta(text) - float(delta_mean))
        if d > float(tau):
            return "pos"
        if d < -float(tau):
            return "neg"

        best_label, margin = self.best_label_and_margin(text)
        if float(margin) >= float(kappa):
            return best_label

        return "neu"

    # ============================================================
    # NEW (Step 2): Adapter (confidence-only) + Tier-1 standardization
    # ============================================================

    def _adapter_delta_adjustment(self, text: str, adapter: Optional[Dict[str, Any]]) -> float:
        """
        Returns the raw adapter sum for tokens in this text.

        Adapter shape:
          adapter = { "token_weights": { "token": weight, ... } }

        IMPORTANT:
        - Adapter is NOT added into delta/z.
        - Adapter affects ONLY confidence via margin boost.
        """
        if not adapter or not isinstance(adapter, dict):
            return 0.0

        token_weights = adapter.get("token_weights")
        if not isinstance(token_weights, dict) or not token_weights:
            return 0.0

        toks = haac_tokenize(text)
        s = 0.0
        for tok in toks:
            w = token_weights.get(tok)
            if isinstance(w, (int, float)):
                s += float(w)
        return float(s)

    def predict_one_gated_v2(
        self,
        text: str,
        *,
        tau: float,
        delta_mean: float = 0.0,
        kappa: float = 0.0,
        delta_bias: float = 0.0,
        delta_scale: float = 1.0,
        adapter: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Tier-1 + Tier-3 decision rule (safe polarity):

        1) raw = delta(text)
        2) z = (raw - delta_mean - delta_bias) / delta_scale
        3) if z outside +/-tau => pos/neg
        4) else (neutral band):
             margin' = margin + abs(adapter_sum)
             if margin' >= kappa => best_label
             else => neu
        """
        raw = float(self.delta(text))

        scale = float(delta_scale)
        if scale == 0.0:
            scale = 1.0

        z = (raw - float(delta_mean) - float(delta_bias)) / scale

        if z > float(tau):
            return "pos"
        if z < -float(tau):
            return "neg"

        best_label, margin = self.best_label_and_margin(text)
        adapter_sum = self._adapter_delta_adjustment(text, adapter)
        margin_prime = float(margin) + abs(float(adapter_sum))

        if margin_prime >= float(kappa):
            return best_label

        return "neu"

    # -------------------------
    # Scoring / analysis
    # -------------------------
    def score(self, texts: List[str], labels: List[Any]) -> float:
        mapped = [self._map_label(y) for y in labels]
        preds = self.predict(texts)
        return macro_f1(mapped, preds, self.classes)

    def analyze(self, text: str) -> Dict[str, Any]:
        d = self.delta(text)
        label = classify_delta(d, self.tau_low_, self.tau_high_)
        best_label, margin = self.best_label_and_margin(text)
        return {
            "text": text,
            "delta": float(d),
            "label": label,
            "tau_low": self.tau_low_,
            "tau_high": self.tau_high_,
            "best_label": best_label,
            "margin": float(margin),
        }

    # -------------------------
    # Serialisation
    # -------------------------
    def to_dict(self) -> Dict[str, Any]:
        if self.log_likelihoods_ is None or self.vocab_ is None:
            raise RuntimeError("Model not fitted/loaded yet. Cannot serialise.")

        return {
            "alpha": self.alpha,
            "max_tau": self.max_tau,
            "tau_step": self.tau_step,
            "dev_ratio": self.dev_ratio,
            "seed": self.seed,
            "classes": list(self.classes),
            "vocab": sorted(list(self.vocab_)),
            "log_likelihoods": self.log_likelihoods_,
            "tau_low": self.tau_low_,
            "tau_high": self.tau_high_,
            "dev_macro_f1": self.dev_macro_f1_,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HACVT":
        obj = cls(
            alpha=data.get("alpha", 1.0),
            max_tau=data.get("max_tau", 10.0),
            tau_step=data.get("tau_step", 0.1),
            dev_ratio=data.get("dev_ratio", 0.2),
            seed=data.get("seed", 42),
        )
        obj.classes = tuple(data.get("classes", ["neg", "neu", "pos"]))  # type: ignore[assignment]
        obj.vocab_ = set(data.get("vocab", []))
        obj.log_likelihoods_ = {c: dict(tok_ll) for c, tok_ll in data["log_likelihoods"].items()}
        obj.tau_low_ = float(data["tau_low"])
        obj.tau_high_ = float(data["tau_high"])
        obj.dev_macro_f1_ = data.get("dev_macro_f1")
        return obj


# ============================================================
# Module-level singleton + public functions (Dashboard API)
# ============================================================

_DEFAULT_MODEL: Optional[HACVT] = None

def _get_default_model() -> HACVT:
    """
    Lazy-load the default HAC-VT model once per process.
    """
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = HACVT.load_default()
    return _DEFAULT_MODEL

def _load_adapter_from_path(adapter_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Loads adapter JSON from disk if provided.
    Expected JSON shape:
      { "token_weights": { "token": weight, ... } }
    """
    if not adapter_path:
        return None
    try:
        with open(adapter_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        # Fail safe: adapter is optional; do not crash predictions
        return None

def log_likelihoods(text: str) -> Dict[str, float]:
    """
    Public API: total log-likelihoods for neg/neu/pos.
    """
    return _get_default_model().log_likelihoods(text)

def best_label_and_margin(text: str) -> Tuple[str, float]:
    """
    Public API: (best_label, margin).
    """
    return _get_default_model().best_label_and_margin(text)

def predict_one_gated(
    text: str,
    tau: float,
    delta_mean: float = 0.0,
    kappa: float = 0.0,
) -> str:
    """
    Public API: legacy gating using centered delta (not standardized).
    """
    return _get_default_model().predict_one_gated(
        text,
        tau=float(tau),
        delta_mean=float(delta_mean),
        kappa=float(kappa),
    )

def predict_one_gated_v2(
    text: str,
    tau: float,
    delta_mean: float = 0.0,
    delta_bias: float = 0.0,
    delta_scale: float = 1.0,
    kappa: float = 0.0,
    adapter_path: Optional[str] = None,
) -> str:
    """
    Public API: Tier-1 standardization + adapter confidence boost.

    Adapter affects ONLY confidence (margin boost) inside the neutral band.
    """
    adapter = _load_adapter_from_path(adapter_path)
    return _get_default_model().predict_one_gated_v2(
        text,
        tau=float(tau),
        delta_mean=float(delta_mean),
        kappa=float(kappa),
        delta_bias=float(delta_bias),
        delta_scale=float(delta_scale),
        adapter=adapter,
    )


__all__ = [
    "HACVT",
    "haac_tokenize",
    "log_likelihoods",
    "best_label_and_margin",
    "predict_one_gated",
    "predict_one_gated_v2",
    "macro_f1",
]
