from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from hacvt.io import load_profile, require_calibrated_tau
from hacvt.model import HACVT


# -------------------------
# Loading (default model + optional profile)
# -------------------------

def load_default_model() -> HACVT:
    """
    Load the bundled default model shipped inside the package (hacvt/default_model.json).
    This is the one that should work for Colab users after pip install.
    """
    return HACVT.load_default()


def load_profile_json(profile_path: str) -> Dict[str, Any]:
    """
    Load a profile.json / car_reviews_profile.json from disk.
    """
    return load_profile(profile_path)


def _get_delta_mean(profile: Dict[str, Any]) -> float:
    calib = profile.get("calibration_report", {})
    if isinstance(calib, dict) and isinstance(calib.get("delta_mean"), (int, float)):
        return float(calib["delta_mean"])
    return 0.0


# -------------------------
# Public inference API (VADER-like)
# -------------------------

def predict_text(
    text: str,
    *,
    model: Optional[HACVT] = None,
    mode: str = "default",                 # "default" or "calibrated"
    profile_path: Optional[str] = None,    # required if mode="calibrated"
    tau: Optional[float] = None,           # optional override
    kappa: float = 0.0,
    use_delta_centering: bool = True,
) -> str:
    """
    Predict a single text.

    mode="default":
      - uses bundled model
      - uses tau if explicitly provided, else uses 0.0 (raw behaviour)

    mode="calibrated":
      - requires profile_path (must contain calibrated tau)
      - uses tau from profile unless overridden
      - can use delta_mean centering if present in calibration_report
    """
    if model is None:
        model = load_default_model()

    mode = mode.strip().lower()
    if mode not in {"default", "calibrated"}:
        raise ValueError("mode must be 'default' or 'calibrated'")

    if mode == "default":
        tau_val = float(tau) if tau is not None else 0.0
        delta_mean = 0.0
    else:
        if not profile_path:
            raise ValueError("profile_path is required when mode='calibrated'")
        profile = load_profile_json(profile_path)
        tau_val = float(tau) if tau is not None else float(require_calibrated_tau(profile))
        delta_mean = _get_delta_mean(profile) if use_delta_centering else 0.0

    return model.predict_one_gated(
        text,
        tau=tau_val,
        delta_mean=delta_mean,
        kappa=float(kappa),
    )


def predict_csv(
    input_csv: str,
    output_csv: str = "predictions.csv",
    *,
    text_col: str = "text",
    model: Optional[HACVT] = None,
    mode: str = "default",
    profile_path: Optional[str] = None,
    tau: Optional[float] = None,
    kappa: float = 0.0,
    use_delta_centering: bool = True,
) -> str:
    """
    Batch prediction for a CSV containing a text column.
    Adds a column: hacvt_pred

    Works for both:
      - unlabelled user datasets
      - labelled datasets (labels are simply carried through)
    """
    if model is None:
        model = load_default_model()

    in_path = Path(input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Resolve tau/delta the same way as predict_text
    mode = mode.strip().lower()
    if mode == "default":
        tau_val = float(tau) if tau is not None else 0.0
        delta_mean = 0.0
    elif mode == "calibrated":
        if not profile_path:
            raise ValueError("profile_path is required when mode='calibrated'")
        profile = load_profile_json(profile_path)
        tau_val = float(tau) if tau is not None else float(require_calibrated_tau(profile))
        delta_mean = _get_delta_mean(profile) if use_delta_centering else 0.0
    else:
        raise ValueError("mode must be 'default' or 'calibrated'")

    # Read
    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        if text_col not in reader.fieldnames:
            raise ValueError(f"CSV missing text column '{text_col}'. Found: {reader.fieldnames}")
        rows = list(reader)

    # Predict
    for r in rows:
        t = (r.get(text_col) or "").strip()
        r["hacvt_pred"] = (
            model.predict_one_gated(t, tau=tau_val, delta_mean=delta_mean, kappa=float(kappa))
            if t else ""
        )

    # Write
    out_path = Path(output_csv)
    fieldnames = list(rows[0].keys()) if rows else [text_col, "hacvt_pred"]
    if "hacvt_pred" not in fieldnames:
        fieldnames.append("hacvt_pred")

    with out_path.open("w", encoding="utf-8", newline="") as g:
        w = csv.DictWriter(g, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return str(out_path)
