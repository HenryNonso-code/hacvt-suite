# hacvt/io.py
import json
from pathlib import Path
from typing import Any, Dict, Optional


def make_profile(
    name: str,
    tau: float,
    label_map: Optional[Dict[str, int]] = None,
    meta: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Creates a single, consistent HAC-VT profile object.

    REQUIRED:
      - tau: float (neutral band half-width or your chosen tau definition)

    OPTIONAL:
      - label_map: e.g. {"neg": 0, "neu": 1, "pos": 2}
      - meta: dataset, timestamp, dev_macroF1, etc.
      - params: any model/config parameters your hacvt_predict uses
    """
    if tau is None:
        raise ValueError("tau must not be None. Calibrate tau before saving a profile.")
    if not isinstance(tau, (int, float)):
        raise TypeError(f"tau must be numeric, got {type(tau)}")

    if label_map is None:
        label_map = {"neg": 0, "neu": 1, "pos": 2}

    profile = {
        "name": str(name),
        "tau": float(tau),
        "label_map": dict(label_map),
        "params": dict(params) if params else {},
        "meta": dict(meta) if meta else {},
        "version": "1.0"
    }
    return profile


def save_profile(profile: Dict[str, Any], path: str) -> str:
    """
    Saves the profile to JSON. Returns the saved path.
    """
    if "tau" not in profile:
        raise ValueError("Profile is missing 'tau'. Refuse to save.")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return str(out)


def load_profile(path: str) -> Dict[str, Any]:
    """
    Loads a profile JSON and validates it contains tau.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    profile = json.loads(p.read_text(encoding="utf-8"))

    if "tau" not in profile:
        raise ValueError("Invalid profile: missing 'tau'. Recalibrate and save again.")
    if not isinstance(profile["tau"], (int, float)):
        raise TypeError(f"Invalid profile: 'tau' must be numeric, got {type(profile['tau'])}")

    # Ensure defaults exist
    profile.setdefault("label_map", {"neg": 0, "neu": 1, "pos": 2})
    profile.setdefault("params", {})
    profile.setdefault("meta", {})
    profile.setdefault("version", "1.0")

    return profile
# =========================
# Step 3: Tau enforcement helpers
# =========================

DEFAULT_TAU = 0.15  # used ONLY for quick single-text prediction (clearly labelled)


def get_tau_or_default(profile: dict) -> tuple[float, str]:
    """
    Quick-prediction helper.

    Returns:
      (tau_value, tau_source)
    tau_source is one of: "calibrated", "default"

    Rules:
    - If profile has a numeric tau -> use it ("calibrated")
    - Otherwise -> use DEFAULT_TAU ("default")
    """
    tau = profile.get("tau", None) if isinstance(profile, dict) else None
    if isinstance(tau, (int, float)):
        return float(tau), "calibrated"
    return float(DEFAULT_TAU), "default"


def require_calibrated_tau(profile: dict) -> float:
    """
    Dataset-evaluation enforcement.

    MUST be called before any dataset evaluation (test-set scoring).
    Raises ValueError with the exact message you want to show to users.
    """
    if not isinstance(profile, dict):
        raise TypeError("profile must be a dict loaded from a profile.json")

    tau = profile.get("tau", None)
    if tau is None or not isinstance(tau, (int, float)):
        raise ValueError("Please calibrate tau using a labelled dev set before evaluation.")

    return float(tau)
