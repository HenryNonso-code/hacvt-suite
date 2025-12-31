import sys
from pathlib import Path

# ============================================================
# Render-safe paths:
# - Streamlit runs from /dashboard, so repo root isn't on sys.path.
# - Add repo root so "import hacvt" works.
# - Use absolute paths for dashboard/data files.
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DASHBOARD_DIR = Path(__file__).resolve().parent
DATA_DIR = DASHBOARD_DIR / "data"
PROFILES_DIR = DASHBOARD_DIR / "profiles"

# ===============================
# HAC-VT Dashboard – Full Version (Render-ready)
# ===============================

import os
import json
import re
import inspect
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score


# ---------- Page config ----------
st.set_page_config(page_title="HAC-VT Dashboard", layout="wide")
st.title("HAC-VT Sentiment Analysis – Dashboard")


# =========================================
# 0) HAC-VT imports (robust)
# =========================================
HACVT_IMPORT_OK = True
HACVT_IMPORT_ERR = None

try:
    # Function-based public API (as per your hacvt/model.py header)
    from hacvt.model import predict_one_gated_v2, log_likelihoods, best_label_and_margin
except Exception as e:
    HACVT_IMPORT_OK = False
    HACVT_IMPORT_ERR = str(e)


# =========================================
# 1) Helpers: label mapping + detection
# =========================================
CANON = ["neg", "neu", "pos"]


def _norm_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip().lower()


def guess_text_column(df: pd.DataFrame) -> str:
    common = {"text", "review", "content", "comment", "sentence", "headline", "title", "body"}
    for c in df.columns:
        if _norm_str(c) in common:
            return c

    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        return df.columns[0]

    avg_len: Dict[str, float] = {}
    for c in obj_cols:
        s = df[c].astype(str).fillna("")
        avg_len[c] = s.map(len).mean()
    return max(avg_len, key=avg_len.get)


def guess_label_column(df: pd.DataFrame) -> Optional[str]:
    common = {"label", "sentiment", "target", "y", "class", "rating", "stars", "star", "score"}
    for c in df.columns:
        if _norm_str(c) in common:
            return c
    return None


def ensure_text(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("").map(lambda x: x.strip())


def detect_label_type(series: pd.Series) -> str:
    """
    returns: "stars" | "3class" | "binary" | "unknown"
    """
    s = series.dropna().map(_norm_str)
    uniq = set([u for u in s.unique() if u != ""])

    # stars: numeric 1..5
    numeric = True
    nums: List[float] = []
    for u in uniq:
        try:
            nums.append(float(u))
        except Exception:
            numeric = False
            break
    if numeric and nums and all(1.0 <= x <= 5.0 for x in nums):
        return "stars"

    def canon(x: str) -> str:
        if x in {"neg", "negative"}:
            return "neg"
        if x in {"neu", "neutral"}:
            return "neu"
        if x in {"pos", "positive"}:
            return "pos"
        return x

    uniq2 = set([canon(u) for u in uniq])

    if uniq2.issubset({"neg", "pos"}) and len(uniq2) == 2:
        return "binary"
    if uniq2.issubset({"neg", "neu", "pos"}) and len(uniq2) >= 2:
        return "3class"

    return "unknown"


def canonicalize_label(val: Any) -> str:
    v = _norm_str(val)
    if v == "":
        return ""

    # numeric? keep as numeric string for possible star handling
    if re.fullmatch(r"\d(\.\d+)?", v):
        return v

    if v in {"neg", "negative"}:
        return "neg"
    if v in {"neu", "neutral"}:
        return "neu"
    if v in {"pos", "positive"}:
        return "pos"

    return v


def map_star_to_sentiment(stars: Any, neutral_is_three: bool = True) -> str:
    try:
        s = float(stars)
    except Exception:
        return ""
    if s <= 2:
        return "neg"
    if neutral_is_three and abs(s - 3.0) < 1e-9:
        return "neu"
    if s >= 4:
        return "pos"
    return "neu"


# =========================================
# 2) Profiles (optional)
# =========================================
def list_profiles(profile_dir: Path = PROFILES_DIR) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if profile_dir.is_dir():
        for p in profile_dir.glob("*.json"):
            out[p.name] = str(p)
    return out


def load_profile_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def profile_from_json(name: str, j: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "tau": float(j.get("tau", 0.20)),
        "kappa": float(j.get("kappa", 0.00)),
        "delta_mean": float(j.get("delta_mean", 0.0)),
        "delta_bias": float(j.get("delta_bias", 0.0)),
        "delta_scale": float(j.get("delta_scale", 1.0)),
        "adapter_path": j.get("adapter_path", None),
    }


# =========================================
# 3) HAC-VT wrappers (single place to adapt API)
# =========================================
def hacvt_predict_explain(
    text: str,
    tau: float,
    kappa: float,
    delta_mean: float,
    delta_bias: float,
    delta_scale: float,
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Signature-safe call:
    - If v2 accepts adapter_path, pass it.
    - Else if v2 accepts adapter, load JSON if provided and pass adapter.
    - Else call without adapter.

    UI clarity:
    - band_status becomes "inside (|z| ≤ τ)" or "outside (|z| > τ)".
    - gate_status shows pass/fail only when inside the neutral band.
    """
    if not HACVT_IMPORT_OK:
        raise RuntimeError(f"HAC-VT import failed: {HACVT_IMPORT_ERR}")

    lls = log_likelihoods(text)
    best_label, margin = best_label_and_margin(text)

    adapter_obj = None
    if adapter_path:
        try:
            with open(adapter_path, "r", encoding="utf-8") as f:
                adapter_obj = json.load(f)
        except Exception:
            adapter_obj = None

    sig = inspect.signature(predict_one_gated_v2)
    kwargs: Dict[str, Any] = dict(
        text=text,
        tau=tau,
        delta_mean=delta_mean,
        delta_bias=delta_bias,
        delta_scale=delta_scale,
        kappa=kappa,
    )

    if "adapter_path" in sig.parameters:
        kwargs["adapter_path"] = adapter_path
    elif "adapter" in sig.parameters:
        kwargs["adapter"] = adapter_obj

    pred = predict_one_gated_v2(**kwargs)

    # Best-effort delta/z
    ll_neg = ll_pos = delta = z = None
    try:
        ll_neg = float(lls["neg"]) if isinstance(lls, dict) else float(lls[0])
        ll_pos = float(lls["pos"]) if isinstance(lls, dict) else float(lls[2])
        delta = ll_pos - ll_neg
        z = (delta - delta_mean - delta_bias) / (delta_scale if delta_scale != 0 else 1.0)
    except Exception:
        pass

    # ---------- UPDATED: band_status wording with (|z| ≤ τ) / (|z| > τ) ----------
    band_status: Optional[str] = None
    gate_status: str = "n/a"

    if z is not None:
        if -float(tau) <= float(z) <= float(tau):
            band_status = "inside (|z| ≤ τ)"
            gate_status = "pass" if (margin is not None and float(margin) >= float(kappa)) else "fail"
        else:
            band_status = "outside (|z| > τ)"
            gate_status = "n/a"

    reason = "outside_tau_or_not_neutral"
    if band_status is not None and band_status.startswith("inside"):
        reason = "inside_tau_stay_neu_margin_lt_kappa" if gate_status == "fail" else "inside_tau_flip_if_margin_ge_kappa"

    return {
        "text": text,
        "pred": pred,
        "best_label": best_label,
        "margin": float(margin) if margin is not None else None,
        "lls": lls,
        "ll_neg": ll_neg,
        "ll_pos": ll_pos,
        "delta": delta,
        "z": z,
        "band_status": band_status,
        "gate_status": gate_status,
        "reason": reason,
    }


def hacvt_batch_predict(texts: List[str], params: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in texts:
        out = hacvt_predict_explain(
            text=t,
            tau=params["tau"],
            kappa=params["kappa"],
            delta_mean=params["delta_mean"],
            delta_bias=params["delta_bias"],
            delta_scale=params["delta_scale"],
            adapter_path=params.get("adapter_path", None),
        )
        rows.append(
            {
                "pred_label": out["pred"],
                "band_status": out["band_status"],
                "gate_status": out["gate_status"],
                "reason": out["reason"],
                "margin": out["margin"],
                "z": out["z"],
                "delta": out["delta"],
            }
        )
    return pd.DataFrame(rows)


def calibrate_tau_on_dev(
    dev_texts: List[str],
    dev_labels: List[str],
    base_params: Dict[str, Any],
    tau_grid: np.ndarray,
) -> Tuple[float, float]:
    """
    Scan tau to maximize macro-F1 on dev. Keeps other params fixed.
    Returns: (best_tau, best_macro_f1)
    """
    y_true = np.array(dev_labels)
    best_tau = float(base_params["tau"])
    best_f1 = -1.0

    for tau in tau_grid:
        params = dict(base_params)
        params["tau"] = float(tau)
        preds = hacvt_batch_predict(dev_texts, params)["pred_label"].values
        f1 = f1_score(y_true, preds, labels=CANON, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)

    return best_tau, float(best_f1)


# =========================================
# 4) Existing CSV loaders (Render-safe paths)
# =========================================
@st.cache_data
def load_metrics():
    return pd.read_csv(DATA_DIR / "metrics_summary.csv")


@st.cache_data
def load_confusions():
    cm_vader = pd.read_csv(DATA_DIR / "confusion_vader.csv", index_col=0)
    cm_textblob = pd.read_csv(DATA_DIR / "confusion_textblob.csv", index_col=0)
    cm_hacvt = pd.read_csv(DATA_DIR / "confusion_hacvt.csv", index_col=0)
    return cm_vader, cm_textblob, cm_hacvt


@st.cache_data
def load_hacvt_data():
    delta_df = pd.read_csv(DATA_DIR / "delta_scores_hacvt.csv")
    examples_df = pd.read_csv(DATA_DIR / "hacvt_examples.csv")
    return delta_df, examples_df


@st.cache_data
def list_local_datasets():
    """
    List all CSV files in dashboard/data/ treated as datasets
    (excluding metrics/confusions/delta/examples).
    """
    if not DATA_DIR.is_dir():
        return {}

    all_csv = [p.name for p in DATA_DIR.glob("*.csv")]
    exclude = {
        "metrics_summary.csv",
        "confusion_vader.csv",
        "confusion_textblob.csv",
        "confusion_hacvt.csv",
        "delta_scores_hacvt.csv",
        "hacvt_examples.csv",
    }
    dataset_files = [f for f in all_csv if f not in exclude]
    return {name: str(DATA_DIR / name) for name in dataset_files}


# =========================================
# 5) Sidebar controls
# =========================================
st.sidebar.header("HAC-VT Controls")

if not HACVT_IMPORT_OK:
    st.sidebar.error("HAC-VT import failed.")
    st.sidebar.code(HACVT_IMPORT_ERR)
else:
    st.sidebar.success("HAC-VT import OK")

profiles = list_profiles(PROFILES_DIR)
profile_choice = st.sidebar.selectbox(
    "Profile (optional)", options=["(Default built-in)"] + list(profiles.keys())
)

params: Dict[str, Any] = {
    "name": "default",
    "tau": 0.20,
    "kappa": 0.00,
    "delta_mean": 0.0,
    "delta_bias": 0.0,
    "delta_scale": 1.0,
    "adapter_path": None,
}

if profile_choice != "(Default built-in)":
    try:
        pj = load_profile_json(profiles[profile_choice])
        params.update(profile_from_json(profile_choice, pj))
    except Exception as e:
        st.sidebar.warning(f"Could not load profile: {e}")

st.sidebar.subheader("Neutral behaviour")
params["tau"] = st.sidebar.slider("τ (neutral band width)", 0.00, 2.00, float(params["tau"]), 0.01)
params["kappa"] = st.sidebar.slider("κ (margin gate for leaving neutral)", 0.00, 5.00, float(params["kappa"]), 0.01)

st.sidebar.subheader("Tier-1 standardisation")
params["delta_mean"] = st.sidebar.number_input("delta_mean", value=float(params["delta_mean"]))
params["delta_bias"] = st.sidebar.number_input("delta_bias", value=float(params["delta_bias"]))
params["delta_scale"] = st.sidebar.number_input("delta_scale", value=float(params["delta_scale"]), min_value=1e-9)

st.sidebar.subheader("Adapter (optional)")
adapter_override = st.sidebar.text_input("adapter_path (optional JSON)", value=params["adapter_path"] or "")
params["adapter_path"] = adapter_override.strip() if adapter_override.strip() else None

mode = st.sidebar.radio(
    "Mode",
    options=["Decision mode (default)", "Evaluation mode (calibrate τ if labels exist)"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Decision mode for general use. Evaluation mode when labels exist and you want Macro-F1 alignment.")


# =========================================
# 6) Load saved outputs if present
# =========================================
data_ok = DATA_DIR.is_dir()

metrics_df = None
cm_vader = cm_textblob = cm_hacvt = None
delta_df = examples_df = None
local_datasets: Dict[str, str] = {}

if data_ok:
    try:
        metrics_df = load_metrics()
    except Exception:
        metrics_df = None

    try:
        cm_vader, cm_textblob, cm_hacvt = load_confusions()
    except Exception:
        cm_vader = cm_textblob = cm_hacvt = None

    try:
        delta_df, examples_df = load_hacvt_data()
        examples_df = examples_df.copy()
        if "true_label" in examples_df.columns and "pred_label" in examples_df.columns:
            examples_df["correct"] = examples_df["true_label"] == examples_df["pred_label"]
    except Exception:
        delta_df = examples_df = None

    try:
        local_datasets = list_local_datasets()
    except Exception:
        local_datasets = {}


# =========================================
# 7) Tabs
# =========================================
tab_metrics, tab_confusion, tab_explain, tab_live, tab_data = st.tabs(
    [
        "Metrics Overview",
        "Confusion Matrices",
        "Explainability (Saved HAC-VT)",
        "Live HAC-VT (Predict + Explain)",
        "Dataset Analysis",
    ]
)


# =====================================
# TAB 1 — METRICS OVERVIEW
# =====================================
with tab_metrics:
    st.markdown("This tab shows the summary metrics from `dashboard/data/metrics_summary.csv` (if present).")

    if metrics_df is None:
        st.warning("metrics_summary.csv not found (or could not be read).")
    else:
        st.subheader("Global Metrics Table")

        numeric_cols = metrics_df.select_dtypes(include="number").columns
        format_dict = {col: "{:.3f}" for col in numeric_cols}
        st.dataframe(metrics_df.style.format(format_dict), use_container_width=True)

        if "model" in metrics_df.columns and "macro_f1" in metrics_df.columns:
            st.subheader("Macro-F1 by Model")
            fig, ax = plt.subplots()
            ax.bar(metrics_df["model"], metrics_df["macro_f1"])
            ax.set_xlabel("Model")
            ax.set_ylabel("Macro-F1")
            ax.set_title("Overall Macro-F1 Comparison")
            st.pyplot(fig)
        else:
            st.info("metrics_summary.csv is missing 'model' or 'macro_f1' columns.")


# =====================================
# TAB 2 — CONFUSION MATRICES
# =====================================
with tab_confusion:
    st.subheader("Confusion Matrices (Saved)")

    if cm_vader is None or cm_textblob is None or cm_hacvt is None:
        st.warning("Confusion matrix CSVs not found (or could not be read).")
    else:
        st.markdown("Rows = true labels, columns = predicted labels.")

        confusion_dict = {"VADER": cm_vader, "TextBlob": cm_textblob, "HAC-VT": cm_hacvt}
        model_choice = st.selectbox("Choose a model:", ["VADER", "TextBlob", "HAC-VT"], index=2)
        cm = confusion_dict[model_choice]

        st.write(f"### Confusion Matrix: {model_choice}")
        st.dataframe(cm, use_container_width=True)

        fig2, ax2 = plt.subplots()
        ax2.imshow(cm.values, aspect="auto")
        ax2.set_xticks(range(len(cm.columns)))
        ax2.set_yticks(range(len(cm.index)))
        ax2.set_xticklabels(cm.columns)
        ax2.set_yticklabels(cm.index)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        ax2.set_title(f"{model_choice} Confusion Matrix")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, int(cm.values[i, j]), ha="center", va="center")

        st.pyplot(fig2)


# =====================================
# TAB 3 — EXPLAINABILITY (Saved HAC-VT)
# =====================================
with tab_explain:
    st.subheader("Saved Δ-Score Distribution and Explorer (from your pipeline outputs)")

    if delta_df is None or examples_df is None:
        st.warning("Saved HAC-VT explainability files not found (delta_scores_hacvt.csv / hacvt_examples.csv).")
    else:
        tau_low = tau_high = None
        if metrics_df is not None and "tau_low" in metrics_df.columns and "tau_high" in metrics_df.columns:
            try:
                row = metrics_df[metrics_df["model"] == "HAC-VT"] if "model" in metrics_df.columns else pd.DataFrame()
                if not row.empty:
                    if pd.notna(row["tau_low"].iloc[0]):
                        tau_low = float(row["tau_low"].iloc[0])
                    if pd.notna(row["tau_high"].iloc[0]):
                        tau_high = float(row["tau_high"].iloc[0])
            except Exception:
                pass

        col_left, col_right = st.columns([2, 1])

        with col_left:
            fig3, ax3 = plt.subplots()
            if "delta" in delta_df.columns:
                ax3.hist(delta_df["delta"], bins=40)
                ax3.set_xlabel("Δ score (LL_pos − LL_neg)")
                ax3.set_ylabel("Count")
                ax3.set_title("HAC-VT Δ-Score Distribution")

                if tau_low is not None:
                    ax3.axvline(tau_low, linestyle="--", label="τ_low")
                if tau_high is not None:
                    ax3.axvline(tau_high, linestyle="--", label="τ_high")
                if tau_low is not None or tau_high is not None:
                    ax3.legend()

                st.pyplot(fig3)
            else:
                st.warning("delta_scores_hacvt.csv does not contain a 'delta' column.")

        with col_right:
            st.markdown(
                """
                **Interpretation**
                - Δ far from zero ⇒ strong positive/negative evidence.
                - Δ values between τ_low and τ_high ⇒ neutral decisions (saved calibration).
                """
            )

        st.markdown("---")
        st.subheader("HAC-VT Explainability Explorer (Saved Examples)")

        labels_options = ["all", "neg", "neu", "pos"]
        filter_true = st.selectbox("Filter by true label", labels_options, index=0)
        filter_pred = st.selectbox("Filter by predicted label", labels_options, index=0)
        show_errors_only = st.checkbox("Show only misclassified reviews", value=False)

        filtered = examples_df.copy()
        if "true_label" in filtered.columns and filter_true != "all":
            filtered = filtered[filtered["true_label"] == filter_true]
        if "pred_label" in filtered.columns and filter_pred != "all":
            filtered = filtered[filtered["pred_label"] == filter_pred]
        if show_errors_only and "correct" in filtered.columns:
            filtered = filtered[filtered["correct"] == False]

        st.write(f"{len(filtered)} samples after filtering.")

        if len(filtered) > 0:
            idx = st.slider("Select example index", 0, len(filtered) - 1, 0)
            row = filtered.iloc[idx]

            st.markdown("### Review Text")
            if "text" in filtered.columns:
                st.write(row.get("text", ""))
            else:
                st.write(row.to_dict())

            if "true_label" in filtered.columns and "pred_label" in filtered.columns:
                st.markdown("### Labels")
                st.write(f"- True label: **{row['true_label']}**")
                st.write(f"- Predicted (HAC-VT): **{row['pred_label']}**")
                if "correct" in filtered.columns:
                    st.write(f"- Correctly classified: **{bool(row['correct'])}**")

            if "delta" in row.index:
                st.markdown("### HAC-VT Decision")
                try:
                    st.write(f"- Δ score: `{float(row['delta']):.4f}`")
                except Exception:
                    st.write(f"- Δ score: `{row['delta']}`")

            if "explanation" in filtered.columns and isinstance(row.get("explanation", ""), str) and row["explanation"].strip():
                st.markdown("### Explanation (Token Contributions)")
                st.write(row["explanation"])
            else:
                st.info("No token-level explanation stored for this sample.")
        else:
            st.warning("No samples match the current filters.")


# =====================================
# TAB 4 — LIVE HAC-VT (Predict + Explain)
# =====================================
with tab_live:
    st.subheader("Live HAC-VT (Type text + Analyse)")

    if not HACVT_IMPORT_OK:
        st.error("HAC-VT is not importable in this environment.")
        st.code(HACVT_IMPORT_ERR)
    else:
        st.markdown("### 1) Type a text and analyse (no dataset needed)")

        default_samples = [
            "The car is okay.",
            "It does the job.",
            "Average experience.",
            "Not good at all.",
            "I love this car, but the mileage is terrible.",
        ]

        sample_pick = st.selectbox("Quick sample (optional)", ["(Type my own)"] + default_samples, index=0)
        single_text = sample_pick if sample_pick != "(Type my own)" else ""

        single_text = st.text_area(
            "Enter text",
            value=single_text,
            height=140,
            placeholder="Type any review/comment here...",
        )

        colA, colB = st.columns([1, 1])

        with colA:
            if st.button("Analyse text now", key="live_explain_btn_top"):
                if not single_text.strip():
                    st.warning("Please type a text first.")
                else:
                    out = hacvt_predict_explain(
                        text=single_text,
                        tau=params["tau"],
                        kappa=params["kappa"],
                        delta_mean=params["delta_mean"],
                        delta_bias=params["delta_bias"],
                        delta_scale=params["delta_scale"],
                        adapter_path=params.get("adapter_path", None),
                    )

                    st.success(f"Prediction: {out['pred']}")
                    st.write("**Decision details**")
                    st.write(f"- best_label (raw): {out['best_label']}")
                    st.write(f"- margin: {out['margin']}")
                    st.write(f"- z: {out['z']}")
                    st.write(f"- band_status: {out['band_status']}" if out["band_status"] else "- band_status: n/a")
                    st.write(f"- gate_status: {out['gate_status']}")
                    st.write(f"- reason: {out['reason']}")

        with colB:
            st.write("**Evidence (log-likelihoods)**")
            if st.button("Show LLs", key="show_lls_btn"):
                if not single_text.strip():
                    st.warning("Please type a text first.")
                else:
                    out_ll = hacvt_predict_explain(
                        text=single_text,
                        tau=params["tau"],
                        kappa=params["kappa"],
                        delta_mean=params["delta_mean"],
                        delta_bias=params["delta_bias"],
                        delta_scale=params["delta_scale"],
                        adapter_path=params.get("adapter_path", None),
                    )
                    st.json(out_ll["lls"])

        st.markdown("---")
        st.markdown("### 2) Dataset mode (optional): upload CSV for batch predictions")

        source = st.radio("Dataset source", ["Upload CSV", "Load from dashboard/data"], index=0, key="live_src")

        live_df = None
        if source == "Upload CSV":
            up = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="live_upload")
            if up is not None:
                live_df = pd.read_csv(up)
        else:
            if not DATA_DIR.is_dir():
                st.warning("No dashboard/data folder found.")
            else:
                candidates = [p.name for p in DATA_DIR.glob("*.csv")]
                if not candidates:
                    st.warning("No CSV files found in dashboard/data.")
                else:
                    pick = st.selectbox("Select a CSV from dashboard/data", candidates, key="live_local_pick")
                    live_df = pd.read_csv(DATA_DIR / pick)

        if live_df is None:
            st.info("Dataset mode is optional. Upload a CSV only if you want batch predictions/evaluation.")
        else:
            st.write("Preview")
            st.dataframe(live_df.head(20), use_container_width=True)

            default_text = guess_text_column(live_df)
            default_label = guess_label_column(live_df)

            c1, c2, c3 = st.columns([1, 1, 1])

            with c1:
                text_col = st.selectbox(
                    "Text column",
                    options=list(live_df.columns),
                    index=list(live_df.columns).index(default_text),
                    key="live_text_col",
                )

            with c2:
                label_col = st.selectbox(
                    "Label column (optional)",
                    options=["(None)"] + list(live_df.columns),
                    index=(0 if default_label is None else 1 + list(live_df.columns).index(default_label)),
                    key="live_label_col",
                )

            with c3:
                label_policy = st.selectbox(
                    "If labels exist, treat as",
                    options=["Auto-detect", "3-class", "Binary", "Stars (1–5)"],
                    index=0,
                    key="live_label_policy",
                )

            texts = ensure_text(live_df[text_col]).tolist()

            y = None

            if label_col != "(None)":
                detected = detect_label_type(live_df[label_col])

                if label_policy == "3-class":
                    detected = "3class"
                elif label_policy == "Binary":
                    detected = "binary"
                elif label_policy.startswith("Stars"):
                    detected = "stars"

                st.write(f"Detected label type: **{detected}**")

                if detected == "stars":
                    neutral_is_three = st.checkbox("Map 3 stars to Neutral (recommended)", value=True, key="live_neu3")
                    y = live_df[label_col].map(lambda v: map_star_to_sentiment(v, neutral_is_three=neutral_is_three))
                else:
                    y = live_df[label_col].map(canonicalize_label)

                st.write("Label distribution (after mapping):")
                st.write(pd.Series(y).value_counts(dropna=False))

            st.markdown("---")
            st.subheader("Batch prediction")

            max_rows = st.number_input(
                "Max rows to run (speed control)",
                min_value=10,
                max_value=max(10, len(texts)),
                value=min(2000, len(texts)),
                key="live_max_rows",
            )

            run_texts = texts[: int(max_rows)]

            if st.button("Run HAC-VT on dataset", key="live_batch_btn"):
                preds_df = hacvt_batch_predict(run_texts, params)
                out_df = live_df.iloc[: int(max_rows)].copy().reset_index(drop=True)
                out_df = pd.concat([out_df, preds_df.reset_index(drop=True)], axis=1)

                st.write("Predictions preview")
                st.dataframe(out_df.head(50), use_container_width=True)

                st.download_button(
                    "Download predictions CSV",
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name="hacvt_predictions.csv",
                    mime="text/csv",
                )

                if y is not None:
                    y_eval = np.array(pd.Series(y).iloc[: int(max_rows)].tolist())
                    y_pred = preds_df["pred_label"].values

                    mask = np.isin(y_eval, CANON)
                    y_eval = y_eval[mask]
                    y_pred = y_pred[mask]

                    if len(y_eval) == 0:
                        st.warning("No usable canonical labels (neg/neu/pos) available for evaluation after mapping.")
                    else:
                        st.markdown("### Evaluation (labels available)")
                        macro = f1_score(y_eval, y_pred, labels=CANON, average="macro", zero_division=0)
                        st.write(f"**Macro-F1:** {macro:.4f}")

                        cm = confusion_matrix(y_eval, y_pred, labels=CANON)
                        cm_df = pd.DataFrame(
                            cm,
                            index=[f"true_{c}" for c in CANON],
                            columns=[f"pred_{c}" for c in CANON],
                        )
                        st.write("Confusion matrix")
                        st.dataframe(cm_df, use_container_width=True)

                        if mode.startswith("Evaluation mode"):
                            st.markdown("### τ calibration (dev optimisation on Macro-F1)")
                            tmin, tmax = st.slider("τ scan range", 0.0, 2.0, (0.05, 1.0), 0.01, key="tau_range")
                            steps = st.number_input("τ scan steps", min_value=5, max_value=100, value=20, key="tau_steps")
                            tau_grid = np.linspace(float(tmin), float(tmax), int(steps))

                            if st.button("Calibrate τ now", key="tau_cal_btn"):
                                best_tau, best_f1 = calibrate_tau_on_dev(
                                    dev_texts=list(np.array(run_texts)[mask]),
                                    dev_labels=list(y_eval),
                                    base_params=params,
                                    tau_grid=tau_grid,
                                )
                                st.success(f"Best τ = {best_tau:.3f}  |  Dev Macro-F1 = {best_f1:.4f}")
                                st.info("Copy this τ into the sidebar (τ slider) to lock it in for this dataset.")


# =====================================
# TAB 5 — DATASET ANALYSIS
# =====================================
with tab_data:
    st.subheader("Dataset Upload and Analysis")

    st.markdown(
        """
        Choose one of the local datasets discovered in the `dashboard/data/` folder
        or switch to an uploaded CSV for ad-hoc analysis.
        """
    )

    local_datasets = local_datasets or {}
    local_options = list(local_datasets.keys())
    has_local = len(local_options) > 0

    options: List[str] = []
    if has_local:
        options.extend(local_options)
    options.append("Uploaded CSV")

    dataset_choice = st.selectbox("Select dataset source", options=options, index=0)

    uploaded_file = st.file_uploader("Optionally upload a CSV file", type=["csv"], key="analysis_upload")

    data_df = None
    source_label = ""

    if dataset_choice == "Uploaded CSV":
        if uploaded_file is not None:
            data_df = pd.read_csv(uploaded_file)
            source_label = "Uploaded CSV"
            st.info("Using uploaded dataset.")
        else:
            st.warning("Select a CSV file above to use the uploaded dataset.")
    else:
        path = local_datasets.get(dataset_choice)
        if path and os.path.exists(path):
            data_df = pd.read_csv(path)
            source_label = f"Local dataset: {dataset_choice}"
            st.info(f"Using local dataset: `{dataset_choice}` from the dashboard/data folder.")
        else:
            st.error(f"File for dataset `{dataset_choice}` not found in dashboard/data.")

    if data_df is not None:
        st.markdown("### Basic Information")
        st.write(f"- Source: **{source_label}**")
        st.write(f"- Rows: **{len(data_df)}**")
        st.write(f"- Columns: **{len(data_df.columns)}**")

        st.markdown("### Preview")
        st.dataframe(data_df.head(30), use_container_width=True)

        st.markdown("### Column Summary")
        st.write(data_df.describe(include="all").transpose())

        st.markdown("### Label / Rating Distribution")
        col_name = st.selectbox(
            "Select the label / rating column",
            options=list(data_df.columns),
            index=0,
            key="analysis_label_col",
        )

        value_counts = data_df[col_name].value_counts(dropna=False).sort_index()
        st.write(value_counts)

        fig4, ax4 = plt.subplots()
        ax4.bar(value_counts.index.astype(str), value_counts.values)
        ax4.set_xlabel(col_name)
        ax4.set_ylabel("Count")
        ax4.set_title(f"Distribution of {col_name}")
        plt.xticks(rotation=45)
        st.pyplot(fig4)

        st.markdown("---")
        st.caption(
            "Note: Live HAC-VT prediction and explainability are available in the 'Live HAC-VT' tab "
            "(with auto-detection, star mapping, τ/κ controls, calibration, and downloads)."
        )
