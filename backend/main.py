from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

# Import HAC-VT module-level API (your model.py exports functions, not a class)
from hacvt import model as hacvt_model

app = FastAPI(title="HAC-VT API", version="1.1")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

    # Dashboard-like controls (defaults match your UI)
    tau: float = 0.2
    kappa: float = 0.0

    # Tier-1 standardisation (v2)
    delta_mean: float = 0.0
    delta_scale: float = 1.0
    delta_bias: float = 0.0

    # Adapter support (optional). For now we keep it off unless you pass something.
    adapter: Optional[Dict[str, Any]] = None


@app.get("/")
def root():
    return {"status": "HAC-VT API running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        # Prefer v2 if available (it is in your stated public API)
        if hasattr(hacvt_model, "predict_one_gated_v2"):
            out = hacvt_model.predict_one_gated_v2(
                text=text,
                tau=req.tau,
                delta_mean=req.delta_mean,
                kappa=req.kappa,
                delta_bias=req.delta_bias,
                delta_scale=req.delta_scale,
                adapter=req.adapter
            )
        else:
            # Fallback to v1 gating if v2 not present
            out = hacvt_model.predict_one_gated(
                text=text,
                tau=req.tau,
                delta_mean=req.delta_mean,
                kappa=req.kappa
            )

        # Also return margin/LL for transparency if available
        details: Dict[str, Any] = {}

        if hasattr(hacvt_model, "best_label_and_margin"):
            best, margin = hacvt_model.best_label_and_margin(text)
            details["best_label"] = best
            details["margin"] = margin

        if hasattr(hacvt_model, "log_likelihoods"):
            ll = hacvt_model.log_likelihoods(text)
            details["log_likelihoods"] = ll

        # out might be a label string or a tuple depending on your implementation
        if isinstance(out, (list, tuple)) and len(out) >= 1:
            label = out[0]
        else:
            label = out

        return {"label": label, "details": details}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
