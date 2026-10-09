"""Step 21: FastAPI service.  Run:  uvicorn app.api:app --reload
Then open http://127.0.0.1:8000/docs"""
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Sales Forecast API")

_model = SARIMAXResults.load(str(ROOT / "models" / "sarima.pkl"))
_meta = joblib.load(ROOT / "models" / "meta.joblib")


@app.get("/health")
def health():
    return {"status": "ok", "last_training_date": _meta["last_date"]}


@app.get("/forecast")
def forecast(horizon: int = 14):
    if not 1 <= horizon <= 90:
        raise HTTPException(400, "horizon must be between 1 and 90")
    f = _model.get_forecast(horizon).summary_frame(alpha=0.05)
    return [{"date": d.strftime("%Y-%m-%d"),
             "forecast": round(max(r["mean"], 0), 2),
             "lower": round(max(r["mean_ci_lower"], 0), 2),
             "upper": round(r["mean_ci_upper"], 2)} for d, r in f.iterrows()]
