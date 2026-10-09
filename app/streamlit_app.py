"""Step 20: Streamlit dashboard.  Run:  streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import make_future_frame
from src.models import FORECASTERS

st.set_page_config(page_title="Sales Forecasting", layout="wide")
st.title("Sales Forecasting Dashboard")

data_path = ROOT / "data" / "processed" / "clean.csv"
if not data_path.exists():
    st.error("Run `python run_pipeline.py` first to generate the data and models.")
    st.stop()


@st.cache_data
def load():
    return pd.read_csv(data_path, parse_dates=["date"], index_col="date").asfreq("D")


@st.cache_data(show_spinner="Fitting model...")
def forecast(model_name, horizon):
    df = load()
    fut = make_future_frame(df, horizon)
    return FORECASTERS[model_name](df, fut), fut.index


df = load()
summary_path = ROOT / "reports" / "summary.json"
summary = json.load(open(summary_path)) if summary_path.exists() else {"best_model": "sarima", "sigma": 0}

# ---- sidebar controls
st.sidebar.header("Controls")
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 90, 30)
options = [m for m in ("naive", "seasonal_naive", "sarima", "prophet", "xgboost") if m in FORECASTERS]
default = options.index(summary["best_model"]) if summary["best_model"] in options else 2
model_name = st.sidebar.selectbox("Model", options, index=default)
history_days = st.sidebar.slider("History to show (days)", 30, 365, 120)

preds, idx = forecast(model_name, horizon)
sigma = summary.get("sigma", 0)

# ---- KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Avg forecast / day", f"{preds.mean():,.0f}")
c2.metric(f"Total next {horizon} days", f"{preds.sum():,.0f}")
last = df["sales"].iloc[-horizon:].sum()
c3.metric("vs previous period", f"{(preds.sum() / last - 1) * 100:+.1f}%")

# ---- chart
fig = go.Figure()
hist = df["sales"].iloc[-history_days:]
fig.add_trace(go.Scatter(x=hist.index, y=hist, name="History", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=idx, y=preds, name="Forecast", line=dict(color="red")))
fig.add_trace(go.Scatter(x=list(idx) + list(idx[::-1]),
                         y=list(preds + 1.96 * sigma) + list((preds - 1.96 * sigma)[::-1]),
                         fill="toself", fillcolor="rgba(255,0,0,0.15)", line=dict(width=0),
                         name="95% interval"))
fig.update_layout(height=450, hovermode="x unified", margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)

# ---- tables
left, right = st.columns(2)
cmp_path = ROOT / "reports" / "model_comparison.csv"
if cmp_path.exists():
    left.subheader("Model comparison")
    left.dataframe(pd.read_csv(cmp_path, index_col=0))
right.subheader("Forecast table")
fc = pd.DataFrame({"forecast": preds.round(0)}, index=idx)
right.dataframe(fc)
right.download_button("Download forecast CSV", fc.to_csv().encode(), "forecast.csv")
