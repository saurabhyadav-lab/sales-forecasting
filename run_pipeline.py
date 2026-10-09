"""End-to-end pipeline: Steps 4-19.

Examples
    python run_pipeline.py                                   # synthetic data, all models
    python run_pipeline.py --source rossmann --store 1       # needs data/raw/train.csv
    python run_pipeline.py --skip lstm prophet               # faster run
    python run_pipeline.py --tune --mlflow                   # tuning + experiment tracking
"""
import argparse
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.data_loader import clean, inspect, load_data
from src.eda import run_eda
from src.features import make_future_frame, stationarity_tests, train_test_split_ts
from src.metrics import evaluate, walk_forward
from src.models import FORECASTERS, fit_sarima, ljung_box, xgb_fc

warnings.filterwarnings("ignore")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "rossmann", "csv", "stock"])
    ap.add_argument("--store", type=int, default=1, help="Rossmann store id")
    ap.add_argument("--csv", help="path for --source csv")
    ap.add_argument("--horizon", type=int, default=C.HORIZON)
    ap.add_argument("--test-days", type=int, default=C.TEST_DAYS)
    ap.add_argument("--cv-folds", type=int, default=3, help="walk-forward folds (0 = skip)")
    ap.add_argument("--skip", nargs="*", default=[], help="models to skip, e.g. lstm prophet")
    ap.add_argument("--tune", action="store_true", help="tune SARIMA (AIC) and XGBoost (Optuna)")
    ap.add_argument("--mlflow", action="store_true", help="log runs to MLflow")
    ap.add_argument("--no-eda", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()

    # ---- Steps 4 & 6: load, inspect, clean
    kw = {"store_id": a.store} if a.source == "rossmann" else {"path": a.csv} if a.source == "csv" else {}
    raw = load_data(a.source, **kw)
    inspect(raw)
    df, _ = clean(raw)
    df.to_csv(C.DATA_PROC / "clean.csv")

    # ---- Step 5: EDA
    if not a.no_eda:
        run_eda(df)

    # ---- Step 7: stationarity
    print("\nStationarity of raw series:"); stationarity_tests(df["sales"])
    print("After seasonal differencing:"); stationarity_tests(df["sales"].diff(C.SEASON))

    # ---- Step 9: split
    train, test = train_test_split_ts(df, a.test_days)
    future_exog = test.drop(columns="sales")

    # ---- Optional Step 16: tuning
    forecasters = dict(FORECASTERS)
    if a.tune:
        from functools import partial
        from src.tuning import tune_sarima_aic, tune_xgb
        order, seasonal = tune_sarima_aic(train["sales"])
        forecasters["sarima"] = partial(forecasters["sarima"], order=order, seasonal_order=seasonal)
        forecasters["xgboost"] = partial(xgb_fc, params=tune_xgb(train))

    # ---- Steps 10-15: fit every model, evaluate on hold-out + walk-forward
    rows, holdout_preds = {}, {}
    for name, fn in forecasters.items():
        if name in a.skip:
            continue
        try:
            print(f"\n>> {name}")
            preds = fn(train, future_exog)
            holdout_preds[name] = preds
            res = {f"{k}": v for k, v in evaluate(test["sales"], preds, train["sales"]).items()}
            if a.cv_folds and name != "lstm":      # LSTM CV is slow; hold-out only
                cv = walk_forward(df, fn, horizon=a.horizon, n_folds=a.cv_folds)
                res.update({f"cv_{k}": v for k, v in cv.items()})
            rows[name] = res
        except ImportError as e:
            print(f"   skipped ({e.name} not installed)")

    table = pd.DataFrame(rows).T.round(3).sort_values("RMSE")
    table.to_csv(C.REPORTS_DIR / "model_comparison.csv")
    print("\n==== Model comparison (hold-out + walk-forward 'cv_') ====")
    print(table.to_string())

    # ---- Step 15: actual vs predicted plot
    fig, ax = plt.subplots(figsize=(13, 5))
    test["sales"].plot(ax=ax, color="black", lw=2, label="actual")
    for n, p in holdout_preds.items():
        ax.plot(test.index, p, label=n, alpha=0.8)
    ax.legend(ncol=3); ax.set_title("Hold-out: actual vs predicted")
    plt.tight_layout(); fig.savefig(C.REPORTS_DIR / "actual_vs_pred.png", dpi=120); plt.close(fig)

    # ---- Step 17: error analysis for the best model
    best = table.index[0]
    err = pd.DataFrame({"abs_err": np.abs(test["sales"].values - holdout_preds[best]),
                        "dow": test.index.day_name(), "promo": test["promo"].values})
    print(f"\n==== Error analysis: {best} ====")
    print("By weekday:\n", err.groupby("dow")["abs_err"].mean().round(1).to_string())
    print("By promo:\n", err.groupby("promo")["abs_err"].mean().round(1).to_string())
    bias = float(np.mean(holdout_preds[best] - test["sales"].values))
    print(f"Mean bias (pred - actual): {bias:.2f}")

    # SARIMA residual diagnostics (Step 11)
    if "sarima" in holdout_preds:
        res = fit_sarima(train["sales"])
        print("\nLjung-Box on SARIMA residuals (p>0.05 = white noise):")
        print(ljung_box(pd.Series(res.resid[C.SEASON + 1:])).round(3).to_string())

    # ---- Step 18: retrain on ALL data and forecast the future with intervals
    fut = make_future_frame(df, a.horizon)
    final = forecasters[best](df, fut)
    sigma = float(np.std(test["sales"].values - holdout_preds[best]))
    out = pd.DataFrame({"forecast": final, "lower": final - 1.96 * sigma,
                        "upper": final + 1.96 * sigma}, index=fut.index).clip(lower=0)
    out.to_csv(C.REPORTS_DIR / "future_forecast.csv")

    fig, ax = plt.subplots(figsize=(13, 5))
    df["sales"].iloc[-120:].plot(ax=ax, label="history")
    out["forecast"].plot(ax=ax, color="red", label=f"forecast ({best})")
    ax.fill_between(out.index, out["lower"], out["upper"], color="red", alpha=0.2, label="95% interval")
    ax.legend(); plt.tight_layout(); fig.savefig(C.REPORTS_DIR / "future_forecast.png", dpi=120); plt.close(fig)

    # ---- Step 19: save artifacts used by the dashboard and API
    sarima_full = fit_sarima(df["sales"])
    sarima_full.save(str(C.MODELS_DIR / "sarima.pkl"))
    joblib.dump({"best_model": best, "sigma": sigma, "last_date": str(df.index[-1].date())},
                C.MODELS_DIR / "meta.joblib")
    json.dump({"best_model": best, "sigma": sigma, "bias": bias,
               "metrics": table.loc[best].to_dict()},
              open(C.REPORTS_DIR / "summary.json", "w"), indent=2)
    print(f"\nBest model: {best}. Artifacts saved to models/ and reports/")

    # ---- Step 23: experiment tracking
    if a.mlflow:
        import mlflow
        mlflow.set_experiment("sales-forecasting")
        for name, row in table.iterrows():
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model": name, "source": a.source, "horizon": a.horizon})
                mlflow.log_metrics({k: float(v) for k, v in row.items() if pd.notna(v)})
        print("Logged to MLflow. View with: mlflow ui")


if __name__ == "__main__":
    main()
