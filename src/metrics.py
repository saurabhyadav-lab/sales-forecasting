"""Step 15: Metrics and walk-forward (rolling-origin) validation."""
import numpy as np
import pandas as pd

from .config import SEASON


def mae(y, p):  return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y, p):
    m = y != 0                                   # skip zero-sales days
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)


def smape(y, p):
    d = (np.abs(y) + np.abs(p)) / 2
    m = d != 0
    return float(np.mean(np.abs(y[m] - p[m]) / d[m]) * 100)


def mase(y, p, y_train, m=SEASON):
    """Error scaled by the in-sample seasonal-naive error. <1 beats seasonal naive."""
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(y - p)) / scale)


def evaluate(y_true, y_pred, y_train):
    y, p, yt = (np.asarray(a, dtype=float) for a in (y_true, y_pred, y_train))
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p),
            "sMAPE": smape(y, p), "MASE": mase(y, p, yt)}


def walk_forward(df, forecaster, horizon=30, n_folds=3):
    """Expanding-window validation. `forecaster(train_df, future_exog_df) -> np.array`.
    Fold k trains on everything before its test window, so there is never leakage."""
    rows, n = [], len(df)
    for k in range(n_folds):
        end = n - (n_folds - 1 - k) * horizon
        train, test = df.iloc[: end - horizon], df.iloc[end - horizon: end]
        preds = forecaster(train, test.drop(columns="sales"))
        rows.append(evaluate(test["sales"], preds, train["sales"]))
    return pd.DataFrame(rows).mean().to_dict()
