"""Steps 10-14: Baselines, SARIMA, Prophet, XGBoost and LSTM.

Every forecaster has the same signature:
    forecaster(train_df, future_df) -> np.ndarray of len(future_df)
where train_df has the 'sales' column and future_df has only the exogenous columns.
That lets the evaluation code treat all models identically.
"""
from functools import partial

import numpy as np
import pandas as pd

from .config import SEASON, SEED
from .features import LAGS, WINDOWS, active_exog, make_features


# ---------------------------------------------------------------- Step 10: baselines
def naive_fc(train, future):
    return np.repeat(train["sales"].iloc[-1], len(future)).astype(float)


def seasonal_naive_fc(train, future, m=SEASON):
    return np.resize(train["sales"].values[-m:], len(future)).astype(float)


def moving_avg_fc(train, future, window=SEASON):
    return np.repeat(train["sales"].iloc[-window:].mean(), len(future))


# ---------------------------------------------------------------- Step 11: SARIMA
def fit_sarima(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, SEASON)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = y.asfreq("D")
    return SARIMAX(y, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def sarima_fc(train, future, order=(1, 1, 1), seasonal_order=(1, 1, 1, SEASON)):
    res = fit_sarima(train["sales"], order, seasonal_order)
    return np.asarray(res.forecast(len(future)))


def auto_arima_order(y, seasonal=True):
    """Pick (p,d,q)(P,D,Q,m) automatically with pmdarima."""
    import pmdarima as pm
    m = pm.auto_arima(y, seasonal=seasonal, m=SEASON, stepwise=True,
                      suppress_warnings=True, error_action="ignore")
    print("auto_arima chose:", m.order, m.seasonal_order)
    return m.order, m.seasonal_order


def ljung_box(residuals, lags=(7, 14, 28)):
    """p > 0.05 => residuals look like white noise (good)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    return acorr_ljungbox(residuals.dropna(), lags=list(lags), return_df=True)


# ---------------------------------------------------------------- Step 12: Prophet
def prophet_fc(train, future, **prophet_kw):
    from prophet import Prophet
    regs = active_exog(train)
    m = Prophet(weekly_seasonality=True, yearly_seasonality="auto",
                daily_seasonality=False, seasonality_mode="multiplicative", **prophet_kw)
    for r in regs:
        m.add_regressor(r)
    dfp = train[["sales"] + regs].reset_index().rename(columns={"date": "ds", "sales": "y"})
    m.fit(dfp)
    fut = future[regs].reset_index().rename(columns={"date": "ds"})
    fc = m.predict(fut)
    prophet_fc.last_model, prophet_fc.last_forecast = m, fc   # for component plots
    return fc["yhat"].values


# ---------------------------------------------------------------- Step 13: XGBoost
def _xgb_model(params=None):
    from xgboost import XGBRegressor
    p = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
             colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
    p.update(params or {})
    return XGBRegressor(**p)


def xgb_fc(train, future, params=None, return_model=False):
    """Train on lag/calendar features, forecast recursively (each prediction
    becomes a lag input for the next day)."""
    full = pd.concat([train, future.assign(sales=np.nan)])
    exog = active_exog(full)
    X = make_features(full, exog=exog)
    Xtr = X.loc[train.index].dropna()
    model = _xgb_model(params).fit(Xtr, train.loc[Xtr.index, "sales"])

    tail = max(max(LAGS), max(WINDOWS)) + 5
    preds = []
    for d in future.index:
        hist = full.loc[:d].tail(tail)
        p = float(model.predict(make_features(hist, exog=exog).iloc[[-1]])[0])
        full.loc[d, "sales"] = p
        preds.append(p)
    preds = np.array(preds)
    return (preds, model, Xtr.columns.tolist()) if return_model else preds


# ---------------------------------------------------------------- Step 14: LSTM
def _lstm_matrix(df):
    """Feature matrix for the LSTM: [sales, exog..., dow_sin, dow_cos]."""
    cols = [df["sales"].values.reshape(-1, 1)]
    for c in active_exog(df):
        cols.append(df[c].values.reshape(-1, 1))
    dow = df.index.dayofweek.values
    cols.append(np.sin(2 * np.pi * dow / 7).reshape(-1, 1))
    cols.append(np.cos(2 * np.pi * dow / 7).reshape(-1, 1))
    return np.hstack(cols).astype("float32")


def lstm_fc(train, future, window=28, epochs=60, batch_size=32, units=64, seed=SEED):
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    tf.keras.utils.set_random_seed(seed)

    n_train = len(train)
    full = pd.concat([train, future.assign(sales=np.nan)])
    A = _lstm_matrix(full)

    # Scaler is fit on TRAIN sales only (no leakage). NaNs in the future pass through.
    scaler = MinMaxScaler().fit(train[["sales"]])
    A[:, 0] = scaler.transform(full[["sales"]]).ravel()

    # Sliding windows: last `window` days -> next day
    Xw = np.stack([A[i - window:i] for i in range(window, n_train)])
    yw = A[window:n_train, 0]
    split = int(len(Xw) * 0.9)                  # chronological validation split
    n_feat = A.shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Input((window, n_feat)),
        tf.keras.layers.LSTM(units, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units // 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    stop = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    model.fit(Xw[:split], yw[:split], validation_data=(Xw[split:], yw[split:]),
              epochs=epochs, batch_size=batch_size, callbacks=[stop], verbose=0)

    # Recursive multi-step forecast
    for i in range(n_train, len(full)):
        p = model.predict(A[i - window:i][None], verbose=0)[0, 0]
        A[i, 0] = p
    return scaler.inverse_transform(A[n_train:, [0]]).ravel()


# ---------------------------------------------------------------- registry
FORECASTERS = {
    "naive": naive_fc,
    "seasonal_naive": seasonal_naive_fc,
    "moving_avg": moving_avg_fc,
    "sarima": sarima_fc,
    "prophet": prophet_fc,
    "xgboost": xgb_fc,
    "lstm": lstm_fc,
}
