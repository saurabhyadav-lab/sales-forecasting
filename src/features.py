"""Steps 7-9: Stationarity tests, feature engineering, train/test split."""
import numpy as np
import pandas as pd

from .config import EXOG, SEASON

LAGS = [1, 2, 7, 14, 28]
WINDOWS = [7, 14, 28]


# ---------------------------------------------------------------- Step 7
def stationarity_tests(series, verbose=True):
    """ADF: H0 = non-stationary (want p < 0.05).  KPSS: H0 = stationary (want p > 0.05)."""
    from statsmodels.tsa.stattools import adfuller, kpss
    s = series.dropna()
    adf_p = adfuller(s, autolag="AIC")[1]
    kpss_p = kpss(s, regression="c", nlags="auto")[1]
    stationary = (adf_p < 0.05) and (kpss_p > 0.05)
    if verbose:
        print(f"ADF p={adf_p:.4f} | KPSS p={kpss_p:.4f} -> "
              f"{'stationary' if stationary else 'NON-stationary: difference or transform'}")
    return {"adf_p": adf_p, "kpss_p": kpss_p, "stationary": stationary}


def make_stationary(series, log=True, d=1, D=1):
    """Log transform, then regular (d) and seasonal (D) differencing."""
    s = np.log1p(series) if log else series.copy()
    for _ in range(D):
        s = s.diff(SEASON)
    for _ in range(d):
        s = s.diff()
    return s.dropna()


# ---------------------------------------------------------------- Step 8
def active_exog(df):
    """Exogenous columns present in df that actually vary."""
    return [c for c in EXOG if c in df and df[c].nunique() > 1]


def calendar_features(index):
    f = pd.DataFrame(index=index)
    f["dow"] = index.dayofweek
    f["day"] = index.day
    f["month"] = index.month
    f["quarter"] = index.quarter
    f["weekofyear"] = index.isocalendar().week.astype(int).values
    f["dayofyear"] = index.dayofyear
    f["is_weekend"] = (index.dayofweek >= 5).astype(int)
    f["is_month_start"] = index.is_month_start.astype(int)
    f["is_month_end"] = index.is_month_end.astype(int)
    return f


def make_features(df, target="sales", exog=None):
    """Calendar + exogenous + lag + rolling features.
    Rolling stats use shift(1) so a row never sees its own target (no leakage)."""
    X = calendar_features(df.index)
    exog_cols = exog if exog is not None else active_exog(df)
    for c in exog_cols:
        if c in df:
            X[c] = df[c].values
    s = df[target]
    for l in LAGS:
        X[f"lag_{l}"] = s.shift(l)
    for w in WINDOWS:
        X[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
        X[f"roll_std_{w}"] = s.shift(1).rolling(w).std()
    return X


def make_future_frame(df, horizon, promo=0, holiday=0):
    """Future rows with known-in-advance exogenous values.
    Replace promo/holiday with your real promo calendar when you have one."""
    idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D", name="date")
    fut = pd.DataFrame(index=idx)
    fut["promo"], fut["holiday"], fut["open"] = promo, holiday, 1
    return fut


# ---------------------------------------------------------------- Step 9
def train_test_split_ts(df, test_days):
    """Chronological split -- never shuffle a time series."""
    train, test = df.iloc[:-test_days], df.iloc[-test_days:]
    print(f"Train: {train.index[0].date()} -> {train.index[-1].date()} ({len(train)} rows)")
    print(f"Test : {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} rows)")
    return train, test
