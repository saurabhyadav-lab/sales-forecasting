"""Steps 2, 4, 6: Load, inspect and clean the data.

Every loader returns a DataFrame indexed by a daily DatetimeIndex named "date"
with columns: sales, promo, holiday, open.
"""
import numpy as np
import pandas as pd

from .config import DATA_RAW, SEED


# ---------------------------------------------------------------- Step 2
def make_synthetic_sales(start="2021-01-01", end="2024-06-30", seed=SEED):
    """Realistic fake retail data: trend + weekly + yearly seasonality + promos
    + holidays + noise + a few injected outliers and missing days."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="D", name="date")
    t = np.arange(len(idx))

    trend = 200 + 0.15 * t
    weekly = np.array([0.90, 0.88, 0.92, 0.97, 1.10, 1.30, 1.15])[idx.dayofweek]
    yearly = 1 + 0.15 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365.25)
    promo = (rng.random(len(idx)) < 0.12).astype(int)
    holiday = idx.strftime("%m-%d").isin(["01-01", "01-26", "08-15", "10-02", "12-25"]).astype(int)
    # Diwali-like festive bump in Oct/Nov
    festive = 1 + 0.25 * ((idx.month == 11) & (idx.day <= 7))

    sales = trend * weekly * yearly * festive
    sales = sales * (1 + 0.25 * promo) * (1 + 0.15 * holiday)
    sales = sales * rng.normal(1, 0.05, len(idx))

    df = pd.DataFrame({"sales": sales.round(2), "promo": promo,
                       "holiday": holiday, "open": 1}, index=idx)
    # inject messy data: outliers and missing values
    out_pos = rng.choice(len(df), 6, replace=False)
    df.iloc[out_pos, 0] *= rng.choice([3.5, 0.1], 6)
    df.iloc[rng.choice(len(df), 8, replace=False), 0] = np.nan
    df = df.drop(df.index[rng.choice(len(df), 5, replace=False)])  # missing dates
    return df


def load_rossmann(store_id=1, path=DATA_RAW / "train.csv"):
    """Rossmann Store Sales (Kaggle). Place train.csv inside data/raw/."""
    raw = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
    raw = raw[raw["Store"] == store_id].sort_values("Date")
    df = pd.DataFrame({
        "sales": raw["Sales"].values,
        "promo": raw["Promo"].values,
        "holiday": (raw["StateHoliday"].astype(str) != "0").astype(int).values,
        "open": raw["Open"].values,
    }, index=pd.DatetimeIndex(raw["Date"].values, name="date"))
    return df


def load_csv(path, date_col="date", target_col="sales"):
    """Any CSV with a date column and a sales column."""
    raw = pd.read_csv(path, parse_dates=[date_col]).sort_values(date_col)
    df = raw.set_index(date_col).rename(columns={target_col: "sales"})
    df.index.name = "date"
    for c, default in (("promo", 0), ("holiday", 0), ("open", 1)):
        if c not in df:
            df[c] = default
    return df[["sales", "promo", "holiday", "open"]]


def load_stock(ticker="RELIANCE.NS", start="2019-01-01"):
    """Stock alternative (needs internet + yfinance). 'Close' is used as the target."""
    import yfinance as yf
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    close = raw["Close"].squeeze()
    df = pd.DataFrame({"sales": close}, index=pd.DatetimeIndex(close.index, name="date"))
    df["promo"], df["holiday"], df["open"] = 0, 0, 1
    return df


def load_data(source="synthetic", **kw):
    loaders = {"synthetic": make_synthetic_sales, "rossmann": load_rossmann,
               "csv": load_csv, "stock": load_stock}
    return loaders[source](**kw)


# ---------------------------------------------------------------- Step 4
def inspect(df):
    """Shape, dtypes, date range, missing values, frequency gaps."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    info = {
        "shape": df.shape,
        "date_range": (str(df.index.min().date()), str(df.index.max().date())),
        "missing_dates": len(full.difference(df.index)),
        "missing_values": df.isna().sum().to_dict(),
        "zero_sales_days": int((df["sales"] == 0).sum()),
        "duplicated_dates": int(df.index.duplicated().sum()),
    }
    print("---- Data inspection ----")
    for k, v in info.items():
        print(f"{k:>18}: {v}")
    print(df.describe().T[["mean", "std", "min", "max"]].round(2))
    return info


# ---------------------------------------------------------------- Step 6
def detect_outliers(sales, open_flag=None, skip_mask=None, z_thresh=5.0):
    """Robust outlier flag on the log scale. Each value is compared with the median
    of the same weekday in a +-2 week window (promo/holiday days are excluded from
    the median and never flagged, because their spikes are real), then a MAD-based
    z-score is applied."""
    s = sales.where(sales > 0)
    if open_flag is not None:
        s = s.where(open_flag == 1)              # ignore closed days
    base = s if skip_mask is None else s.where(~skip_mask)
    med = base.groupby(base.index.dayofweek).transform(
        lambda x: x.rolling(5, center=True, min_periods=2).median())
    resid = np.log(s) - np.log(med)
    mad = (resid - resid.median()).abs().median()
    z = 0.6745 * (resid - resid.median()) / (mad if mad > 0 else 1)
    flag = z.abs() > z_thresh
    if skip_mask is not None:
        flag &= ~skip_mask
    return flag.fillna(False)


def clean(df, z_thresh=5.0):
    """Fill missing dates/values, handle zeros and outliers. Returns (clean_df, report)."""
    report = {}
    df = df[~df.index.duplicated()].asfreq("D")
    report["missing_sales_before"] = int(df["sales"].isna().sum())

    # Closed days are genuine zeros (store shut), not missing demand -> keep as 0.
    df["open"] = df["open"].fillna(1).astype(int)
    df["promo"] = df["promo"].fillna(0).astype(int)
    df["holiday"] = df["holiday"].fillna(0).astype(int)
    df.loc[df["open"] == 0, "sales"] = 0.0

    # Outliers on open days: flag, then treat as missing (alternative: winsorize)
    special = (df["promo"] == 1) | (df["holiday"] == 1)
    outliers = detect_outliers(df["sales"], df["open"], special, z_thresh)
    report["outliers_removed"] = int(outliers.sum())
    df.loc[outliers, "sales"] = np.nan

    # Interpolate gaps over time (open days only), keep closed days at 0
    open_sales = df["sales"].where(df["open"] == 1)
    open_sales = open_sales.interpolate(method="time", limit_direction="both")
    df["sales"] = np.where(df["open"] == 1, open_sales, 0.0)

    report["missing_sales_after"] = int(df["sales"].isna().sum())
    print("---- Cleaning report ----", report)
    return df, report
