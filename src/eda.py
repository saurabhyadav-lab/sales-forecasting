"""Step 5: Exploratory data analysis. All plots are saved to reports/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import REPORTS_DIR, SEASON


def plot_series(df, name="sales_series.png"):
    fig, ax = plt.subplots(2, 1, figsize=(13, 7))
    df["sales"].plot(ax=ax[0], lw=0.8, title="Daily sales")
    df["sales"].rolling(30).mean().plot(ax=ax[0], color="red", label="30-day mean")
    ax[0].legend()
    df["sales"].resample("MS").sum().plot(ax=ax[1], marker="o", title="Monthly total sales")
    plt.tight_layout(); fig.savefig(REPORTS_DIR / name, dpi=120); plt.close(fig)


def plot_seasonality(df, name="seasonality.png"):
    d = df.copy()
    d["dow"], d["month"] = d.index.day_name(), d.index.month
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    d.groupby("dow")["sales"].mean().reindex(order).plot.bar(ax=ax[0, 0], title="Avg sales by weekday")
    d.groupby("month")["sales"].mean().plot.bar(ax=ax[0, 1], title="Avg sales by month")
    d.groupby("promo")["sales"].mean().plot.bar(ax=ax[1, 0], title="Promo (0/1) vs avg sales")
    d.groupby("holiday")["sales"].mean().plot.bar(ax=ax[1, 1], title="Holiday (0/1) vs avg sales")
    plt.tight_layout(); fig.savefig(REPORTS_DIR / name, dpi=120); plt.close(fig)


def plot_decomposition(df, model="multiplicative", name="decomposition.png"):
    from statsmodels.tsa.seasonal import seasonal_decompose
    y = df["sales"].where(df["sales"] > 0).interpolate() if model == "multiplicative" else df["sales"]
    res = seasonal_decompose(y, model=model, period=SEASON)
    fig = res.plot(); fig.set_size_inches(13, 8)
    plt.tight_layout(); fig.savefig(REPORTS_DIR / name, dpi=120); plt.close(fig)


def plot_acf_pacf(series, lags=40, name="acf_pacf.png"):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(series.dropna(), lags=lags, ax=ax[0])
    plot_pacf(series.dropna(), lags=lags, ax=ax[1], method="ywm")
    plt.tight_layout(); fig.savefig(REPORTS_DIR / name, dpi=120); plt.close(fig)


def run_eda(df):
    plot_series(df)
    plot_seasonality(df)
    plot_decomposition(df)
    plot_acf_pacf(df["sales"].diff(SEASON))   # seasonally differenced
    print(f"EDA plots saved to {REPORTS_DIR}")
