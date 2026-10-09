# Sales Forecasting (SARIMA, Prophet, XGBoost, LSTM)

Forecast daily retail sales with baselines, statistical, machine-learning and deep-learning models,
compare them fairly with walk-forward validation, then serve the best one through a dashboard and an API.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py                                  # synthetic data, runs every step
streamlit run app/streamlit_app.py                      # dashboard
uvicorn app.api:app --reload                            # API docs at /docs
```
Faster run: `python run_pipeline.py --skip lstm prophet`

## Use real data
- **Rossmann**: download `train.csv` from Kaggle into `data/raw/`, then
  `python run_pipeline.py --source rossmann --store 1`
- **Your CSV** (needs `date`, `sales`, optional `promo`, `holiday`, `open`):
  `python run_pipeline.py --source csv --csv data/raw/mine.csv`
- **Stocks**: `python run_pipeline.py --source stock` (edit the ticker in `src/data_loader.py`)

## Step to file map
| Steps | What | File |
|---|---|---|
| 1-3 | Problem definition, config, folders | `src/config.py` |
| 2, 4, 6 | Load, inspect, clean (gaps, zeros, outliers) | `src/data_loader.py` |
| 5 | EDA plots | `src/eda.py` |
| 7-9 | Stationarity tests, features, chronological split | `src/features.py` |
| 10-14 | Baselines, SARIMA, Prophet, XGBoost, LSTM | `src/models.py` |
| 15 | Metrics and walk-forward validation | `src/metrics.py` |
| 16 | Tuning (AIC grid, Optuna) | `src/tuning.py` (`--tune`) |
| 17-19 | Error analysis, final forecast, save model | `run_pipeline.py` |
| 20 | Dashboard | `app/streamlit_app.py` |
| 21 | REST API | `app/api.py` |
| 22 | Docker | `Dockerfile` |
| 23 | MLflow tracking | `run_pipeline.py --mlflow` |

## Design notes
- Every model shares one interface, `fn(train_df, future_exog_df) -> predictions`, so evaluation is identical for all.
- No leakage: chronological splits, scaler fit on train only, rolling features use `shift(1)`.
- XGBoost and LSTM forecast recursively (each prediction feeds the next day's lags).
- Future `promo`/`holiday` default to 0 in `make_future_frame`; pass your real calendar for better forecasts.
- Prediction intervals use the hold-out residual standard deviation (approximate).
- The API serves SARIMA; the dashboard can run any non-deep model.

## Results (fill in after running)
| Model | MAE | RMSE | MASE |
|---|---|---|---|
| seasonal_naive | | | |
| sarima | | | |
| xgboost | | | |

Business impact: _e.g. "XGBoost cut forecast error by X% vs seasonal naive, supporting better stock planning."_
