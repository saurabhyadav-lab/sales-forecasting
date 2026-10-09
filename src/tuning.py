"""Step 16: Hyperparameter tuning (SARIMA grid by AIC, XGBoost with Optuna)."""
import itertools
import warnings

import numpy as np
import pandas as pd

from .config import SEASON
from .features import make_features
from .models import _xgb_model, fit_sarima


def tune_sarima_aic(y, p_range=(0, 1, 2), q_range=(0, 1, 2), d=1):
    """Grid-search ARIMA orders; lowest AIC wins."""
    best = (np.inf, None)
    for p, q in itertools.product(p_range, q_range):
        for P, Q in itertools.product((0, 1), (0, 1)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    aic = fit_sarima(y, (p, d, q), (P, 1, Q, SEASON)).aic
                if aic < best[0]:
                    best = (aic, ((p, d, q), (P, 1, Q, SEASON)))
            except Exception:
                continue
    print(f"Best SARIMA by AIC: {best[1]} (AIC={best[0]:.1f})")
    return best[1]


def tune_xgb(train, n_trials=30, n_splits=3):
    """Optuna search with TimeSeriesSplit (respects time order)."""
    import optuna
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import TimeSeriesSplit

    X = make_features(train).dropna()
    y = train.loc[X.index, "sales"]

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        )
        errs = []
        for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
            m = _xgb_model(params).fit(X.iloc[tr], y.iloc[tr])
            errs.append(mean_absolute_error(y.iloc[va], m.predict(X.iloc[va])))
        return float(np.mean(errs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best XGB params:", study.best_params)
    return study.best_params
