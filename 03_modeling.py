"""
03_modeling.py

Merged logic:
- Keep ETS and ARIMA from the teammate R script.
- Keep baseline rolling volatility from both versions.
- Keep GARCH from the notebook.
- Upgrade RF using notebook-style time-series tuning with Optuna.

This script standardises everything on one 80/20 holdout period so the
portfolio layer receives one consistent predictions table for all models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import optuna
import pandas as pd
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from config import (
    SPY_MODEL_DATASET_FILE,
    TLT_MODEL_DATASET_FILE,
    OUTPUT_DIR,
    PREDICTIONS_FILE,
    MODEL_METRICS_FILE,
    WARNING_METRICS_FILE,
    FEATURE_IMPORTANCE_FILE,
    BEST_PARAMS_FILE,
    RANDOM_STATE,
    TRAIN_RATIO,
    RF_TUNING_INNER_SPLITS,
    RF_OPTUNA_TRIALS,
    WARNING_THRESHOLD,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def safe_corr(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) < 2:
        return np.nan
    return float(np.corrcoef(actual, pred)[0, 1])


def load_model_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def train_test_split_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(len(df) * TRAIN_RATIO)
    return df.iloc[:train_size].copy(), df.iloc[train_size:].copy()


def tune_rf_on_train(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    X_train = train_df[feature_cols].values
    y_train = train_df["target_vol_5d"].values
    inner_cv = TimeSeriesSplit(n_splits=RF_TUNING_INNER_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
        scores = []
        for tr_idx, va_idx in inner_cv.split(X_train):
            model = RandomForestRegressor(
                **params,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(X_train[tr_idx], y_train[tr_idx])
            pred = model.predict(X_train[va_idx])
            scores.append(rmse(y_train[va_idx], pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=RF_OPTUNA_TRIALS, show_progress_bar=False)
    return study.best_params


def fit_predict_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, RandomForestRegressor]:
    best_params = tune_rf_on_train(train_df, feature_cols)
    model = RandomForestRegressor(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train_df[feature_cols].values, train_df["target_vol_5d"].values)
    pred = model.predict(test_df[feature_cols].values)
    return pred, model


def fit_predict_baseline(test_df: pd.DataFrame) -> np.ndarray:
    return test_df["vol_5d"].values.copy()


def fit_predict_ets(full_df: pd.DataFrame, train_size: int) -> np.ndarray:
    series = full_df["vol_5d"].values
    n = len(series)
    preds = np.full(n - train_size, np.nan)
    for i in range(n - train_size):
        current_index = train_size + i
        y_train = pd.Series(series[:current_index]).dropna()
        try:
            fit = ExponentialSmoothing(
                y_train,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            forecast = fit.forecast(5)
            preds[i] = float(forecast.iloc[-1])
        except Exception:
            preds[i] = np.nan
    return preds


def fit_predict_arima(full_df: pd.DataFrame, train_size: int) -> np.ndarray:
    series = full_df["vol_5d"].values
    n = len(series)
    preds = np.full(n - train_size, np.nan)
    # Keep simple and robust. This mirrors teammate logic but avoids pmdarima dependency.
    candidate_orders = [(1, 0, 0), (1, 0, 1), (2, 0, 1)]
    for i in range(n - train_size):
        current_index = train_size + i
        y_train = pd.Series(series[:current_index]).dropna()
        best_aic = np.inf
        best_fit = None
        for order in candidate_orders:
            try:
                fit = ARIMA(y_train, order=order).fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_fit = fit
            except Exception:
                continue
        if best_fit is None:
            preds[i] = np.nan
        else:
            try:
                forecast = best_fit.forecast(steps=5)
                preds[i] = float(forecast.iloc[-1])
            except Exception:
                preds[i] = np.nan
    return preds


def fit_predict_garch(full_df: pd.DataFrame, train_size: int) -> np.ndarray:
    returns = full_df["return"].values * 100.0
    n = len(returns)
    preds = np.full(n - train_size, np.nan)
    for i in range(n - train_size):
        current_index = train_size + i
        try:
            am = arch_model(
                returns[:current_index],
                vol="Garch",
                p=1,
                q=1,
                mean="Constant",
                dist="normal",
            )
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=5)
            var_5d = fc.variance.iloc[-1].values
            daily_vol = np.sqrt(np.mean(var_5d)) / 100.0
            preds[i] = float(daily_vol * np.sqrt(252))
        except Exception:
            preds[i] = np.nan
    return preds


def warning_metrics(actual: np.ndarray, pred: np.ndarray, model_name: str, asset_name: str) -> Dict[str, float]:
    actual_warn = (actual > WARNING_THRESHOLD).astype(int)
    pred_warn = (pred > WARNING_THRESHOLD).astype(int)
    return {
        "asset": asset_name,
        "model": model_name,
        "precision": precision_score(actual_warn, pred_warn, zero_division=0),
        "recall": recall_score(actual_warn, pred_warn, zero_division=0),
        "f1": f1_score(actual_warn, pred_warn, zero_division=0),
        "actual_warning_rate": actual_warn.mean(),
        "pred_warning_rate": pred_warn.mean(),
    }


def evaluate_asset_models(model_df: pd.DataFrame, asset_name: str, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict], pd.DataFrame]:
    train_df, test_df = train_test_split_time(model_df)
    train_size = len(train_df)

    # Predictions
    baseline_pred = fit_predict_baseline(test_df)
    ets_pred = fit_predict_ets(model_df, train_size)
    arima_pred = fit_predict_arima(model_df, train_size)
    garch_pred = fit_predict_garch(model_df, train_size)
    rf_pred, rf_model = fit_predict_rf(train_df, test_df, feature_cols)

    actual = test_df["target_vol_5d"].values
    pred_df = pd.DataFrame({
        "date": test_df["date"].values,
        f"{asset_name.lower()}_return": test_df["return"].values,
        f"{asset_name.lower()}_next_day_return": test_df["next_day_return"].values,
        f"{asset_name.lower()}_actual_vol": actual,
        f"{asset_name.lower()}_baseline_pred": baseline_pred,
        f"{asset_name.lower()}_ets_pred": ets_pred,
        f"{asset_name.lower()}_arima_pred": arima_pred,
        f"{asset_name.lower()}_garch_pred": garch_pred,
        f"{asset_name.lower()}_rf_pred": rf_pred,
    })

    metrics = []
    warnings = []
    model_map = {
        "Baseline": baseline_pred,
        "ETS": ets_pred,
        "ARIMA": arima_pred,
        "GARCH(1,1)": garch_pred,
        "Random Forest": rf_pred,
    }
    for model_name, pred in model_map.items():
        mask = ~np.isnan(pred)
        act_m = actual[mask]
        pred_m = pred[mask]
        metrics.append({
            "asset": asset_name,
            "model": model_name,
            "n_test": int(mask.sum()),
            "rmse": rmse(act_m, pred_m),
            "mae": mean_absolute_error(act_m, pred_m),
            "correlation": safe_corr(act_m, pred_m),
        })
        warnings.append(warning_metrics(act_m, pred_m, model_name, asset_name))

    feature_importance = pd.DataFrame({
        "asset": asset_name,
        "feature": feature_cols,
        "importance": rf_model.feature_importances_,
    }).sort_values(["asset", "importance"], ascending=[True, False])

    return pred_df, pd.DataFrame(metrics), pd.DataFrame(warnings), [rf_model.get_params()], feature_importance


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spy_df = load_model_df(SPY_MODEL_DATASET_FILE)
    tlt_df = load_model_df(TLT_MODEL_DATASET_FILE)

    feature_cols = [
        c for c in spy_df.columns
        if c not in {"date", "return", "next_day_return", "target_vol_5d"}
    ]

    spy_pred, spy_metrics, spy_warning, spy_params, spy_importance = evaluate_asset_models(
        spy_df, "SPY", feature_cols
    )
    tlt_pred, tlt_metrics, tlt_warning, tlt_params, tlt_importance = evaluate_asset_models(
        tlt_df, "TLT", feature_cols
    )

    predictions = spy_pred.merge(tlt_pred, on="date", how="inner")
    metrics = pd.concat([spy_metrics, tlt_metrics], ignore_index=True)
    warnings = pd.concat([spy_warning, tlt_warning], ignore_index=True)
    feature_importance = pd.concat([spy_importance, tlt_importance], ignore_index=True)

    predictions.to_csv(PREDICTIONS_FILE, index=False)
    metrics.to_csv(MODEL_METRICS_FILE, index=False)
    warnings.to_csv(WARNING_METRICS_FILE, index=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(
            {
                "note": "RF parameters below are the final fitted model params after notebook-style tuning on the train split.",
                "SPY": spy_params[0],
                "TLT": tlt_params[0],
            },
            f,
            indent=2,
            default=str,
        )

    print("=" * 60)
    print("MODELING COMPLETE")
    print("=" * 60)
    print("Saved:")
    print(f"  {PREDICTIONS_FILE}")
    print(f"  {MODEL_METRICS_FILE}")
    print(f"  {WARNING_METRICS_FILE}")
    print(f"  {FEATURE_IMPORTANCE_FILE}")
    print(f"  {BEST_PARAMS_FILE}")
    print("\nModel summary:")
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()