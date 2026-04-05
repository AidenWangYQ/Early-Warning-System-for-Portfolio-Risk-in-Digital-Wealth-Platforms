"""
03_modeling.py  

Changes from original:
- Added RF + GARCH stacking with linear meta-learner
- Meta-learner trained on out-of-fold predictions (no data leakage)
- Stacked prediction = data-driven combination of RF and GARCH
  that addresses RF's extrapolation ceiling

Models included:
- Baseline rolling volatility
- ETS (Exponential Smoothing)
- ARIMA
- GARCH(1,1)
- Random Forest with Optuna tuning
- Stacked (RF + GARCH, constrained weight optimization)
- Regime-Switch (calibrated RF normally, GARCH when models disagree)
- Calibrated RF (isotonic regression on RF output)  <- FINAL BEST MODEL
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar
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


# ================================================================
# HELPER FUNCTIONS
# ================================================================

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


# ================================================================
# INDIVIDUAL MODELS
# ================================================================

def tune_rf_on_train(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    X_train = train_df[feature_cols].values
    y_train = train_df["target_vol_5d"].values
    inner_cv = TimeSeriesSplit(n_splits=RF_TUNING_INNER_SPLITS)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
        scores = []
        for tr_idx, va_idx in inner_cv.split(X_train):
            model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(X_train[tr_idx], y_train[tr_idx])
            scores.append(rmse(y_train[va_idx], model.predict(X_train[va_idx])))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=RF_OPTUNA_TRIALS, show_progress_bar=False)
    return study.best_params


def fit_predict_rf(train_df, test_df, feature_cols):
    best_params = tune_rf_on_train(train_df, feature_cols)
    model = RandomForestRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(train_df[feature_cols].values, train_df["target_vol_5d"].values)
    pred = model.predict(test_df[feature_cols].values)
    return pred, model, best_params


def fit_predict_baseline(test_df):
    return test_df["vol_5d"].values.copy()


def fit_predict_ets(full_df, train_size):
    series = full_df["vol_5d"].values
    n = len(series)
    preds = np.full(n - train_size, np.nan)
    for i in range(n - train_size):
        y_train = pd.Series(series[: train_size + i]).dropna()
        try:
            fit = ExponentialSmoothing(y_train, trend="add", seasonal=None,
                                       initialization_method="estimated").fit(optimized=True)
            forecast = fit.forecast(5)
            preds[i] = float(forecast.iloc[-1])
        except Exception:
            preds[i] = np.nan
    return preds


def fit_predict_arima(full_df, train_size):
    series = full_df["vol_5d"].values
    n = len(series)
    preds = np.full(n - train_size, np.nan)
    candidate_orders = [(1, 0, 0), (1, 0, 1), (2, 0, 1)]
    for i in range(n - train_size):
        y_train = pd.Series(series[: train_size + i]).dropna()
        best_aic, best_fit = np.inf, None
        for order in candidate_orders:
            try:
                fit = ARIMA(y_train, order=order).fit()
                if fit.aic < best_aic:
                    best_aic, best_fit = fit.aic, fit
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


def fit_predict_garch(full_df, train_size):
    returns = full_df["return"].values * 100.0
    n = len(returns)
    preds = np.full(n - train_size, np.nan)
    for i in range(n - train_size):
        current_index = train_size + i
        try:
            am = arch_model(returns[:current_index], vol="Garch", p=1, q=1,
                           mean="Constant", dist="normal")
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=5)
            var_5d = fc.variance.iloc[-1].values
            daily_vol = np.sqrt(np.mean(var_5d)) / 100.0
            preds[i] = float(daily_vol * np.sqrt(252))
        except Exception:
            preds[i] = np.nan
    return preds


# ================================================================
# STACKING: RF + GARCH WITH LINEAR META-LEARNER  (NEW)
# ================================================================

def fit_predict_stacked(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    rf_test_pred: np.ndarray,
    garch_test_pred: np.ndarray,
    rf_best_params: Dict,
) -> Tuple[np.ndarray, float]:
    """
    Stacked ensemble: RF + GARCH combined by a linear meta-learner.
    
    Why this works:
    - RF is accurate in normal markets (multi-dimensional features)
      but cannot extrapolate beyond training data range
    - GARCH is less accurate day-to-day but has no prediction ceiling
      (parametric formula scales with return magnitude)
    - A linear meta-learner learns the optimal combination from data,
      and being parametric itself, can also extrapolate
    
    To avoid data leakage, the meta-learner is trained on
    OUT-OF-FOLD predictions from the training set:
    - RF: TimeSeriesSplit OOF predictions
    - GARCH: expanding window predictions on calibration period
    
    The meta-learner never sees the test set during training.
    """
    train_size = len(train_df)
    X_train = train_df[feature_cols].values
    y_train = train_df["target_vol_5d"].values
    
    # ------------------------------------------------------------------
    # Step 1: Get RF out-of-fold predictions on training data
    # ------------------------------------------------------------------
    oof_rf = np.full(train_size, np.nan)
    tscv = TimeSeriesSplit(n_splits=3)
    
    for tr_idx, val_idx in tscv.split(X_train):
        rf_temp = RandomForestRegressor(
            **rf_best_params, random_state=RANDOM_STATE, n_jobs=-1
        )
        rf_temp.fit(X_train[tr_idx], y_train[tr_idx])
        oof_rf[val_idx] = rf_temp.predict(X_train[val_idx])
    
    # ------------------------------------------------------------------
    # Step 2: Get GARCH predictions on training data
    # ------------------------------------------------------------------
    # Run expanding window GARCH starting from 50% of training data
    garch_start = train_size // 2
    returns_pct = full_df["return"].values * 100.0
    oof_garch = np.full(train_size, np.nan)
    
    for t in range(garch_start, train_size):
        try:
            am = arch_model(returns_pct[:t], vol="Garch", p=1, q=1,
                           mean="Constant", dist="normal")
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=5)
            var_5d = fc.variance.iloc[-1].values
            daily_vol = np.sqrt(np.mean(var_5d)) / 100.0
            oof_garch[t] = float(daily_vol * np.sqrt(252))
        except Exception:
            oof_garch[t] = np.nan
    
    # ------------------------------------------------------------------
    # Step 3: Find optimal weight where α + β = 1
    # ------------------------------------------------------------------
    # Search for one number w (RF weight), GARCH weight = 1 - w
    # Minimize RMSE on calibration data
    # This is fully data-driven: no human-chosen parameters
    cal_mask = (~np.isnan(oof_rf)) & (~np.isnan(oof_garch))
    
    if cal_mask.sum() < 10:
        print("    [Stacking] Warning: insufficient calibration data, using simple average")
        stacked_pred = 0.5 * rf_test_pred + 0.5 * garch_test_pred
        return stacked_pred, None
    
    rf_cal = oof_rf[cal_mask]
    garch_cal = oof_garch[cal_mask]
    y_cal = y_train[cal_mask]
    
    def objective_weight(w):
        """RMSE of weighted combination on calibration data."""
        pred = w * rf_cal + (1 - w) * garch_cal
        return rmse(y_cal, pred)
    
    result = minimize_scalar(objective_weight, bounds=(0.0, 1.0), method='bounded')
    rf_weight = result.x
    garch_weight = 1.0 - rf_weight
    
    cal_pred = rf_weight * rf_cal + garch_weight * garch_cal
    cal_rmse = rmse(y_cal, cal_pred)
    
    print(f"    [Stacking] Optimal weights: "
          f"{rf_weight:.1%} RF + {garch_weight:.1%} GARCH")
    print(f"    [Stacking] Calibration samples: {cal_mask.sum()}, "
          f"RMSE: {cal_rmse:.4f}")
    
    # ------------------------------------------------------------------
    # Step 4: Apply weights to test predictions
    # ------------------------------------------------------------------
    test_mask = ~np.isnan(garch_test_pred)
    stacked_pred = np.full(len(rf_test_pred), np.nan)
    
    stacked_pred[test_mask] = (
        rf_weight * rf_test_pred[test_mask] +
        garch_weight * garch_test_pred[test_mask]
    )
    
    # For rows where GARCH failed, fall back to RF
    stacked_pred[np.isnan(stacked_pred)] = rf_test_pred[np.isnan(stacked_pred)]
    
    return stacked_pred, rf_weight


# ================================================================
# WARNING METRICS
# ================================================================

def warning_metrics(actual, pred, model_name, asset_name):
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


# ================================================================
# MAIN EVALUATION PIPELINE (PER ASSET)
# ================================================================

def evaluate_asset_models(model_df, asset_name, feature_cols):
    print(f"\n{'=' * 55}")
    print(f"  {asset_name}")
    print(f"{'=' * 55}")
    
    train_df, test_df = train_test_split_time(model_df)
    train_size = len(train_df)
    actual = test_df["target_vol_5d"].values
    
    print(f"  Train: {train_size}, Test: {len(test_df)}")
    
    # --- Individual models ---
    print(f"  Running Baseline...")
    baseline_pred = fit_predict_baseline(test_df)
    
    print(f"  Running ETS (rolling forecast)...")
    ets_pred = fit_predict_ets(model_df, train_size)
    
    print(f"  Running ARIMA (rolling forecast)...")
    arima_pred = fit_predict_arima(model_df, train_size)
    
    print(f"  Running GARCH...")
    garch_pred = fit_predict_garch(model_df, train_size)
    
    print(f"  Running RF + Optuna ({RF_OPTUNA_TRIALS} trials)...")
    rf_pred, rf_model, rf_best_params = fit_predict_rf(train_df, test_df, feature_cols)
    
    # --- Stacked ensemble ---
    print(f"  Running Stacked (RF + GARCH)...")
    stacked_pred, stacked_rf_weight = fit_predict_stacked(
        full_df=model_df,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        rf_test_pred=rf_pred,
        garch_test_pred=garch_pred,
        rf_best_params=rf_best_params,
    )
    
    # --- Isotonic calibration of RF (FINAL MODEL) ---
    # RF has the best ranking ability (highest correlation) but its
    # output scale is compressed because tree-based models average
    # training samples in leaf nodes, squeezing predictions toward
    # the middle. Isotonic regression learns a monotonic mapping from
    # RF's compressed scale back to the correct scale.
    #
    # IMPORTANT: calibration is done on the TRAINING set's validation
    # portion (last 30%), NOT on the test set. This ensures strictly
    # forward-looking methodology with no data leakage.
    print(f"  Running Isotonic Calibration...")
    
    # Get RF predictions on training set's validation portion
    # Use the last 30% of training data as calibration set
    X_train_full = train_df[feature_cols].values
    y_train_full = train_df["target_vol_5d"].values
    cal_start = int(train_size * 0.7)
    
    # Train RF on first 70% of training data, predict on last 30%
    rf_cal_model = RandomForestRegressor(
        **rf_best_params, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_cal_model.fit(X_train_full[:cal_start], y_train_full[:cal_start])
    rf_cal_preds = rf_cal_model.predict(X_train_full[cal_start:])
    y_cal_actual = y_train_full[cal_start:]
    
    # Fit isotonic on training validation set
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(rf_cal_preds, y_cal_actual)
    
    # Apply to test set predictions
    calibrated_rf_pred = iso.predict(rf_pred)
    
    print(f"    Calibration set: {len(y_cal_actual)} training rows (no test data used)")
    print(f"    RF raw range:        [{np.min(rf_pred):.4f}, {np.max(rf_pred):.4f}]")
    print(f"    RF calibrated range: [{np.min(calibrated_rf_pred):.4f}, {np.max(calibrated_rf_pred):.4f}]")
    print(f"    Actual range:        [{np.min(actual):.4f}, {np.max(actual):.4f}]")
    
    # --- RF Uncertainty (for Aiden's confidence-aware adjustment) ---
    # Each of RF's trees gives its own prediction. The std across trees
    # measures how much they disagree. High disagreement = model is
    # uncertain = decision engine should be more conservative.
    print(f"  Computing RF Uncertainty...")
    
    X_test = test_df[feature_cols].values
    all_tree_preds = np.array([tree.predict(X_test) for tree in rf_model.estimators_])
    rf_uncertainty = np.std(all_tree_preds, axis=0)
    
    print(f"    Uncertainty range: [{np.min(rf_uncertainty):.4f}, {np.max(rf_uncertainty):.4f}]")
    print(f"    Mean uncertainty:  {np.mean(rf_uncertainty):.4f}")
    
    # --- Regime-switching: Calibrated RF + GARCH ---
    # Use calibrated RF normally, switch to GARCH when GARCH sees
    # significantly more risk than calibrated RF. After calibration
    # both models are on the same scale, so disagreement is meaningful.
    # Threshold learned from training validation data (same cal set).
    print(f"  Running Regime-Switching (Calibrated RF / GARCH)...")
    
    disagreement = garch_pred - calibrated_rf_pred
    
    # Learn threshold from training validation set
    # Get GARCH predictions on training validation period
    returns_pct_full = model_df["return"].values * 100.0
    garch_cal_preds = np.full(train_size - cal_start, np.nan)
    for i in range(train_size - cal_start):
        t = cal_start + i
        try:
            am = arch_model(returns_pct_full[:t], vol="Garch", p=1, q=1,
                           mean="Constant", dist="normal")
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=5)
            var_5d = fc.variance.iloc[-1].values
            garch_cal_preds[i] = float(np.sqrt(np.mean(var_5d)) / 100.0 * np.sqrt(252))
        except Exception:
            garch_cal_preds[i] = np.nan
    
    # Calibrate the training RF predictions too
    rf_cal_preds_calibrated = iso.predict(rf_cal_preds)
    cal_disagreement = garch_cal_preds - rf_cal_preds_calibrated
    
    cal_valid = ~np.isnan(cal_disagreement)
    if cal_valid.sum() > 10:
        def regime_objective(threshold):
            ensemble = np.where(
                cal_disagreement[cal_valid] > threshold,
                garch_cal_preds[cal_valid],
                rf_cal_preds_calibrated[cal_valid]
            )
            return np.sqrt(np.mean((y_cal_actual[cal_valid] - ensemble)**2))
        
        regime_result = minimize_scalar(regime_objective, bounds=(-0.05, 0.15), method='bounded')
        regime_threshold = regime_result.x
    else:
        regime_threshold = 0.05  # fallback
    
    regime_pred = np.where(disagreement > regime_threshold, garch_pred, calibrated_rf_pred)
    garch_days = (disagreement > regime_threshold).sum()
    
    print(f"    Switch to GARCH when disagreement > {regime_threshold:.4f}")
    print(f"    Calibrated RF days: {len(actual) - garch_days}, GARCH days: {garch_days}")
    
    # --- Build predictions table ---
    prefix = asset_name.lower()
    pred_df = pd.DataFrame({
        "date": test_df["date"].values,
        f"{prefix}_return": test_df["return"].values,
        f"{prefix}_next_day_return": test_df["next_day_return"].values,
        f"{prefix}_actual_vol": actual,
        f"{prefix}_baseline_pred": baseline_pred,
        f"{prefix}_ets_pred": ets_pred,
        f"{prefix}_arima_pred": arima_pred,
        f"{prefix}_garch_pred": garch_pred,
        f"{prefix}_rf_pred": rf_pred,
        f"{prefix}_rf_uncertainty": rf_uncertainty,
        f"{prefix}_stacked_pred": stacked_pred,
        f"{prefix}_regime_pred": regime_pred,
        f"{prefix}_calibrated_rf_pred": calibrated_rf_pred,
    })
    
    # --- Evaluate all models ---
    model_map = {
        "Baseline": baseline_pred,
        "ETS": ets_pred,
        "ARIMA": arima_pred,
        "GARCH(1,1)": garch_pred,
        "Random Forest": rf_pred,
        "Stacked (RF+GARCH)": stacked_pred,
        "Regime-Switch": regime_pred,
        "Calibrated RF": calibrated_rf_pred,
    }
    
    metrics = []
    warnings = []
    for name, pred in model_map.items():
        mask = ~np.isnan(pred)
        if mask.sum() == 0:
            continue
        a, p = actual[mask], pred[mask]
        metrics.append({
            "asset": asset_name, "model": name,
            "n_test": int(mask.sum()),
            "rmse": rmse(a, p), "mae": mean_absolute_error(a, p),
            "correlation": safe_corr(a, p),
        })
        warnings.append(warning_metrics(a, p, name, asset_name))
    
    # --- Feature importance ---
    feature_importance = pd.DataFrame({
        "asset": asset_name,
        "feature": feature_cols,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    # --- Print summary ---
    metrics_df = pd.DataFrame(metrics)
    warn_df = pd.DataFrame(warnings)
    print(f"\n  Results:")
    print(metrics_df[['model', 'rmse', 'mae', 'correlation']].round(4).to_string(index=False))
    print(f"\n  Warning classification:")
    print(warn_df[['model', 'precision', 'recall', 'f1']].round(3).to_string(index=False))
    
    return pred_df, metrics_df, warn_df, [rf_best_params], feature_importance


# ================================================================
# MAIN
# ================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    spy_df = load_model_df(SPY_MODEL_DATASET_FILE)
    tlt_df = load_model_df(TLT_MODEL_DATASET_FILE)
    
    feature_cols = [
        c for c in spy_df.columns
        if c not in {"date", "return", "next_day_return", "target_vol_5d"}
    ]
    
    spy_pred, spy_met, spy_warn, spy_params, spy_imp = evaluate_asset_models(spy_df, "SPY", feature_cols)
    tlt_pred, tlt_met, tlt_warn, tlt_params, tlt_imp = evaluate_asset_models(tlt_df, "TLT", feature_cols)
    
    predictions = spy_pred.merge(tlt_pred, on="date", how="inner")
    metrics = pd.concat([spy_met, tlt_met], ignore_index=True)
    warnings = pd.concat([spy_warn, tlt_warn], ignore_index=True)
    importance = pd.concat([spy_imp, tlt_imp], ignore_index=True)
    
    predictions.to_csv(PREDICTIONS_FILE, index=False)
    metrics.to_csv(MODEL_METRICS_FILE, index=False)
    warnings.to_csv(WARNING_METRICS_FILE, index=False)
    importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
    
    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump({"SPY": spy_params[0], "TLT": tlt_params[0]}, f, indent=2, default=str)
    
    print(f"\n{'=' * 55}")
    print("MODELING COMPLETE")
    print(f"{'=' * 55}")
    print(f"Saved: {PREDICTIONS_FILE} ({len(predictions)} rows)")
    print(f"\nFull model comparison:")
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
