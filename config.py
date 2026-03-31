# This file contains all the constants and setting used across the whole project, so nobody hardcodes different values in different scripts.

from pathlib import Path

# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

SPY_FILE = DATA_DIR / "spy5years.csv"
TLT_FILE = DATA_DIR / "TLT_raw.csv"
VIX_FILE = DATA_DIR / "VIX_raw.csv"

MASTER_DATASET_FILE = OUTPUT_DIR / "master_dataset.csv"
SPY_MODEL_DATASET_FILE = OUTPUT_DIR / "spy_model_dataset.csv"
TLT_MODEL_DATASET_FILE = OUTPUT_DIR / "tlt_model_dataset.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.csv"
MODEL_METRICS_FILE = OUTPUT_DIR / "model_metrics.csv"
WARNING_METRICS_FILE = OUTPUT_DIR / "warning_metrics.csv"
PORTFOLIO_RESULTS_FILE = OUTPUT_DIR / "portfolio_results.csv"
PORTFOLIO_KPIS_FILE = OUTPUT_DIR / "portfolio_kpis.csv"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.csv"
BEST_PARAMS_FILE = OUTPUT_DIR / "best_params.json"

# =========================
# General settings
# =========================
DATE_COL = "date"
RANDOM_STATE = 42
TRAIN_RATIO = 0.8

# =========================
# RF settings (prioritise notebook logic)
# =========================
RF_TUNING_INNER_SPLITS = 3
RF_OPTUNA_TRIALS = 30  # reduce if runtime is too slow

# =========================
# Warning / allocation settings
# =========================
WARNING_THRESHOLD = 0.15  # same as teammate R code

# Static benchmark
STATIC_SPY_WEIGHT = 0.60
STATIC_TLT_WEIGHT = 0.40

# Simple upgraded SPY/TLT allocation rule
# Default = 60/40
# If SPY is high risk -> shift 25% from SPY to TLT (35/65)
# If TLT is high risk but SPY is normal -> tilt to 75/25
# If both are high risk -> neutral 50/50
RISK_OFF_SPY_WEIGHT = 0.35
RISK_OFF_TLT_WEIGHT = 0.65

RISK_ON_SPY_WEIGHT = 0.75
RISK_ON_TLT_WEIGHT = 0.25

BOTH_HIGH_SPY_WEIGHT = 0.50
BOTH_HIGH_TLT_WEIGHT = 0.50

TRANSACTION_COST_BPS = 10  # 10 bps = 0.10%
TRADING_DAYS_PER_YEAR = 252
