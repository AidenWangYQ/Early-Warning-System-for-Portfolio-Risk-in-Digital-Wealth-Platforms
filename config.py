# This file contains all the constants and setting used across the whole project, so nobody hardcodes different values in different scripts.

# e.g. Structure (Not actual one, update later when necessary)
DATA_DIR = "data/"
OUTPUT_DIR = "outputs/"

SPY_FILE = DATA_DIR + "spy.csv"
TLT_FILE = DATA_DIR + "tlt.csv"
VIX_FILE = DATA_DIR + "vix.csv"

DATE_COL = "date"
SEED = 42

TRAIN_RATIO = 0.8
WARNING_THRESHOLD = 0.15
TARGET_PORTFOLIO_VOL = 0.10

STATIC_SPY_WEIGHT = 0.60
STATIC_TLT_WEIGHT = 0.40

TRANSACTION_COST = 0.001   # 0.1%
REBALANCE_FREQUENCY = "daily"