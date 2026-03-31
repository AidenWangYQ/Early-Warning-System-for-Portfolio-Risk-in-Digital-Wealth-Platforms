# Early-Warning-System-for-Portfolio-Risk-in-Digital-Wealth-Platforms
This repository contains the data and modelling pipeline on my BC2407 final group project.

## Setup

### 1. Create and activate a virtual environment

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Data

Place the following raw data files inside the `data/` folder:

- `spy.csv`
- `tlt.csv`
- `vix.csv`

### Expected minimum columns

- `date`
- `open`
- `high`
- `low`
- `close`
- `adjusted` or `adj_close`
- `volume`

### For VIX, the main required columns

- `date`
- `close`

Make sure the date column is consistent and parseable.

---

## How to Run

Run the scripts in this order:

```bash
python 01_data_prep.py
python 02_feature_engineering.py
python 03_modeling.py
python 04_decision_engine.py
python 05_evaluation_and_charts.py
```

You can also use `final_notebook.ipynb` as a presentation-friendly walkthrough of the full pipeline.
