# 📊 Early Warning System for Portfolio Risk in Digital Wealth Platforms

## Overview

This project develops a **data-driven early warning system** for portfolio risk management in digital wealth platforms.

The system integrates:
- **Machine learning and econometric models** to forecast short-term volatility
- A **multi-layer decision engine** to translate forecasts into portfolio allocations
- A **backtesting and evaluation framework** to assess real-world performance

The goal is not to replace traditional portfolio construction (e.g., 60/40), but to provide a **tactical overlay** that improves risk responsiveness, drawdown control, and implementation quality.

---

## 🧠 Key Idea

Most robo-advisors rely on static or slow-moving allocation rules.

This project introduces a system that:
1. **Predicts near-term volatility (5-day horizon)**
2. **Adjusts portfolio weights dynamically**
3. **Balances risk, uncertainty, momentum, and market conditions**

---

## ⚙️ System Architecture

```
Data → Feature Engineering → Modeling → Decision Engine → Evaluation & Charts
```

### 1. Data Layer
- Assets:
  - SPY (Equities)
  - TLT (Bonds)
  - VIX (Market volatility proxy)
- Source: Yahoo Finance via `yfinance` 

---

### 2. Feature Engineering
From the master dataset, the system builds predictive features such as:
- Lagged returns (short-term signals)
- Rolling volatility (5d, 10d, 20d)
- Drawdowns (risk state)
- Moving average gaps (trend)
- VIX-based stress indicators
- SPY–TLT correlation (regime detection)

Target:
- **Next 5-day realised volatility (annualised)** 

---

### 3. Modeling Layer

Models implemented:

| Model Type | Purpose |
|-----------|--------|
| Baseline (rolling vol) | Benchmark |
| ETS | Time-series smoothing |
| ARIMA | Statistical forecasting |
| GARCH(1,1) | Volatility clustering |
| Random Forest (Optuna tuned) | Non-linear feature learning |
| Stacked RF + GARCH | Combines ML + parametric |
| Regime Switching | Adaptive model selection |
| **Calibrated RF (Final Model)** | Best performing |

Key improvement:
- **Out-of-fold stacking** avoids data leakage
- **GARCH complements RF’s extrapolation limits** 

---

### 4. Decision Engine (Core Contribution)

This is the **most important part of the project**.

The decision engine converts forecasts into portfolio weights across:
- SPY (Equities)
- TLT (Bonds)
- Cash

#### Core Logic Layers

1. **Volatility Targeting**
   - Reduce exposure when predicted risk rises

2. **Uncertainty Adjustment**
   - Penalise allocations when model confidence is low

3. **Relative Attractiveness**
   - Compare SPY vs TLT risk-adjusted appeal

4. **Momentum Overlay**
   - Incorporate short-term trend signals

5. **Drawdown Control**
   - Reduce risk during portfolio stress

6. **Stress Regime Handling**
   - Hard caps during extreme volatility

7. **Re-entry Mechanism**
   - Gradually increase exposure when conditions improve

This creates a **realistic, institutional-style allocation system**, rather than a simple rule-based model.

---

### 5. Evaluation Framework

The system evaluates both:
#### Model Performance
- RMSE
- Correlation
- Warning precision / recall (high-volatility detection)

#### Portfolio Performance
- Annualised return
- Volatility
- Sharpe ratio
- Maximum drawdown
- Turnover (implementation cost)
- Target volatility gap

---

### 6. Benchmarks

The dynamic strategy is compared against:
- Static 60/40 portfolio
- Naive volatility targeting
- 100% SPY

---

### 7. Visualisation & Reporting

Automated outputs include:
- Forecast vs actual volatility charts
- Model comparison bar charts
- Portfolio value curves
- Drawdown analysis
- Allocation weights over time
- KPI summary tables

All charts are exported for:
- charts

---

## 📁 Project Structure

```
project/
│
├── 00_download_data.py          # Download raw market data
├── 01_data_prep.py             # Clean and merge datasets
├── 02_feature_engineering.py   # Build features + targets
├── 03_modeling.py              # Train and evaluate models
├── 04_decision_engine.py       # Portfolio allocation logic
├── 05_evaluation_and_charts.py # Generate charts and tables
│
├── config.py                   # Global configuration and paths
├── requirements.txt            # Dependencies
│
├── data/                       # Raw data files
├── outputs/                    # All generated outputs
│   ├── master_dataset.csv
│   ├── predictions.csv
│   ├── portfolio_results.csv
│   ├── portfolio_kpis.csv
│   └── charts/
```

---

## 🚀 How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```
Dependencies include:
- pandas, numpy
- scikit-learn
- statsmodels
- optuna
- arch
- yfinance

---

### 2. Run pipeline (in order)

```
python 00_download_data.py
python 01_data_prep.py
python 02_feature_engineering.py
python 03_modeling.py
python 04_decision_engine.py
python 05_evaluation_and_charts.py
```

---

## 📊 Pre-generated Outputs

To make evaluation easier, we have included:
- Model predictions and metrics
- Portfolio backtest results
- All charts used in the report

These can be found in the `outputs/` folder.

To reproduce results from scratch, run the full pipeline as described above.

---

## 📈 Key Insights

- Volatility forecasting alone is **not sufficient**
- Performance improves significantly when forecasts are embedded into:
  - **structured decision logic**
  - **risk-aware allocation rules**
- The dynamic strategy:
  - Reduces **maximum drawdown**
  - Improves **risk control**
  - Maintains competitive returns

---

## 💼 Business Relevance

This system is designed for:
- Robo-advisors
- Digital wealth platforms
- Portfolio risk monitoring tools

It can be implemented as:
- A backend allocation engine
- A risk overlay module
- A client-facing advisory enhancement

---

## ⚠️ Limitations

- Limited to 3 assets (SPY, TLT, Cash)
- Model performance depends on historical regimes
- Transaction costs simplified (bps assumption)
- No macroeconomic or alternative data integration

---

## 🔮 Future Improvements

- Multi-asset expansion (commodities, FX, alternatives)
- Real-time data integration
- Reinforcement learning allocation policies
- Client-level customization
- Live deployment API

---

## 👨‍💻 Authors

BC2407 Sem 6 Team 5

---

## 📌 Final Note

This project demonstrates that:

> The real value is not in predicting risk —  
> but in **how those predictions are translated into decisions**.