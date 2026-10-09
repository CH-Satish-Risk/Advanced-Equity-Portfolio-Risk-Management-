# 🛡️ Advanced Equity Portfolio Risk Management Platform
### VaR, Expected Shortfall, GARCH Modeling, Backtesting & Stress Testing

An institutional-grade quantitative risk analytics platform built in Python, designed to calculate, scale, backtest, and visualize Value at Risk (VaR) and Expected Shortfall (ES) for a multi-asset equity portfolio. The entire pipeline is wrapped into an interactive, multi-tab web application using **Streamlit** and **Plotly**.



## 📊 Project Completion Summary & Analytics Workflow

With the completion of **Phase 10**, this project successfully implements an end-to-end institutional risk management pipeline across 10 distinct phases:

1. **Data Collection & Cleaning (Phases 1–2):** Automated ingestion of 3 years of daily adjusted closing prices and benchmark index data via Yahoo Finance (`yfinance`), ensuring protection against corporate action distortions, missing date alignment, and log-return transformations.
2. **Portfolio Construction & P&L (Phase 3):** Allocation of capital across an equity universe (`AAPL`, `MSFT`, `NVDA`, `JPM`, `XOM`, `AMZN`, `META`, `GOOGL`, `TSLA`, `UNH`) to generate daily dollar P&L and cumulative portfolio value trajectories.
3. **VaR Modeling & Volatility Scaling (Phases 4–7):** 
   - **Historical Simulation VaR & Expected Shortfall:** Computed across multiple rolling windows (250D, 500D, 750D) at 95%, 99%, and 99.5% confidence levels.
   - **EWMA Volatility Scaling ($\lambda = 0.94$):** Dynamic risk adjustment reacting to recent volatility clustering.
   - **GARCH(1,1) & GJR-GARCH Scaling:** Maximum likelihood estimation capturing conditional variance and the **asymmetric leverage effect** (where negative market drops amplify future volatility more aggressively than positive returns).
4. **Rigorous Backtesting & Basel Traffic Light Validation (Phase 8):** Out-of-sample exception analysis, Kupiec POF Likelihood Ratio Tests, and Basel Traffic Light zone classifications (Green, Yellow, Red) ensuring model regulatory compliance.
5. **Model Comparison & Stress Testing (Phases 9–10):** Side-by-side model rankings, historical black-swan crisis simulations (2008 Financial Crisis, COVID-19 Crash 2020, 2022 Inflation Shock), and hypothetical macro market drop scenarios.


## 🚀 Live Dashboard Features (`app.py`)

The Streamlit web application is organized into five core functional pages:
* **Portfolio Page:** Asset allocation tables, initial capital configurations, and cumulative return tracking.
* **VaR & Expected Shortfall Page:** Interactive historical window selectors and tail-risk threshold charts.
* **Volatility Scaling Page:** Conditional volatility comparisons across EWMA, GARCH, and GJR-GARCH models.
* **Backtesting Page:** Exception timelines, statistical test outputs, and Basel Traffic Light risk zones.
* **Stress Testing Page:** Historical shock evaluations and hypothetical portfolio drawdown scenario rankings.

---

## 📁 Repository Structure

```text
advanced-portfolio-risk-platform/
│
├── app.py              # Complete Streamlit multi-tab dashboard & risk engine
├── requirements.txt    # Python package dependencies
└── README.md           # Project documentation
