# GJR-GARCH Volatility Regime Strategy

This project implements a volatility-regime trading strategy using a rolling GJR-GARCH model. It downloads TSLA and S&P 500 data from Yahoo Finance, forecasts conditional volatility, classifies low-volatility and high-volatility regimes, and backtests a simple exposure-control strategy that invests during low-volatility regimes and moves to cash during high-volatility regimes.

## Resume Summary

Built a Python-based volatility-regime strategy that estimates rolling GJR-GARCH forecasts, classifies market states by expanding volatility percentiles, backtests regime-based TSLA exposure with commission assumptions, and compares performance against the S&P 500 benchmark.

## Project Highlights

- Downloads equity and benchmark data with `yfinance`.
- Fits rolling GJR-GARCH(1,1,1) models with Student-t errors.
- Uses conditional volatility percentiles to define market regimes.
- Generates lagged trading signals to reduce look-ahead bias.
- Includes commission assumptions in the backtest.
- Reports CAGR, Sharpe, Sortino, Calmar, drawdown, win rate, and regime diagnostics.

## Methodology

1. Download TSLA and S&P 500 adjusted close prices.
2. Compute daily returns for the traded asset.
3. Estimate rolling GJR-GARCH conditional volatility.
4. Classify each date as low-volatility or high-volatility by percentile rank.
5. Invest in TSLA during low-volatility regimes and move to cash during high-volatility regimes.
6. Backtest daily portfolio value after commissions.
7. Compare strategy performance with the S&P 500 benchmark.
8. Save performance tables and diagnostic charts.

## Repository Structure

```text
gjr-garch-regime-strategy/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── gjr_garch_regime_strategy.py
└── outputs/
```

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/gjr_garch_regime_strategy.py
```

## Saved Outputs

Each run writes the following artifacts to `outputs/`:

- `strategy_results.csv`
- `performance_report.csv`
- `detailed_stats.csv`
- `output_manifest.csv`
- `gjr_garch_regime_results.png`

## Notes

This project is a research prototype, not investment advice or a production trading system. A production version should add stronger data validation, walk-forward parameter selection, slippage modeling, position limits, and robustness checks across multiple tickers and market regimes.

