# Demand Forecasting and Inventory Decision Support

[![python-check](https://github.com/Ikteder/Demand-Forecasting-and-Inventory-Decision-Support/actions/workflows/python-check.yml/badge.svg)](https://github.com/Ikteder/Demand-Forecasting-and-Inventory-Decision-Support/actions/workflows/python-check.yml)

<p align="center">
  <img src="reports/figures/summary_cards.svg" width="100%" alt="Demand forecasting summary cards" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Forecasting-Time%20Series-0F766E?style=for-the-badge" alt="Time series forecasting" />
  <img src="https://img.shields.io/badge/Inventory-Decision%20Support-7C3AED?style=for-the-badge" alt="Inventory decision support" />
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-F43F5E?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit dashboard" />
</p>

A retail forecasting project that estimates store-product demand, compares forecasting approaches, and translates forecast error into inventory planning signals.

## Project Snapshot

| Area | Result |
| --- | --- |
| Best model on the bundled demo run | `LinearRegression` |
| Lowest RMSE | `3.10` |
| Lowest MAE | `2.47` |
| Highest store inventory exposure | `S2` |
| Highest product inventory exposure | `P2` |
| Chronological evaluation window | `35 days` |

## Visual Walkthrough

<p align="center">
  <img src="reports/figures/daily_demand_trend.svg" width="48%" alt="Daily demand trend" />
  <img src="reports/figures/model_comparison.svg" width="48%" alt="Model comparison" />
</p>
<p align="center">
  <img src="reports/figures/inventory_risk_by_store.svg" width="48%" alt="Inventory risk by store" />
  <img src="reports/figures/forecast_horizon_reliability.svg" width="48%" alt="Forecast horizon reliability" />
</p>
<p align="center">
  <img src="reports/figures/forecast_vs_actual_top_risk_series.svg" width="48%" alt="Forecast versus actual for the highest-risk series" />
  <img src="reports/figures/promo_effect.svg" width="48%" alt="Promotion error profile" />
</p>
<p align="center">
  <img src="reports/figures/hardest_series_mape.svg" width="48%" alt="Hardest series by MAPE" />
</p>

## Overview

The workflow is built around three practical questions:

- Which model is most reliable for short-horizon store-product forecasting?
- Which stores and products are hardest to forecast when variability increases?
- Where is the largest overstock and understock exposure if planners act on the forecast?

The repository includes a deterministic synthetic retail dataset so the full pipeline can run immediately and later be swapped for Rossmann, Favorita, Walmart, or M5. The bundled data contains 2,190 daily observations across two stores and three products from January 1 through December 30, 2024.

## Workflow

```mermaid
flowchart LR
    A["Retail demand history"] --> B["Store-product feature engineering"]
    B --> C["Lag features"]
    B --> D["Rolling demand statistics"]
    B --> E["Promo, holiday, and calendar signals"]
    C --> F["Naive baseline"]
    D --> G["Linear regression"]
    E --> H["Rolling mean baseline"]
    H --> I["Compare RMSE, MAE, and MAPE"]
    F --> I
    G --> I
    I --> J["Inventory risk summary"]
    J --> K["Planner-facing dashboard"]
```

## Key Findings

- Linear regression produced the strongest overall performance on the bundled synthetic data, with RMSE `3.10` and MAE `2.47` on the final 35 dates.
- These results demonstrate pipeline behavior, not expected performance on real retail operations.
- Store **S2** and product **P2** carried the largest simulated inventory exposure in the sample run.
- Forecast-horizon values are illustrative planning scenarios rather than separately trained horizon models.

## Repository Guide

| Path | Purpose |
| --- | --- |
| `src/` | Reusable code for loading data, engineering features, training models, and writing outputs |
| `app/streamlit_app.py` | Lightweight dashboard for metrics and figures |
| `data/demo/` | Bundled demo dataset |
| `reports/` | Metrics tables and chart assets used in the README |
| `sql/` | Business-facing rollup queries |
| `tests/` | Small checks for feature engineering helpers |
| `scripts/generate_demo_data.py` | Reproduce the bundled synthetic dataset with seed `20260409` |
| `docs/` | Dataset, experiment, model, decision, and working notes |

## Quick Start

```bash
pip install -r requirements.txt
python scripts/generate_demo_data.py  # optional: reproduce the bundled data
python -m pytest -q
python -m src.forecasting
streamlit run app/streamlit_app.py
```

## Verification

The automated workflow runs the test suite and the complete forecasting pipeline on Python 3.11. The current verified local run produced three passing tests and regenerated the checked-in reports from the deterministic dataset.

## Limitations

- The bundled data is synthetic and intentionally simple; it cannot validate real-world demand accuracy or inventory savings.
- The linear model and rolling baselines are demonstrations, not production forecasting recommendations.
- Inventory exposure is a simulated error-cost proxy and excludes lead times, service levels, carrying cost, lost sales, and operational constraints.
- The horizon reliability table is an illustrative scenario table rather than metrics from separately trained horizon-specific models.
- A real deployment needs leakage-controlled backtesting over multiple origins and evaluation across stores, products, seasonal regimes, and stockout behavior.

## License

Released under the [MIT License](LICENSE).

## Extensions

- Swap the demo dataset for Rossmann, Favorita, Walmart, or M5.
- Add XGBoost or LightGBM for a stronger boosted baseline.
- Split forecast quality by promotion regime, holiday periods, and stockout events.
- Add quantile forecasts for uncertainty-aware inventory planning.
- Add reorder-point and service-level policy recommendations.
