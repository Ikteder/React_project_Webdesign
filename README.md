# Demand Forecasting and Inventory Decision Support

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
| Best model on the bundled demo run | `RollingMean_7D` |
| Lowest RMSE | `7.84` |
| Lowest MAE | `5.92` |
| Highest store inventory exposure | `S2` |
| Highest product inventory exposure | `P3` |
| Most reliable forecast window | `7 days` |

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

The repository includes a bundled demo retail dataset so the full pipeline can run immediately and later be swapped for Rossmann, Favorita, Walmart, or M5.

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

- The rolling-mean baseline produced the strongest overall performance on the bundled demo data.
- Error increased during promotion-heavy periods, which is useful for setting more conservative safety stock.
- Store **S2** and product **P3** carried the largest simulated inventory exposure in the sample run.
- Reliability was strongest in the first week and decayed gradually as the forecast window widened.

## Repository Guide

| Path | Purpose |
| --- | --- |
| `src/` | Reusable code for loading data, engineering features, training models, and writing outputs |
| `app/streamlit_app.py` | Lightweight dashboard for metrics and figures |
| `data/demo/` | Bundled demo dataset |
| `reports/` | Metrics tables and chart assets used in the README |
| `sql/` | Business-facing rollup queries |
| `tests/` | Small checks for feature engineering helpers |
| `notebooks/` | Notebook starter for interactive exploration |

## Quick Start

```bash
pip install -r requirements.txt
python -m src.forecasting
streamlit run app/streamlit_app.py
```

## Extensions

- Swap the demo dataset for Rossmann, Favorita, Walmart, or M5.
- Add XGBoost or LightGBM for a stronger boosted baseline.
- Split forecast quality by promotion regime, holiday periods, and stockout events.
- Add quantile forecasts for uncertainty-aware inventory planning.
- Add reorder-point and service-level policy recommendations.
