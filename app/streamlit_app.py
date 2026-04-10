from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
FIGS = REPORTS / "figures"

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")
st.title("Demand Forecasting and Inventory Decision Support")

st.subheader("Model Comparison")
st.dataframe(pd.read_csv(REPORTS / "metrics_summary.csv"), use_container_width=True)

st.subheader("Saved Charts")
for fig_name in [
    "summary_cards.svg",
    "daily_demand_trend.svg",
    "promo_effect.svg",
    "model_comparison.svg",
    "forecast_vs_actual_top_risk_series.svg",
    "hardest_series_mape.svg",
    "inventory_risk_by_store.svg",
    "forecast_horizon_reliability.svg",
]:
    path = FIGS / fig_name
    if path.exists():
        st.image(str(path), use_container_width=True)
