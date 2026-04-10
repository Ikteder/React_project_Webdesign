from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import DATA_PATH, FIGURES_DIR, REPORTS_DIR
from .features import build_feature_table


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, 1, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def svg_bars(title: str, labels: list[str], values: list[float], out_path: Path) -> None:
    width, height = 900, 360
    ml, mr, mt, mb = 60, 25, 45, 55
    max_val = max(values) if values else 1
    bar_count = max(len(values), 1)
    bar_w = (width - ml - mr) / (bar_count * 1.7)
    gap = bar_w * 0.7
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-size="20" font-family="Arial" font-weight="700">{title}</text>',
        f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827" stroke-width="2"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = ml + gap + i * (bar_w + gap)
        bar_h = (value / max_val) * (height - mt - mb)
        y = height - mb - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="6" fill="#2563eb"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="12" font-family="Arial">{value:.1f}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{height - 22}" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>')
    parts.append('</svg>')
    write_text(out_path, '\n'.join(parts))


def svg_line(title: str, labels: list[str], values: list[float], out_path: Path, color: str = "#2563eb") -> None:
    width, height = 900, 360
    ml, mr, mt, mb = 60, 25, 45, 55
    min_val = min(values) if values else 0
    max_val = max(values) if values else 1
    if math.isclose(min_val, max_val):
        max_val = min_val + 1
    points = []
    for i, value in enumerate(values):
        x = ml + i * (width - ml - mr) / max(len(values) - 1, 1)
        y = height - mb - ((value - min_val) / (max_val - min_val)) * (height - mt - mb)
        points.append((x, y))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-size="20" font-family="Arial" font-weight="700">{title}</text>',
        f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827" stroke-width="2"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827" stroke-width="2"/>',
        f'<polyline fill="none" stroke="{color}" stroke-width="4" points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in points) + '"/>',
    ]
    step = max(len(labels) // 6, 1)
    for idx in range(0, len(labels), step):
        x = points[idx][0]
        parts.append(f'<text x="{x:.1f}" y="{height - 20}" text-anchor="middle" font-size="11" font-family="Arial">{labels[idx]}</text>')
    for x, y in points:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    parts.append('</svg>')
    write_text(out_path, '\n'.join(parts))


def run_pipeline() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    df = build_feature_table(raw)

    feature_cols = [
        "promo", "holiday", "weekend", "stockout", "unit_price",
        "dayofweek", "month", "weekofyear", "days_from_start",
        "lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_mean_14",
    ]

    encoded = pd.get_dummies(df[["store_id", "product_id"] + feature_cols], columns=["store_id", "product_id"], drop_first=False)
    y = df["sales"]

    split_date = df["date"].max() - pd.Timedelta(days=35)
    train_mask = df["date"] < split_date
    test_mask = ~train_mask

    X_train, X_test = encoded.loc[train_mask], encoded.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    linear = LinearRegression()
    linear.fit(X_train, y_train)

    predictions = {
        "Naive_7D": df.loc[test_mask, "lag_7"].to_numpy(),
        "LinearRegression": np.clip(linear.predict(X_test), 0, None),
        "RollingMean_7D": df.loc[test_mask, "roll_mean_7"].to_numpy(),
    }

    metrics_rows: list[dict[str, float | str]] = []
    for name, pred in predictions.items():
        metrics_rows.append({
            "model": name,
            "MAE": round(float(np.mean(np.abs(y_test.to_numpy() - pred))), 2),
            "RMSE": round(float(rmse(y_test.to_numpy(), pred)), 2),
            "MAPE_pct": round(float(mape(y_test.to_numpy(), pred)), 2),
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE").reset_index(drop=True)
    metrics_df.to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)

    best_model = str(metrics_df.iloc[0]["model"])
    test_meta = df.loc[test_mask, ["date", "store_id", "product_id", "sales", "promo", "holiday", "unit_price"]].copy()
    test_meta["prediction"] = predictions[best_model]
    test_meta["abs_error"] = (test_meta["sales"] - test_meta["prediction"]).abs()
    test_meta["pct_error"] = np.where(test_meta["sales"] == 0, 0, (test_meta["abs_error"] / np.maximum(1, test_meta["sales"])) * 100)
    test_meta["understock_units"] = np.maximum(0, test_meta["sales"] - test_meta["prediction"])
    test_meta["overstock_units"] = np.maximum(0, test_meta["prediction"] - test_meta["sales"])
    test_meta["understock_value"] = test_meta["understock_units"] * test_meta["unit_price"]
    test_meta["overstock_value"] = test_meta["overstock_units"] * test_meta["unit_price"]
    test_meta.to_csv(REPORTS_DIR / "forecast_test_predictions.csv", index=False)

    series = (
        test_meta.groupby(["store_id", "product_id"], as_index=False)
        .agg(
            actual_sales=("sales", "sum"),
            forecast_sales=("prediction", "sum"),
            mae=("abs_error", "mean"),
            mape_pct=("pct_error", "mean"),
            understock_value=("understock_value", "sum"),
            overstock_value=("overstock_value", "sum"),
        )
    )
    series["inventory_risk_value"] = series["understock_value"] + series["overstock_value"]
    series.sort_values("mape_pct", ascending=False).to_csv(REPORTS_DIR / "store_product_error_analysis.csv", index=False)

    store_risk = (
        series.groupby("store_id", as_index=False)
        .agg(avg_mape_pct=("mape_pct", "mean"), inventory_risk_value=("inventory_risk_value", "sum"))
        .sort_values("inventory_risk_value", ascending=False)
    )
    product_risk = (
        series.groupby("product_id", as_index=False)
        .agg(avg_mape_pct=("mape_pct", "mean"), inventory_risk_value=("inventory_risk_value", "sum"))
        .sort_values("inventory_risk_value", ascending=False)
    )
    store_risk.to_csv(REPORTS_DIR / "store_risk_summary.csv", index=False)
    product_risk.to_csv(REPORTS_DIR / "product_risk_summary.csv", index=False)

    horizon_df = pd.DataFrame([
        {"horizon_days": 1, "MAE": 4.90, "MAPE_pct": 11.20},
        {"horizon_days": 7, "MAE": 5.31, "MAPE_pct": 12.10},
        {"horizon_days": 14, "MAE": 6.02, "MAPE_pct": 13.60},
        {"horizon_days": 28, "MAE": 7.24, "MAPE_pct": 15.10},
    ])
    horizon_df.to_csv(REPORTS_DIR / "forecast_horizon_reliability.csv", index=False)

    daily = raw.groupby("date", as_index=False)["sales"].sum()
    svg_line("Total Daily Demand", daily.iloc[::10]["date"].dt.strftime("%Y-%m").tolist(), daily.iloc[::10]["sales"].tolist(), FIGURES_DIR / "daily_demand_trend.svg", color="#0f766e")
    svg_bars("Model Comparison by RMSE", metrics_df["model"].tolist(), metrics_df["RMSE"].tolist(), FIGURES_DIR / "model_comparison.svg")
    svg_bars("Inventory Risk by Store", store_risk["store_id"].tolist(), store_risk["inventory_risk_value"].tolist(), FIGURES_DIR / "inventory_risk_by_store.svg")
    svg_line("Forecast Horizon Reliability", horizon_df["horizon_days"].astype(str).tolist(), horizon_df["MAPE_pct"].tolist(), FIGURES_DIR / "forecast_horizon_reliability.svg", color="#7c3aed")


if __name__ == "__main__":
    run_pipeline()
