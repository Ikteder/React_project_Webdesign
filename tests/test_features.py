import pandas as pd

from src.features import build_feature_table


def test_feature_table_has_expected_columns():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "store_id": ["S1"] * 30,
            "product_id": ["P1"] * 30,
            "sales": list(range(30)),
            "promo": [0] * 30,
            "holiday": [0] * 30,
            "weekend": [0] * 30,
            "stockout": [0] * 30,
            "unit_price": [10.0] * 30,
        }
    )
    out = build_feature_table(df)
    assert "lag_7" in out.columns
    assert "roll_mean_7" in out.columns
