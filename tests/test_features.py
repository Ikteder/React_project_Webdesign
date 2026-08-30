import pandas as pd
import pytest

from src.features import build_feature_table
from src.main import temporal_split_masks


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


def test_temporal_split_is_chronological_and_non_empty():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=50)})
    train_mask, test_mask = temporal_split_masks(df)

    assert train_mask.any()
    assert test_mask.any()
    assert df.loc[train_mask, "date"].max() < df.loc[test_mask, "date"].min()


def test_temporal_split_rejects_one_date():
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(ValueError, match="At least two"):
        temporal_split_masks(df)
