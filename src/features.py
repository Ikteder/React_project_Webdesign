import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dayofweek"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["days_from_start"] = (out["date"] - out["date"].min()).dt.days
    return out


def add_lag_features(df: pd.DataFrame, lags=(1, 7, 14)) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(["store_id", "product_id"])["sales"]
    for lag in lags:
        out[f"lag_{lag}"] = grouped.shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, windows=(7, 14)) -> pd.DataFrame:
    out = df.copy()
    for window in windows:
        out[f"roll_mean_{window}"] = out.groupby(["store_id", "product_id"])["sales"].transform(
            lambda s: s.shift(1).rolling(window).mean()
        )
    return out


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(df)
    out = add_lag_features(out)
    out = add_rolling_features(out)
    return out.dropna().reset_index(drop=True)
