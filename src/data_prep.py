import pandas as pd

from .config import DATA_PATH


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
