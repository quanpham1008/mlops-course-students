import pandas as pd
import os

def load_data(data_path: str = "data/housing.csv") -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)