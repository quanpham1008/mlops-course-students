from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "Price",
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe. Available: {list(df.columns)}")

    y = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # Keep numeric features only (drops columns like Address)
    numeric_cols = X_raw.select_dtypes(include="number").columns
    X = X_raw[numeric_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler