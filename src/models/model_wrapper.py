from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    _has_xgboost = True
except Exception:
    _has_xgboost = False

def get_model(model_name: str = "linear", params: dict = None):
    params = params or {}

    if model_name == "linear":
        return LinearRegression(**params)
    elif model_name == "random_forest":
        return RandomForestRegressor(**params)
    elif model_name == "decision_tree":
        return DecisionTreeRegressor(**params)
    elif model_name == "knn":
        return KNeighborsRegressor(**params)
    elif model_name == "xgboost":
        if not _has_xgboost:
            raise ImportError("xgboost is not installed. Please add 'xgboost' to requirements and install.")
        return XGBRegressor(**params)
    else:
        raise ValueError(f"Model not supported: {model_name}")
