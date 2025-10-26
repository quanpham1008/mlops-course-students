import mlflow
import mlflow.sklearn
from src.models.model_wrapper import get_model
from src.utils.logger import get_logger
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

logger = get_logger("training")

def train_model(X_train, X_test, y_train, y_test, model_name="random_forest", params=None):
    model = get_model(model_name, params)

    with mlflow.start_run():
        logger.info(f"Training {model_name} với params: {params}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params or {})
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return model, {"rmse": rmse, "r2": r2}
