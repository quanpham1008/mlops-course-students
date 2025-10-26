import mlflow
import os

def setup_mlflow(experiment_name: str = "default_experiment"):
    """
    Setup MLflow with SQLite backend and custom experiment.
    """
    os.makedirs("mlruns", exist_ok=True)

    tracking_db_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_db_uri)

    mlflow.set_experiment(experiment_name)
    print(f"✅ MLflow setup complete | DB: {tracking_db_uri} | Experiment: {experiment_name}")