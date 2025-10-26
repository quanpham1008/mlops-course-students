import argparse
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.tracking.mlflow_utils import setup_mlflow
from src.data.data_loader import load_data
from src.data.preprocess import preprocess_data
from src.models.train import train_model
from src.tuning.tune_ray import tune_hyperparameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML model for housing price prediction")

    parser.add_argument("--config", type=str, default="./src/config/params.yaml", help="Path to YAML config")
    parser.add_argument("--model", type=str, help="Model name override (e.g., random_forest, linear)")
    parser.add_argument("--tune", action="store_true", help="Enable Ray Tune hyperparameter search")
    parser.add_argument("--experiment", type=str, help="Override MLflow experiment name")
    parser.add_argument("--data-path", type=str, help="Path to input dataset CSV")

    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("pipeline")

    # --- Load YAML config ---
    config = load_config(args.config)

    # --- Override with CLI arguments ---
    model_name = args.model or config.get("model_name")
    data_path = args.data_path or config.get("data_path")
    experiment_name = args.experiment or config.get("experiment_name")
    random_seed = config.get("random_seed", 42)
    target_col = config.get("target_col", "Price")
    model_params = config.get("model_params", {}).get(model_name, {})

    # --- Setup MLflow ---
    setup_mlflow(experiment_name)

    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Seed: {random_seed}")

    # --- Load and preprocess data ---
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, _ = preprocess_data(
        df, target_col=target_col, random_state=random_seed
    )

    # --- Tuning or direct training ---
    if args.tune:
        logger.info("Starting Ray Tune search...")
        best_params = tune_hyperparameters(
            model_name=model_name,
            data_path=data_path,
            experiment_name=f"{experiment_name}_tune",
            num_samples=config.get("tune_num_samples", 10),
        )
    else:
        best_params = model_params

    # --- Train model ---
    model, metrics = train_model(X_train, X_test, y_train, y_test, model_name, best_params)

    logger.info(f"✅ Training done | RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
