from ray import tune
from ray.tune.schedulers import ASHAScheduler
from src.models.train import train_model
from src.data.data_loader import load_data
from src.data.preprocess import preprocess_data
import mlflow
from ray.air import session


def _search_space(model_name: str):
    if model_name == "linear":
        return {
            "fit_intercept": tune.choice([True, False]),
            "positive": tune.choice([False, True]),
        }
    elif model_name == "random_forest":
        return {
            "n_estimators": tune.randint(50, 200),
            "max_depth": tune.randint(3, 10),
            "min_samples_split": tune.uniform(0.1, 1.0),
        }
    elif model_name == "decision_tree":
        return {
            "max_depth": tune.randint(2, 16),
            "min_samples_split": tune.randint(2, 20),
        }
    elif model_name == "knn":
        return {
            "n_neighbors": tune.randint(3, 50),
            "weights": tune.choice(["uniform", "distance"]),
        }
    elif model_name == "xgboost":
        return {
            "n_estimators": tune.randint(100, 500),
            "max_depth": tune.randint(3, 10),
            "learning_rate": tune.loguniform(1e-3, 3e-1),
            "subsample": tune.uniform(0.5, 1.0),
            "colsample_bytree": tune.uniform(0.5, 1.0),
        }
    else:
        raise ValueError(f"Unsupported model for tuning: {model_name}")


def tune_hyperparameters(
    model_name: str = "linear",
    data_path: str | None = None,
    experiment_name: str = "ray_tune_experiments",
    num_samples: int = 10,
    resources_per_trial: dict | None = None,
):
    df = load_data(data_path) if data_path else load_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)

    search_space = _search_space(model_name)

    def trainable(config):
        mlflow.set_experiment(experiment_name)
        _, metrics = train_model(X_train, X_test, y_train, y_test, model_name, config)
        session.report({"rmse": metrics["rmse"], "r2": metrics["r2"]})

    scheduler = ASHAScheduler(metric="rmse", mode="min")
    analysis = tune.run(
        trainable,
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        resources_per_trial=resources_per_trial or {"cpu": 1},
    )

    best_cfg = analysis.get_best_config(metric="rmse", mode="min")
    print("Best config:")
    print(best_cfg)
    return best_cfg
