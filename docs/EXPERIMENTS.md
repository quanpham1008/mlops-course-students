# Experiments Guide (Training, Ray Tune, MLflow)

This guide explains how to set up the environment, configure experiments, run training with or without Ray Tune, use GPU with XGBoost, and view metrics in MLflow.

## 1) Environment uv setup

```bash
# From project root
uv venv
source .venv/bin/activate
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -r requirements.txt
```

- Activate later: `source .venv/bin/activate`
- Deactivate: `deactivate`

## 2) Data

- Default dataset path: `data/housing.csv`
- Target column: `Price`
- Preprocessing keeps numeric features only, so non-numeric columns like `Address` are safely ignored.

If your dataset changes, update `target_col` and `data_path` in the config (see below).

## 3) Configuration

File: `src/config/params.yaml`

- `experiment_name`: MLflow experiment name
- `data_path`: CSV path, e.g. `data/housing.csv`
- `model_name`: one of `linear`, `decision_tree`, `knn`, `xgboost`, `random_forest`
- `random_seed`: integer
- `target_col`: default `Price`
- `model_params`: default parameters per model
- Optional: `tune_num_samples`: Ray Tune trials count (default 10 if missing)

Example:
```yaml
experiment_name: housing_price_prediction
data_path: data/housing.csv
model_name: linear
random_seed: 42
target_col: Price

model_params:
  linear:
    fit_intercept: true
  random_forest:
    n_estimators: 150
    max_depth: 6
    min_samples_split: 0.2
  decision_tree:
    max_depth: 6
    min_samples_split: 2
  knn:
    n_neighbors: 5
    weights: distance
  xgboost:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    # For GPU with XGBoost, add:
    # tree_method: gpu_hist
    # device: cuda
```

## 4) Run training

Basic run (uses config’s model and params):
```bash
source .venv/bin/activate
python3 train_pipeline.py --config ./src/config/params.yaml
```

Override model via CLI:
```bash
python3 train_pipeline.py --config ./src/config/params.yaml --model linear
python3 train_pipeline.py --config ./src/config/params.yaml --model random_forest
python3 train_pipeline.py --config ./src/config/params.yaml --model xgboost
```

## 5) Ray Tune hyperparameter search

Enable tuning (example for linear):
```bash
python3 train_pipeline.py \
  --config ./src/config/params.yaml \
  --model linear \
  --tune \
  --experiment linear_tune
```

Notes:
- The tuner uses per-model search spaces (e.g., `fit_intercept`, `positive` for linear).
- After tuning, the best config is printed, and the pipeline trains the final model with those params and logs to MLflow.

## 6) GPU runs (XGBoost)

- scikit-learn models (e.g., `linear`, `random_forest`) do not use GPU.
- To use GPU, choose `xgboost` and set GPU params in config:
```yaml
model_params:
  xgboost:
    tree_method: gpu_hist
    device: cuda
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
```
Run:
```bash
python3 train_pipeline.py --config ./src/config/params.yaml --model xgboost
```

## 7) View metrics in MLflow UI

Start the UI from project root:
```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:./mlruns --port 5000
```
Open `http://127.0.0.1:5000` and select experiments:
- Main training runs: `experiment_name` (e.g., `housing_price_prediction`, `linear_tune`)
- Ray Tune trials: `linear_tune_tune` (created by the tuner)

Metrics logged: `rmse`, `r2_score`

Models and artifacts: under each run’s Artifacts tab (e.g., `model`).

## 8) Logs and outputs

- Logs: `logs/pipeline.log`, `logs/training.log`
- MLflow DB: `mlflow.db`
- MLflow artifacts: `mlruns/`
- Ray Tune results (console): `~/ray_results/...`

## 9) Troubleshooting

- ModuleNotFoundError (e.g., `mlflow`): activate venv and run `python3 -m pip install -r requirements.txt`.
- Non-numeric data error: preprocessing selects numeric features; ensure `target_col` exists in the CSV.
- Ray Tune `report()` error: use Ray `session.report({"rmse": ..., "r2": ...})`.
- Best config error: use `analysis.get_best_config(metric="rmse", mode="min")`.
- Linear + GPU: not supported; use `xgboost` for GPU.

## 10) Examples

- Baseline linear:
```bash
python3 train_pipeline.py --config ./src/config/params.yaml --model linear
```
- Linear with tuning + MLflow experiment name:
```bash
python3 train_pipeline.py --config ./src/config/params.yaml --model linear --tune --experiment linear_tune
```
- XGBoost GPU (after setting GPU params in config):
```bash
python3 train_pipeline.py --config ./src/config/params.yaml --model xgboost --experiment xgb_gpu
```
