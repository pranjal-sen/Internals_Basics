"""
Task 1 — Experiment Tracking & Model Comparison (6 marks)
Train SVR and GradientBoosting models, log to MLflow, select best by MAE.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
TRACKING_URI = "file:///" + MLRUNS_DIR.replace("\\", "/")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── MLflow setup ───────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(TRACKING_URI)
experiment_name = "testgenai-coverage-pct"
mlflow.set_experiment(experiment_name)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X = df.drop("coverage_pct", axis=1)
y = df["coverage_pct"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Define models ──────────────────────────────────────────────────────────────
models_config = {
    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
}

# ── Train and log ─────────────────────────────────────────────────────────────
results = []
best_model_name = None
best_mae = float("inf")
best_run_id = None

for name, model in models_config.items():
    with mlflow.start_run(run_name=name) as run:
        # Tag
        mlflow.set_tag("domain", "ai_test_generation")

        # Log all hyperparameters
        params = model.get_params()
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Also save locally
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))

        results.append({"name": name, "mae": round(mae, 4), "rmse": round(rmse, 4)})

        if mae < best_mae:
            best_mae = mae
            best_model_name = name
            best_run_id = run.info.run_id

        print(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f} (run_id={run.info.run_id})")

# ── Save results JSON ─────────────────────────────────────────────────────────
output = {
    "experiment_name": experiment_name,
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "mae",
    "best_metric_value": round(best_mae, 4),
}

results_path = os.path.join(RESULTS_DIR, "step1_s1.json")
with open(results_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[OK] Task 1 completed. Results saved to {results_path}")
print(json.dumps(output, indent=2))
