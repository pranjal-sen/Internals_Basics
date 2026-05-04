"""
Task 3 — Model Promotion (8 marks)
Assign "live" alias, train a challenger with random_state=99, compare & promote.
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
from mlflow.tracking import MlflowClient
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
TRACKING_URI = "file:///" + MLRUNS_DIR.replace("\\", "/")

# ── MLflow setup ───────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("testgenai-coverage-pct")
client = MlflowClient(tracking_uri=TRACKING_URI)

# ── Load previous results ─────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "step1_s1.json"), "r") as f:
    step1 = json.load(f)
with open(os.path.join(RESULTS_DIR, "step2_s6.json"), "r") as f:
    step2 = json.load(f)

registered_model_name = step2["registered_model_name"]
champion_version = step2["version"]
best_model_name = step1["best_model"]
champion_mae = step2["source_metric_value"]

# ── Step 1: Assign "live" alias to champion (version 1) ───────────────────────
client.set_registered_model_alias(registered_model_name, "live", str(champion_version))
print(f"Assigned 'live' alias to {registered_model_name} version {champion_version}")

# ── Step 2: Load data & prepare ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X = df.drop("coverage_pct", axis=1)
y = df["coverage_pct"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Step 3: Train challenger with random_state=99 ─────────────────────────────
if best_model_name == "GradientBoosting":
    challenger_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=99
    )
else:
    # SVR has no random_state; vary C to create a meaningful variant
    challenger_model = SVR(kernel="rbf", C=1.5, epsilon=0.1)

with mlflow.start_run(run_name=f"{best_model_name}_challenger_rs99") as run:
    mlflow.set_tag("domain", "ai_test_generation")
    mlflow.set_tag("variant", "challenger_random_state_99")

    mlflow.log_params(challenger_model.get_params())

    challenger_model.fit(X_train, y_train)
    y_pred = challenger_model.predict(X_test)

    challenger_mae = float(mean_absolute_error(y_test, y_pred))
    challenger_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    mlflow.log_metric("mae", challenger_mae)
    mlflow.log_metric("rmse", challenger_rmse)
    mlflow.sklearn.log_model(challenger_model, "model")

    challenger_run_id = run.info.run_id

print(f"Challenger MAE: {challenger_mae:.4f}  |  Champion MAE: {champion_mae:.4f}")

# ── Step 4: Register challenger as next version ───────────────────────────────
model_uri = f"runs:/{challenger_run_id}/model"
result = mlflow.register_model(model_uri, registered_model_name)
challenger_version = int(result.version)
print(f"Registered challenger as version {challenger_version}")

# Save challenger model locally
joblib.dump(challenger_model, os.path.join(MODELS_DIR, f"{best_model_name}_challenger.joblib"))

# ── Step 5: Compare and decide ────────────────────────────────────────────────
if challenger_mae < champion_mae:
    action = "promoted"
    client.set_registered_model_alias(registered_model_name, "live", str(challenger_version))
    print(f"[PROMOTED] Challenger (v{challenger_version}) promoted to 'live'")
else:
    action = "kept"
    print(f"[KEPT] Champion (v{champion_version}) retained as 'live'")

# ── Save results JSON ─────────────────────────────────────────────────────────
output = {
    "registered_model_name": registered_model_name,
    "alias_name": "live",
    "champion_version": champion_version,
    "challenger_version": challenger_version,
    "action": action,
}

results_path = os.path.join(RESULTS_DIR, "step3_s7.json")
with open(results_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[OK] Task 3 completed. Results saved to {results_path}")
print(json.dumps(output, indent=2))
