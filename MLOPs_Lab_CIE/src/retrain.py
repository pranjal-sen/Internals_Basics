"""
Task 4 — Retraining Pipeline (8 marks)
Combine old + new data, retrain, compare RMSE, promote if improvement >= 0.5.
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
NEW_DATA_PATH = os.path.join(BASE_DIR, "data", "new_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
TRACKING_URI = "file:///" + MLRUNS_DIR.replace("\\", "/")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── MLflow setup ───────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("testgenai-coverage-pct")
client = MlflowClient(tracking_uri=TRACKING_URI)

# ── Load previous results ─────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "step1_s1.json"), "r") as f:
    step1 = json.load(f)
with open(os.path.join(RESULTS_DIR, "step3_s7.json"), "r") as f:
    step3 = json.load(f)

best_model_name = step1["best_model"]
registered_model_name = step3["registered_model_name"]

# ── Combine datasets ──────────────────────────────────────────────────────────
df_original = pd.read_csv(DATA_PATH)
df_new = pd.read_csv(NEW_DATA_PATH)
df_combined = pd.concat([df_original, df_new], ignore_index=True)

original_rows = len(df_original)
new_rows = len(df_new)
combined_rows = len(df_combined)

print(f"Original: {original_rows} rows | New: {new_rows} rows | Combined: {combined_rows} rows")

X = df_combined.drop("coverage_pct", axis=1)
y = df_combined["coverage_pct"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Evaluate champion on the combined test set ────────────────────────────────
champion_model_uri = f"models:/{registered_model_name}@live"
champion_model = mlflow.sklearn.load_model(champion_model_uri)

y_pred_champion = champion_model.predict(X_test)
champion_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_champion)))
print(f"Champion RMSE on combined test set: {champion_rmse:.4f}")

# ── Retrain on combined data ──────────────────────────────────────────────────
if best_model_name == "GradientBoosting":
    retrained_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
else:
    retrained_model = SVR(kernel="rbf", C=1.0, epsilon=0.1)

with mlflow.start_run(run_name=f"{best_model_name}_retrained_combined") as run:
    mlflow.set_tag("domain", "ai_test_generation")
    mlflow.set_tag("retrain", "combined_data")

    mlflow.log_params(retrained_model.get_params())

    retrained_model.fit(X_train, y_train)
    y_pred_retrained = retrained_model.predict(X_test)

    retrained_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_retrained)))
    retrained_mae = float(mean_absolute_error(y_test, y_pred_retrained))

    mlflow.log_metric("rmse", retrained_rmse)
    mlflow.log_metric("mae", retrained_mae)
    mlflow.sklearn.log_model(retrained_model, "model")

    retrained_run_id = run.info.run_id

print(f"Retrained RMSE: {retrained_rmse:.4f}")

# ── Compare and decide ────────────────────────────────────────────────────────
improvement = champion_rmse - retrained_rmse
print(f"Improvement: {improvement:.4f} (threshold: 0.5)")

if improvement >= 0.5:
    action = "promoted"
    # Register retrained model as new version
    model_uri = f"runs:/{retrained_run_id}/model"
    result = mlflow.register_model(model_uri, registered_model_name)
    new_version = int(result.version)
    client.set_registered_model_alias(registered_model_name, "live", str(new_version))
    print(f"[PROMOTED] Retrained model promoted to 'live' (version {new_version})")
else:
    action = "kept_champion"
    print("[KEPT] Champion retained - improvement below threshold")

# Save retrained model locally
joblib.dump(retrained_model, os.path.join(MODELS_DIR, f"{best_model_name}_retrained.joblib"))

# ── Save results JSON ─────────────────────────────────────────────────────────
output = {
    "original_data_rows": original_rows,
    "new_data_rows": new_rows,
    "combined_data_rows": combined_rows,
    "champion_rmse": round(champion_rmse, 4),
    "retrained_rmse": round(retrained_rmse, 4),
    "improvement": round(improvement, 4),
    "min_improvement_threshold": 0.5,
    "action": action,
    "comparison_metric": "rmse",
}

results_path = os.path.join(RESULTS_DIR, "step4_s8.json")
with open(results_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[OK] Task 4 completed. Results saved to {results_path}")
print(json.dumps(output, indent=2))
