"""
Task 2 — Model Versioning (8 marks)
Register the best model from Task 1 in the MLflow Model Registry.
"""
import os
import json
import mlflow
from mlflow.tracking import MlflowClient

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
TRACKING_URI = "file:///" + MLRUNS_DIR.replace("\\", "/")

# ── MLflow setup ───────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

# ── Load Task 1 results ───────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "step1_s1.json"), "r") as f:
    step1 = json.load(f)

best_model_name = step1["best_model"]
best_metric_value = step1["best_metric_value"]

# ── Find the MLflow run for the best model ─────────────────────────────────────
experiment = mlflow.get_experiment_by_name("testgenai-coverage-pct")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{best_model_name}'",
    order_by=["start_time DESC"],
    max_results=1,
)

if runs.empty:
    raise RuntimeError(f"No MLflow run found for model '{best_model_name}'")

run_id = runs.iloc[0]["run_id"]
print(f"Found best model run: {run_id} ({best_model_name})")

# ── Register model ─────────────────────────────────────────────────────────────
registered_model_name = "testgenai-coverage-pct-predictor"
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(model_uri, registered_model_name)
version = int(result.version)
print(f"Registered model '{registered_model_name}' version {version}")

# ── Save results JSON ─────────────────────────────────────────────────────────
output = {
    "registered_model_name": registered_model_name,
    "version": version,
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": round(best_metric_value, 4),
}

results_path = os.path.join(RESULTS_DIR, "step2_s6.json")
with open(results_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[OK] Task 2 completed. Results saved to {results_path}")
print(json.dumps(output, indent=2))
