import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

try:
	import mlflow
	import mlflow.sklearn
except Exception:
	mlflow = None


def resolve_model_version():
	env_version = os.getenv("MODEL_VERSION")
	if env_version:
		return env_version
	github_sha = os.getenv("GITHUB_SHA")
	if github_sha:
		return github_sha[:7]
	return datetime.utcnow().strftime("%Y%m%d%H%M%S")


def log_mlflow(model, version, model_path, test_size, accuracy, train_rows, test_rows):
	if mlflow is None:
		return
	tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
	experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "ml-cicd")
	registered_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment(experiment_name)
	with mlflow.start_run(run_name=f"train-{version}"):
		mlflow.log_param("model_version", version)
		mlflow.log_param("test_size", test_size)
		mlflow.log_param("n_estimators", model.n_estimators)
		if model.random_state is not None:
			mlflow.log_param("random_state", model.random_state)
		mlflow.log_metric("accuracy", accuracy)
		mlflow.log_metric("train_rows", train_rows)
		mlflow.log_metric("test_rows", test_rows)
		if registered_name:
			mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=registered_name)
		else:
			mlflow.sklearn.log_model(model, artifact_path="model")
		mlflow.log_artifact(str(model_path))


df = pd.read_csv("data/processed.csv")

X = df[["feature1", "feature2"]]
y = df["target"]

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=test_size, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

version = resolve_model_version()
versioned_path = models_dir / f"model_{version}.pkl"
joblib.dump(model, versioned_path)
joblib.dump(model, models_dir / "model.pkl")
(models_dir / "model_version.txt").write_text(version)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
log_mlflow(model, version, versioned_path, test_size, acc, len(X_train), len(X_test))

print(f"Model Trained Successfully (version: {version})")
