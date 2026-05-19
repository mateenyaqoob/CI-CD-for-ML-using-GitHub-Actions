import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime
import mlflow
import mlflow.sklearn

# Setup absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Fix Issue #31: Ensure models directory exists before saving
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df[['feature1', 'feature2']]
y = df['target']

# Fix Issue #1: Start MLflow tracking run
with mlflow.start_run():
    # Define parameters
    test_size = 0.2
    n_estimators = 100
    
    # Train/Test Split (Added random_state for reproducible CI/CD runs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Initialize and train model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate performance metric
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Log to MLflow
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Fix Issue #16: Apply timestamp versioning to the physical file
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_v_{version}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    joblib.dump(model, model_path)

    print(f"Model Trained Successfully")
    print(f"Version: {model_filename} | Accuracy: {accuracy:.4f}")