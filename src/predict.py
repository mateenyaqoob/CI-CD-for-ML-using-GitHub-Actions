import joblib
import numpy as np
import os
import glob

# Setup absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Find all model files in the directory
model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))

if not model_files:
    raise FileNotFoundError(f"No trained models found in {MODELS_DIR}. Please run train.py first.")

# Automatically grab the latest model file based on creation time
latest_model_path = max(model_files, key=os.path.getctime)
print(f"Loading latest model: {os.path.basename(latest_model_path)}")

# Load the model
model = joblib.load(latest_model_path)

# Prepare sample and predict
sample = np.array([[5, 6]])
pred = model.predict(sample)

print("Prediction:", pred[0])