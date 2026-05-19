import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import glob

# Setup absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load data
df = pd.read_csv(DATA_PATH)
X = df[['feature1', 'feature2']]
y = df['target']

# CRITICAL FIX: Ensure test_size and random_state perfectly match train.py
# Otherwise, you are evaluating on data the model may have been trained on!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Find all model files in the directory
model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))

if not model_files:
    raise FileNotFoundError(f"No trained models found in {MODELS_DIR}. Please run train.py first.")

# Automatically grab the latest model file based on creation time
latest_model_path = max(model_files, key=os.path.getctime)
print(f"Evaluating latest model: {os.path.basename(latest_model_path)}")

# Load the model
model = joblib.load(latest_model_path)

# Predict and calculate accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc:.4f}")