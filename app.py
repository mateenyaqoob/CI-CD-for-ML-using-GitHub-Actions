from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import glob

app = Flask(__name__)

# Setup absolute paths and dynamic model loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Find all model files
model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))

if not model_files:
    print("Warning: No model found. Please run train.py first.")
    model = None
else:
    # Automatically grab the latest model file based on creation time
    latest_model_path = max(model_files, key=os.path.getctime)
    model = joblib.load(latest_model_path)
    print(f"Loaded model: {os.path.basename(latest_model_path)}")

@app.route("/")
def home():
    return "ML API Running"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained yet"}), 500
        
    data = request.json
    values = np.array([data["features"]])
    pred = model.predict(values)
    return jsonify({"prediction": int(pred[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)