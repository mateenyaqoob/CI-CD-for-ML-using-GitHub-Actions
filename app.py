from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "models/model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run src/train.py first.")
    return joblib.load(MODEL_PATH)

model = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "feature1" not in data or "feature2" not in data:
        return jsonify({"error": "Request must include feature1 and feature2"}), 400
    features = np.array([[data["feature1"], data["feature2"]]])
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
