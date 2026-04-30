from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Fix #11: lazy-load model on first request instead of crashing at import time
_model = None

def get_model():
    global _model
    if _model is None:
        model_path = "models/model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'. Run the training pipeline first.")
        _model = joblib.load(model_path)
    return _model

@app.route("/")
def home():
    return "ML API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    values = np.array([data["features"]])
    model = get_model()
    pred = model.predict(values)
    return jsonify({"prediction": int(pred[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
