import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))

@app.route("/")
def home():
    return "ML API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key in request body"}), 400
    values = np.array([data["features"]])
    pred = model.predict(values)
    return jsonify({"prediction": int(pred[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
