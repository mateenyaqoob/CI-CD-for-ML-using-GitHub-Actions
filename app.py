from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = Path("models/model.pkl")

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route("/")
def home():
    return "ML API Running"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not trained yet"}), 503

    data = request.json
    if not data or "features" not in data:
        return jsonify({"error": "Request must include a 'features' array"}), 400

    values = pd.DataFrame([data["features"]], columns=["feature1", "feature2"])
    pred = model.predict(values)
    return jsonify({"prediction": int(pred[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
