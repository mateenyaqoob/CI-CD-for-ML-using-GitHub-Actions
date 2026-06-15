from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/model.pkl")

@app.route("/")
def home():
    return "ML API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    values = np.array([data["features"]])
    pred = model.predict(values)
    return jsonify({"prediction": int(pred[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
