import joblib
import numpy as np
import os

def test_model_file_exists():
    assert os.path.exists("models/model.pkl"), "Model file not found at models/model.pkl"

def test_prediction_returns_valid_class():
    model = joblib.load("models/model.pkl")
    pred = model.predict(np.array([[5, 6]]))
    assert int(pred[0]) in [0, 1], f"Expected 0 or 1, got {pred[0]}"

def test_prediction_output_shape():
    model = joblib.load("models/model.pkl")
    pred = model.predict(np.array([[5, 6]]))
    assert pred.shape == (1,), f"Expected shape (1,), got {pred.shape}"

def test_prediction_deterministic():
    model = joblib.load("models/model.pkl")
    pred1 = model.predict(np.array([[5, 6]]))
    pred2 = model.predict(np.array([[5, 6]]))
    assert int(pred1[0]) == int(pred2[0]), "Model gives different results for same input"
