import joblib
import pandas as pd


def test_prediction():
    model = joblib.load("models/model.pkl")
    sample = pd.DataFrame([[5, 6]], columns=["feature1", "feature2"])
    pred = model.predict(sample)
    assert pred[0] in [0, 1], f"Unexpected prediction value: {pred[0]}"


def test_prediction_shape():
    model = joblib.load("models/model.pkl")
    samples = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["feature1", "feature2"])
    preds = model.predict(samples)
    assert len(preds) == 3, "Expected 3 predictions for 3 samples"


def test_prediction_type():
    model = joblib.load("models/model.pkl")
    sample = pd.DataFrame([[5, 6]], columns=["feature1", "feature2"])
    pred = model.predict(sample)
    assert isinstance(int(pred[0]), int), "Prediction should be castable to int"
