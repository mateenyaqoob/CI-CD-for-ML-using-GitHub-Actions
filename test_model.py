import os
import joblib
import numpy as np
import pytest


MODEL_PATH = "models/model.pkl"


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"


def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None


@pytest.mark.parametrize("features", [[1, 2], [5, 6], [8, 9]])
def test_prediction_returns_valid_class(features):
    model = joblib.load(MODEL_PATH)
    pred = model.predict(np.array([features]))
    assert pred[0] in [0, 1]


def test_prediction_shape():
    model = joblib.load(MODEL_PATH)
    pred = model.predict(np.array([[5, 6], [1, 2]]))
    assert pred.shape == (2,)
