import joblib
import numpy as np


def test_prediction():
    """Test that model predictions return valid class labels (0 or 1)."""
    model = joblib.load("models/model.pkl")
    pred = model.predict(np.array([[5, 6]]))
    assert pred[0] in [0, 1], f"Prediction {pred[0]} is not a valid class label"


def test_model_exists():
    """Test that trained model file exists."""
    import os
    assert os.path.exists("models/model.pkl"), "Model file not found"


def test_prediction_shape():
    """Test that prediction returns correct output shape."""
    model = joblib.load("models/model.pkl")
    pred = model.predict(np.array([[5, 6]]))
    assert pred.shape == (1,), f"Expected shape (1,), got {pred.shape}"
