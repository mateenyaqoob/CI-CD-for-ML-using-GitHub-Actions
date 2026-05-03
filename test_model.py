import os
import joblib
import numpy as np

def test_prediction():
    model_path = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
    model = joblib.load(model_path)
    pred = model.predict(np.array([[5, 6]]))
    assert pred[0] in [0, 1]
