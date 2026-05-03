import os
import joblib
import numpy as np

def test_prediction():
    assert os.path.exists("models/model.pkl"), "Model file not found!"

    model = joblib.load("models/model.pkl")
    pred = model.predict(np.array([[5, 6]]))

    assert pred[0] in [0, 1]
