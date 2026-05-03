import joblib
import numpy as np


def test_prediction():
    model = joblib.load("models/model.pkl")
    prediction = model.predict(np.array([[5, 6]]))

    assert prediction[0] in [0, 1]