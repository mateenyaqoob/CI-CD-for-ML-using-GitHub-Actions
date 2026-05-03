import joblib
import numpy as np

def test_model_exists():
    model = joblib.load("models/model.pkl")
    assert model is not None

def test_prediction():
    model = joblib.load("models/model.pkl")
    pred = model.predict(np.array([[5, 6]]))
    assert pred[0] in [0, 1]

if __name__ == "__main__":
    test_model_exists()
    test_prediction()
    print("✅ All tests passed")import joblib
import numpy as np

def test_prediction():
    model = joblib.load("models/model.pkl")
    pred = model.predict(np.array([[5,6]]))
    assert pred[0] in [0,1]
