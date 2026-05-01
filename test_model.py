import joblib
import pandas as pd

def test_prediction():
    model, columns = joblib.load("models/model.pkl")

    sample = pd.DataFrame([[5, 6]], columns=columns)

    pred = model.predict(sample)

    assert pred[0] in [0, 1]