import joblib
import pandas as pd

def test_prediction():
    model = joblib.load("models/model.pkl")
    pred = model.predict(pd.DataFrame([[5,6]], columns=['feature1', 'feature2']))
    assert pred[0] in [0,1]
