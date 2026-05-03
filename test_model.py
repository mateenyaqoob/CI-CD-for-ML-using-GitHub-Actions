import joblib
import numpy as np
import pandas as pd

def test_prediction():
    model = joblib.load("models/model.pkl")
    # Create DataFrame with proper feature names to avoid sklearn warning
    test_data = pd.DataFrame([[5, 6]], columns=['feature1', 'feature2'])
    pred = model.predict(test_data)
    assert pred[0] in [0, 1]
