import joblib
import numpy as np

def predict(feature1, feature2):
    model = joblib.load("models/model.pkl")
    sample = np.array([[feature1, feature2]])
    pred = model.predict(sample)
    return int(pred[0])

if __name__ == "__main__":
    result = predict(5, 6)
    print("Prediction:", result)
