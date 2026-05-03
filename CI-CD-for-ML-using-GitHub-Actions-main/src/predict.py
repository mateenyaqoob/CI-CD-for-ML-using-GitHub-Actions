import joblib
import numpy as np

model = joblib.load("models/model.pkl")

sample = np.array([[5,6]])

pred = model.predict(sample)

print("Prediction:", pred[0])
