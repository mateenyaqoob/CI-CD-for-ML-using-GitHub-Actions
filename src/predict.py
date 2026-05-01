import joblib
import pandas as pd

model, columns = joblib.load("models/model.pkl")

sample = pd.DataFrame([[5, 6]], columns=columns)

pred = model.predict(sample)

print("Prediction:", pred[0])