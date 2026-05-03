import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

# Create DataFrame with proper feature names to avoid sklearn warning
sample = pd.DataFrame([[5, 6]], columns=['feature1', 'feature2'])

pred = model.predict(sample)

print("Prediction:", pred[0])
