import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

df = pd.read_csv("data/processed.csv")

X = df[['feature1', 'feature2']]
y = df['target']

# reproducible split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_path = "models/model.pkl"

# safety check (important for CI/CD)
if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found. Run train.py first.")

model = joblib.load(model_path)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc}")