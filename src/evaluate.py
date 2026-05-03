from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

df = pd.read_csv(DATA_DIR / "processed.csv")

X = df[['feature1', 'feature2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = joblib.load(MODELS_DIR / "model.pkl")

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc}")
