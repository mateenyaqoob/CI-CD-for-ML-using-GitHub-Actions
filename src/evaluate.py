import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

ACCURACY_THRESHOLD = 0.5

df = pd.read_csv("data/processed.csv")

X = df[['feature1', 'feature2']]
y = df['target']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("models/model.pkl")

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc}")

if acc < ACCURACY_THRESHOLD:
    print(
        f"Quality gate FAILED: accuracy {acc} below threshold {ACCURACY_THRESHOLD}"
    )
    sys.exit(1)

print(f"Quality gate PASSED (>= {ACCURACY_THRESHOLD})")
