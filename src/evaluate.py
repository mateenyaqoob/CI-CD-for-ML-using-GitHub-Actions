import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import sys

df = pd.read_csv("data/processed.csv")

X = df[['feature1', 'feature2']]
y = df['target']

# Fix #3: same random_state as train.py — evaluates on the true held-out test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load("models/model.pkl")

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc:.4f}")

# Fix #7: accuracy gate — fail the pipeline if model quality is below threshold
ACCURACY_THRESHOLD = 0.70
if acc < ACCURACY_THRESHOLD:
    print(f"ERROR: Accuracy {acc:.4f} is below the required threshold of {ACCURACY_THRESHOLD}.")
    sys.exit(1)

print("Model passed accuracy threshold check.")
