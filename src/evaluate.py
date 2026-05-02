import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import sys

df = pd.read_csv("data/processed.csv")

X = df[['feature1', 'feature2']]
y = df['target']

# 22I-1936: added random_state=42 to match the same split used during training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load("models/model.pkl")

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc}")

ACCURACY_THRESHOLD = 0.5
if acc < ACCURACY_THRESHOLD:
    print(f"ERROR: Accuracy {acc:.2f} is below minimum threshold {ACCURACY_THRESHOLD}. Pipeline failed.")
    sys.exit(1)
