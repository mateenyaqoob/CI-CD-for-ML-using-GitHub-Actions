import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("data/processed.csv")

X = df[['feature1', 'feature2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = joblib.load("models/model.pkl")

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc}")

ACCURACY_THRESHOLD = 0.7
if acc < ACCURACY_THRESHOLD:
    raise ValueError(f"Model accuracy {acc:.2f} below threshold {ACCURACY_THRESHOLD}")
