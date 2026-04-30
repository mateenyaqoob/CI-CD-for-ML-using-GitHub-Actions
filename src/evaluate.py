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

# Adding a Quality gate
THRESHOLD = 0.75

if acc < THRESHOLD:
    print(f" Model failed. Accuracy {acc:.2f} is below threshold {THRESHOLD}")
    sys.exit(1)
else:
    print(f" Model passed. Accuracy {acc:.2f} is above threshold {THRESHOLD}")
