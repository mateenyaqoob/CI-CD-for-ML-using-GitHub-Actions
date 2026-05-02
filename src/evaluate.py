import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import sys
df = pd.read_csv("data/processed.csv")
X = df[['feature1', 'feature2']]
y = df['target']
# Use same random_state as training to get the same test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = joblib.load("models/model.pkl")
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc}")
# Quality gate: fail the pipeline if accuracy is below threshold
ACCURACY_THRESHOLD = 0.7
if acc < ACCURACY_THRESHOLD:
    print(f"FAILED: Accuracy {acc:.2f} is below threshold {ACCURACY_THRESHOLD}")
    sys.exit(1)
print(f"PASSED: Accuracy {acc:.2f} meets the threshold {ACCURACY_THRESHOLD}")
