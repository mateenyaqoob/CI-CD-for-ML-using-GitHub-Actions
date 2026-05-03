import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load processed data
df = pd.read_csv("data/processed.csv")

# Features and target
X = df[['feature1', 'feature2']]
y = df['target']

# Same split as training (for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
model = joblib.load("models/model.pkl")

# Predict
pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc}")
