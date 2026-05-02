import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load data
df = pd.read_csv("data/processed.csv")

X = df[['feature1', 'feature2']]
y = df['target']

#  reproducibility fix
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  stable model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# prediction check
pred = model.predict(X_test)


# save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Model Trained Successfully")
