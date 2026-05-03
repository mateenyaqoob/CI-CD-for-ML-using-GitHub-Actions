import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load processed data
df = pd.read_csv("data/processed.csv")

# Features and target
X = df[['feature1', 'feature2']]
y = df['target']

# Train-test split (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model Trained Successfully")
