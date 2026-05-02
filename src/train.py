import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/processed.csv")

X = df[['feature1', 'feature2']]
y = df['target']

# 22I-1936:  added random_state=42 to ensure consistent split between train and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 22I-1936: create models/ directory if it doesn't exist
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Model Trained Successfully")
