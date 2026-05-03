import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# load dataset
df = pd.read_csv("data/processed.csv")

# features and target
X = df[['feature1', 'feature2']]
y = df['target']

# reproducible split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# reproducible model
model = RandomForestClassifier(random_state=40)
model.fit(X_train, y_train)

# ensure models directory exists
os.makedirs("models", exist_ok=True)

# save model
joblib.dump(model, "models/model.pkl")

print("Model Trained Successfully")
