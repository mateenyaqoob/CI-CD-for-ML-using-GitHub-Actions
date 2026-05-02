from sklearn.metrics import accuracy_score
import joblib

model = joblib.load("models/model.pkl")
X_test, y_test = joblib.load("models/test_data.pkl")

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc}")
