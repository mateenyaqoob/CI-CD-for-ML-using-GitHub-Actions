from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed.csv")
MODEL_PATH = Path("models/model.pkl")


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[["feature1", "feature2"]]
    y = df["target"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = joblib.load(MODEL_PATH)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(f"Accuracy: {acc}")


if __name__ == "__main__":
    main()
