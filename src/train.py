from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed.csv")
MODEL_PATH = Path("models/model.pkl")


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[["feature1", "feature2"]]
    y = df["target"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model Trained Successfully")


if __name__ == "__main__":
    main()
