from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/model.pkl")


def main():
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([[5, 6]], columns=["feature1", "feature2"])
    pred = model.predict(sample)
    print("Prediction:", pred[0])


if __name__ == "__main__":
    main()
