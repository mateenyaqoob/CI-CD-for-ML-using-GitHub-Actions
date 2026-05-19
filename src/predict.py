from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

model = joblib.load(MODELS_DIR / "model.pkl")

sample = np.array([[5,6]])

pred = model.predict(sample)

print("Prediction:", pred[0])
