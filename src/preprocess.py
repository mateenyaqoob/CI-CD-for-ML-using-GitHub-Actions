from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

df = pd.read_csv(DATA_DIR / "sample.csv")
df.dropna(inplace=True)
df.to_csv(DATA_DIR / "processed.csv", index=False)

print("Preprocessing Completed")
