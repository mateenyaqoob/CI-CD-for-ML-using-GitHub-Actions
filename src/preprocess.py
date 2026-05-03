from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "sample.csv")
df.dropna(inplace=True)
df.to_csv(BASE_DIR / "data" / "processed.csv", index=False)

print("Preprocessing Completed")
