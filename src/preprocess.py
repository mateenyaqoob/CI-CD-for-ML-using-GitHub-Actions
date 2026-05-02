import pandas as pd
import sys

# Fix: added input validation for required columns and data integrity
REQUIRED_COLUMNS = ['feature1', 'feature2', 'target']

df = pd.read_csv("data/sample.csv")

# Validate required columns exist
missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing_cols:
    print(f"ERROR: Missing required columns: {missing_cols}")
    sys.exit(1)

# Validate no all-NaN columns
if df[REQUIRED_COLUMNS].isnull().all().any():
    print("ERROR: One or more required columns are entirely NaN.")
    sys.exit(1)

df.dropna(inplace=True)
df.to_csv("data/processed.csv", index=False)

print("Preprocessing Completed")
