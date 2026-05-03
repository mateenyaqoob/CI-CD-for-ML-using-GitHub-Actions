import pandas as pd

df = pd.read_csv("data/sample.csv")

required_cols = ['feature1', 'feature2', 'target']
missing = set(required_cols) - set(df.columns)
assert not missing, f"Missing expected columns: {missing}"

print(f"Rows before dropna: {len(df)}")
df.dropna(inplace=True)
print(f"Rows after dropna: {len(df)}")

df.to_csv("data/processed.csv", index=False)

print("Preprocessing Completed")
