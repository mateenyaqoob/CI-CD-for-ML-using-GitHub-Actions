import pandas as pd


df = pd.read_csv("data/sample.csv")
df.dropna(inplace=True)
df.to_csv("data/processed.csv", index=False)

print("Preprocessing Completed")