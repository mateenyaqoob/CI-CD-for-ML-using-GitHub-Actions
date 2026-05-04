import pandas as pd
import os

# Build absolute paths based on the script's location
# __file__ is src/preprocess.py, so we go up two levels to the root repo directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "sample.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed.csv")

# Load, process, and save
df = pd.read_csv(INPUT_PATH)
df.dropna(inplace=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Preprocessing Completed")