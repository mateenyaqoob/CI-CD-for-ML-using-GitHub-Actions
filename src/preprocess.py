from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "sample.csv"
OUTPUT_PATH = DATA_DIR / "processed.csv"


def main():
    df = pd.read_csv(INPUT_PATH)
    df.dropna(inplace=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Preprocessing Completed")


if __name__ == "__main__":
    main()
