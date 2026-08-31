import pandas as pd
import os

def validate_data(df):
    """Validate the input dataframe before preprocessing."""

    # Check if dataframe is empty
    if df.empty:
        raise ValueError("[Validation Error] The dataset is empty. Please provide a valid CSV file.")

    # Check for missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(f"[Validation Error] Missing values found in columns:\n{missing}")

    # Check for duplicate rows
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        print(f"[Validation Warning] {num_duplicates} duplicate row(s) found. Dropping duplicates.")
        df = df.drop_duplicates()

    # Check that there is more than one column (features + label)
    if df.shape[1] < 2:
        raise ValueError("[Validation Error] Dataset must have at least one feature column and one target column.")

    print(f"[Validation Passed] Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def preprocess():
    input_path = "data/sample.csv"
    output_path = "data/processed.csv"

    # Check file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"[Error] Data file not found at: {input_path}")

    df = pd.read_csv(input_path)
    print(f"[Info] Loaded data from {input_path}")

    # Run validation
    df = validate_data(df)

    # Drop missing values (original behavior kept)
    df.dropna(inplace=True)

    # Save processed data
    df.to_csv(output_path, index=False)
    print("Preprocessing Completed")


if __name__ == "__main__":
    preprocess()
