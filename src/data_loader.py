import pandas as pd
from sklearn.model_selection import train_test_split
import os


LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def load_raw_data(path: str = "data/raw/train.csv") -> pd.DataFrame:
    """Load raw training data from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at path: {path}")

    df = pd.read_csv(path)

    print(f"Number of rows loaded: {len(df)}")
    print(f"Column names: list({df.columns.tolist()})")
    label_sums = {}
    for column in LABELS:
        if column in df.columns:
            label_sums[column] = int(df[column].sum())
        else:
            label_sums[column] = 0
    print(f"Label sums (counts): {label_sums}")

    return df


def get_features_and_labels(df: pd.DataFrame) -> tuple:
    """Extract features and labels from the dataframe."""
    X = df["comment_text"].fillna("").astype(str)
    y = df[LABELS]

    return X, y


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Split the dataframe into train and test sets."""
    X, y = get_features_and_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    return (X_train, X_test, y_train, y_test)

