from pathlib import Path
import pandas as pd

from src.config import RAW_DATA_DIR, DATA_DIR


PROCESSED_DATA_DIR = DATA_DIR / "processed"


def preprocess_data():
    """
    Load raw Titanic data, clean it, and save processed dataset.
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_path = RAW_DATA_DIR / "train.csv"
    df = pd.read_csv(train_path)

    # Drop unnecessary columns
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encode categorical variables
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # Save processed data
    output_path = PROCESSED_DATA_DIR / "train_processed.csv"
    df.to_csv(output_path, index=False)

    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    preprocess_data()