from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import DATA_DIR


PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def train():
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = PROCESSED_DATA_DIR / "train_processed.csv"
    df = pd.read_csv(data_path)

    # Split features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Model 1: Logistic Regression
    # -------------------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    lr_preds = lr.predict(X_val)
    lr_acc = accuracy_score(y_val, lr_preds)

    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    # Save model
    joblib.dump(lr, MODELS_DIR / "logistic_regression.pkl")

    # -------------------------
    # Model 2: Random Forest
    # -------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_preds)

    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # Save model
    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")


if __name__ == "__main__":
    train()