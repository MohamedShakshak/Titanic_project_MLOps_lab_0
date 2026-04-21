from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import DATA_DIR


PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def evaluate():
    # Load processed data
    data_path = PROCESSED_DATA_DIR / "train_processed.csv"
    df = pd.read_csv(data_path)

    # Split features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Same split as training
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load models
    lr = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    rf = joblib.load(MODELS_DIR / "random_forest.pkl")

    # Evaluate function
    def evaluate_model(model, X_val, y_val, name):
        preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds)

        print(f"\n{name}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")

    # Evaluate both models
    evaluate_model(lr, X_val, y_val, "Logistic Regression")
    evaluate_model(rf, X_val, y_val, "Random Forest")


if __name__ == "__main__":
    evaluate()