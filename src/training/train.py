from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import DATA_DIR
from src.training.preprocess import build_preprocessor
from src.training.pipeline import build_pipeline
from src.training.evaluate import evaluate_model


PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(PROCESSED_DATA_DIR / "train_processed.csv")

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessor
    preprocessor = build_preprocessor(X_train)

    # Models
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for name, model in models.items():

        pipeline = build_pipeline(preprocessor, model)

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_val, y_val)

        results[name] = metrics

        # Save full pipeline
        joblib.dump(pipeline, MODELS_DIR / f"{name}_pipeline.pkl")

        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    # Save report
    report_path = REPORTS_DIR / "metrics.txt"
    with open(report_path, "w") as f:
        for name, metrics in results.items():
            f.write(f"{name}\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n")

    print(f"\nSaved models to {MODELS_DIR}")
    print(f"Saved reports to {REPORTS_DIR}")


if __name__ == "__main__":
    train()