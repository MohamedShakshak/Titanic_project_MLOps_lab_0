# from pathlib import Path
# import json
# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# from src.config import DATA_DIR


# PROCESSED_DATA_DIR = DATA_DIR / "processed"
# MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
# REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


# def evaluate():
#     REPORTS_DIR.mkdir(parents=True, exist_ok=True)

#     # Load data
#     data_path = PROCESSED_DATA_DIR / "train_processed.csv"
#     df = pd.read_csv(data_path)

#     X = df.drop("Survived", axis=1)
#     y = df["Survived"]

#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # Load models
#     models = {
#         "logistic_regression": joblib.load(MODELS_DIR / "logistic_regression.pkl"),
#         "random_forest": joblib.load(MODELS_DIR / "random_forest.pkl"),
#     }

#     results = {}

#     for name, model in models.items():
#         preds = model.predict(X_val)

#         results[name] = {
#             "accuracy": accuracy_score(y_val, preds),
#             "precision": precision_score(y_val, preds),
#             "recall": recall_score(y_val, preds),
#             "f1_score": f1_score(y_val, preds),
#         }

#     # -------------------------
#     # Print results (NEW)
#     # -------------------------
#     print("\nModel Evaluation Results")
#     print("-" * 30)

#     for model_name, metrics in results.items():
#         print(f"\n{model_name}")
#         for k, v in metrics.items():
#             print(f"{k}: {v:.4f}")

#     # -------------------------
#     # Save JSON
#     # -------------------------
#     json_path = REPORTS_DIR / "metrics.json"
#     with open(json_path, "w") as f:
#         json.dump(results, f, indent=4)

#     # -------------------------
#     # Save TXT
#     # -------------------------
#     txt_path = REPORTS_DIR / "metrics.txt"
#     with open(txt_path, "w") as f:
#         for model_name, metrics in results.items():
#             f.write(f"{model_name}\n")
#             for k, v in metrics.items():
#                 f.write(f"{k}: {v:.4f}\n")
#             f.write("\n")

#     print(f"\nReports saved to {REPORTS_DIR}")


# if __name__ == "__main__":
#     evaluate()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)

    return {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
        "f1_score": f1_score(y_val, preds),
    }