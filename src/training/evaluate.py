import logging

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def evaluate_model(pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Evaluate a fitted pipeline and return metrics dict."""
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "f1": f1_score(y_val, y_pred),
    }

    return metrics

def save_metrics(
    metrics: dict,
    model_name: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add metadata
    metrics["model"] = model_name
    metrics["timestamp"] = datetime.now().isoformat()

    # Save as JSON — easier to parse later than txt
    output_path = output_dir / f"{model_name}_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved metrics to %s", output_path)