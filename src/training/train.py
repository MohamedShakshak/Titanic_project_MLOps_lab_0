import logging

import hydra
import joblib
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.training.data.load import load_raw_data
from src.training.features.preprocess import build_preprocessor
from src.training.pipeline import build_pipeline
from src.training.evaluate import evaluate_model

logger = logging.getLogger(__name__)

def get_model(cfg: DictConfig):
    """Instantiate model from config."""
    if cfg.model.name == "logistic":
        return LogisticRegression(
            C=cfg.model.C,
            max_iter=cfg.model.max_iter,
            random_state=cfg.model.random_state,
        )
    elif cfg.model.name == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            random_state=cfg.model.random_state,
        )
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")




# @hydra.main(config_path="../../conf", config_name="config", version_base=None)
# def train(cfg: DictConfig) -> None:
#     logger.info("Starting training — model: %s", cfg.model.name)

#     # Load raw data via load.py
#     train_df, _ = load_raw_data(cfg)

#     # Split features and target
#     X = train_df.drop(columns=["Survived"])
#     y = train_df["Survived"]

#     # Split train/validation
#     X_train, X_val, y_train, y_val = train_test_split(
#         X,
#         y,
#         test_size=cfg.training.test_size,
#         random_state=cfg.training.random_state,
#     )
#     logger.info("Train size: %d, Val size: %d", len(X_train), len(X_val))

#     # Build pipeline
#     preprocessor = build_preprocessor()
#     model = get_model(cfg)
#     pipeline = build_pipeline(preprocessor, model)

#     # Train
#     pipeline.fit(X_train, y_train)
#     logger.info("Training complete.")

#     # Evaluate
#     metrics = evaluate_model(pipeline, X_val, y_val)
#     for k, v in metrics.items():
#         logger.info("%s: %.4f", k, v)

#     # Save artifact
#     models_dir = Path(to_absolute_path(cfg.training.model_output_dir))
#     models_dir.mkdir(parents=True, exist_ok=True)
#     output_path = models_dir / f"{cfg.model.name}_pipeline.pkl"
#     joblib.dump(pipeline, output_path)
#     logger.info("Saved pipeline to %s", output_path)


# if __name__ == "__main__":
#     train()
