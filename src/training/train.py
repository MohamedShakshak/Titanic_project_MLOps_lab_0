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

