# src/titanic/data/load.py
from pathlib import Path
import logging

import pandas as pd
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "PassengerId", "Survived", "Pclass", "Name",
    "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
}

def load_raw_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = Path(to_absolute_path(cfg.data.raw_train_path))
    test_path = Path(to_absolute_path(cfg.data.raw_test_path))

    _validate_file(train_path)
    _validate_file(test_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info("Loaded train: %s rows, test: %s rows", len(train_df), len(test_df))

    _validate_schema(train_df)

    return train_df, test_df


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Run the download step first."
        )


def _validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    logger.info("Schema validation passed.")