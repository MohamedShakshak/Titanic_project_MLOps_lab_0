from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Kaggle dataset
KAGGLE_DATASET = "competitions/titanic"