from pathlib import Path
import shutil
import kagglehub

from src.config import RAW_DATA_DIR


def download_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download("yasserh/titanic-dataset")

    for file in Path(path).glob("*"):
        shutil.copy(file, RAW_DATA_DIR / "train.csv")

    print("Data downloaded to raw/")