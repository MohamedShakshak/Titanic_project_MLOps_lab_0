from pathlib import Path
import shutil
import kagglehub

from src.config import RAW_DATA_DIR


def download_data():
    """
    Download Titanic dataset using kagglehub
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


    # Example dataset (public Titanic dataset)
    path = kagglehub.dataset_download("heptapod/titanic")

    print(f"Downloaded to: {path}")

    # Copy files into data/raw
    for file in Path(path).glob("*"):
        shutil.copy(file, RAW_DATA_DIR / file.name)

    print(f"Data copied to {RAW_DATA_DIR}")


if __name__ == "__main__":
    download_data()