from src.training.download_data import download_data
from src.training.preprocess_data import preprocess_data
from src.training.train import train
from src.training.evaluate import evaluate


def run_pipeline():
    print("Step 1: Downloading data")
    download_data()

    print("Step 2: Preprocessing data")
    preprocess_data()

    print("Step 3: Training models")
    train()

    print("Step 4: Evaluating models")
    evaluate()

    print("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()