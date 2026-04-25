# trainer.py
from dotenv import load_dotenv
load_dotenv()

import hydra
import joblib
import logging
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from src.training.data.download_data import download_data
from src.training.data.load import load_raw_data, save_processed_data
from src.training.evaluate import evaluate_model, save_metrics
from src.training.features.preprocess import build_preprocessor
from src.training.pipeline import build_pipeline
from src.training.train import get_model

logger = logging.getLogger(__name__)

def run_download(cfg: DictConfig) -> None:
    download_data(cfg)

def run_train(cfg: DictConfig) -> None:
    train_df= load_raw_data(cfg)

    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
    )
    logger.info("Train size: %d, Val size: %d", len(X_train), len(X_val))

    processed_dir = Path(to_absolute_path(cfg.data.processed_path))
    save_processed_data(X_train, X_val, y_train, y_val, processed_dir)

    pipeline = build_pipeline(build_preprocessor(), get_model(cfg))
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    metrics = evaluate_model(pipeline, X_val, y_val)
    for k, v in metrics.items():
        logger.info("%s: %.4f", k, v)

    metrics_dir = Path(to_absolute_path(cfg.training.metrics_output_dir))
    save_metrics(metrics, cfg.model.name, metrics_dir)

    models_dir = Path(to_absolute_path(cfg.training.model_output_dir))
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / f"{cfg.model.name}_pipeline.pkl")
    logger.info("Saved pipeline to %s", models_dir)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Pipeline started — stage: %s", cfg.stage)
    logger.info("Parameters:\n%s", OmegaConf.to_yaml(cfg))

    if cfg.stage == "download":
        run_download(cfg)
    elif cfg.stage == "train":
        run_train(cfg)
    elif cfg.stage == "all":
        run_download(cfg)
        run_train(cfg)
    else:
        raise ValueError(f"Unknown stage: {cfg.stage}")

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
