from pathlib import Path
import shutil
import logging
import kagglehub

from dotenv import load_dotenv
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

load_dotenv()  # loads .env into environment variables automatically

logger = logging.getLogger(__name__)


def download_data(cfg: DictConfig) -> None:
    raw_dir = Path(to_absolute_path(cfg.data.raw_dir))
    raw_dir.mkdir(parents=True, exist_ok=True)

    competition = cfg.data.competition_name

    logger.info("Downloading competition data: %s", competition)

    downloaded_path = Path(
        kagglehub.dataset_download("yasserh/titanic-dataset")
    )

    for file in downloaded_path.glob("*"):
        destination = raw_dir / "train.csv"

        if destination.exists() and not cfg.data.overwrite:
            logger.info("Skipping existing file: %s", file.name)
            continue

        shutil.copy(file, destination)
        logger.info("Copied %s", file.name)

    logger.info("Download complete.")