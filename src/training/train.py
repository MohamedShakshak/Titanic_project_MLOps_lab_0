import logging

from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def get_model_params(cfg: DictConfig) -> dict:
    """Return parameters for the selected model."""
    model_name = cfg.model.name
    params = cfg.model.get("params", {})

    if model_name in params:
        params = params[model_name]

    return OmegaConf.to_container(params, resolve=True) or {}


def get_model(cfg: DictConfig):
    """Instantiate model from config."""
    model_name = cfg.model.name
    model_params = get_model_params(cfg)

    if model_name == "logistic":
        return LogisticRegression(
            C=model_params.get("C", 1.0),
            max_iter=model_params.get("max_iter", 1000),
            random_state=model_params.get("random_state", 42),
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 5),
            random_state=model_params.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
