# src/titanic/features/transformers.py
import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class TitleExtractor(BaseEstimator, TransformerMixin):
    """Extracts and groups passenger title from the Name column."""

    RARE_TITLES = {
        "Lady", "Countess", "Capt", "Col", "Don",
        "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"
    }
    TITLE_MAPPING = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Title"] = (
            X["Name"]
            .str.extract(r" ([A-Za-z]+)\.", expand=False)
            .map(lambda t: self.TITLE_MAPPING.get(t, t))
            .map(lambda t: "Rare" if t in self.RARE_TITLES else t)
        )
        logger.debug("TitleExtractor: unique titles found: %s", X["Title"].unique())
        return X


class FamilySizeExtractor(BaseEstimator, TransformerMixin):
    """Creates FamilySize feature from SibSp and Parch."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        return X


class IsAloneExtractor(BaseEstimator, TransformerMixin):
    """Creates binary IsAlone flag from FamilySize."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "FamilySize" not in X.columns:
            raise ValueError(
                "FamilySize column not found. "
                "Ensure FamilySizeExtractor runs before IsAloneExtractor."
            )
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        return X