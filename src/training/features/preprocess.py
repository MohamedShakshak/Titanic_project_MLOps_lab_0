import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .transformers import (
    TitleExtractor,
    FamilySizeExtractor,
    IsAloneExtractor,
)

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title"]


def build_preprocessor() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Feature engineering runs BEFORE the column transformer
    preprocessor = Pipeline(steps=[
        ("title", TitleExtractor()),
        ("family_size", FamilySizeExtractor()),
        ("is_alone", IsAloneExtractor()),
        ("column_transformer", column_transformer),
    ])

    return preprocessor