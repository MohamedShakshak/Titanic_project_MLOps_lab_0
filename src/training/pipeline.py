from sklearn.pipeline import Pipeline


def build_pipeline(preprocessor, model):
    """
    Combine preprocessing + model into one sklearn pipeline
    """
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])