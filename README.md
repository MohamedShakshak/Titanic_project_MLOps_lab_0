# Titanic Survival MLOps Pipeline with DVC

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/Cookiecutter-Data%20Science-328F97?logo=cookiecutter" alt="Cookiecutter Data Science badge" />
</a>

## Overview

This project implements an end-to-end machine learning pipeline for the Titanic survival prediction task, with a focus on reproducibility, modularity, and practical MLOps workflows.

The project now uses **DVC** to orchestrate and track the pipeline stages, outputs, metrics, and parameters. Hydra is still used for structured configuration, while DVC manages experiment flow and artifact tracking.

## Features

- **Reproducible environment** managed with `uv`
- **Pipeline orchestration with DVC**
- **Parameter tracking** through `params.yaml`
- **Config-driven workflow** using Hydra
- **Modular training code** under `src/training`
- **Multiple model options** including Logistic Regression and Random Forest
- **Tracked outputs and metrics** for datasets, models, and reports

## Tech Stack

- Python `3.11`
- DVC
- Hydra
- scikit-learn
- pandas
- NumPy
- joblib
- `uv`

## Project Structure

```text
Titanic_project_MLOps/
|-- .dvc/
|-- conf/
|   |-- config.yaml
|   |-- data/
|   |   `-- default.yaml
|   |-- model/
|   |   |-- logistic.yaml
|   |   `-- random_forest.yaml
|   `-- training/
|       `-- default.yaml
|-- data/
|   |-- external/
|   |-- interim/
|   |-- processed/
|   `-- raw/
|-- docs/
|-- models/
|-- notebooks/
|-- outputs/
|-- references/
|-- reports/
|-- src/
|   |-- __init__.py
|   `-- training/
|       |-- __init__.py
|       |-- evaluate.py
|       |-- pipeline.py
|       |-- train.py
|       |-- data/
|       |   |-- __init__.py
|       |   |-- download_data.py
|       |   `-- load.py
|       `-- features/
|           |-- __init__.py
|           |-- preprocess.py
|           `-- transformers.py
|-- tests/
|-- .dvcignore
|-- .env
|-- .gitignore
|-- dvc.lock
|-- dvc.yaml
|-- Makefile
|-- params.yaml
|-- pyproject.toml
|-- trainer.py
`-- README.md
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Titanic_project_MLOps
```

2. Create a virtual environment:

```bash
make create_environment
```

3. Activate the environment:

- Windows:

```bash
.\.venv\Scripts\activate
```

- Linux/macOS:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
make requirements
```

Or directly with:

```bash
uv sync
```

## Data Access

The dataset is downloaded automatically through `kagglehub` as part of the DVC pipeline.

Before running the project, make sure your Kaggle credentials are available, typically by placing `kaggle.json` in:

- The project root
- Your default Kaggle credentials directory

## DVC Pipeline

This project uses a two-stage DVC pipeline defined in `dvc.yaml`:

1. `download`
2. `train`

### Stage: `download`

This stage downloads the Titanic dataset and stores it in:

```text
data/raw/train.csv
```

Run it with:

```bash
dvc repro download
```

### Stage: `train`

This stage:

1. Loads the raw dataset
2. Splits the data into train and validation sets
3. Applies preprocessing
4. Trains the selected model
5. Evaluates the model
6. Saves processed data, metrics, and model artifacts

Run it with:

```bash
dvc repro train
```

### Run the Full Pipeline

To execute all DVC stages in order:

```bash
dvc repro
```

## Parameters

Project parameters are tracked in `params.yaml`. This file controls:

- Data settings
- Training split settings
- Model selection
- Model hyperparameters
- Output locations

Example model selection:

```yaml
model:
  name: random_forest
```

To switch models, update `params.yaml` and rerun:

```bash
dvc repro
```

## Hydra Usage

Hydra configuration is stored under `conf/`, while DVC-controlled values are merged from `params.yaml` at runtime.

The default stage in `conf/config.yaml` is:

```yaml
stage: all
```

You can still run the script directly if needed:

```bash
python trainer.py
```

Or run a specific stage:

```bash
python trainer.py stage=download
python trainer.py stage=train
```

For normal project usage, the recommended entry point is:

```bash
dvc repro
```

## Outputs

DVC tracks the main outputs of the pipeline, including:

- Raw data in `data/raw/`
- Processed data in `data/processed/`
- Trained models in `models/`
- Metrics in `reports/`

Examples of generated artifacts:

- `data/raw/train.csv`
- `data/processed/`
- `models/random_forest_pipeline.pkl`
- `reports/random_forest_metrics.json`

## Development Commands

Install dependencies:

```bash
make requirements
```

Run tests:

```bash
make test
```

Run lint checks:

```bash
make lint
```

Format the codebase:

```bash
make format
```

## Notes

- DVC is the main workflow runner for this repository.
- Hydra provides structured configuration, while `params.yaml` is used for DVC-tracked runtime parameters.
- The project is designed for reproducible local experimentation and introductory MLOps practice.
