# Titanic Survival MLOps Pipeline

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/Cookiecutter-Data%20Science-328F97?logo=cookiecutter" alt="Cookiecutter Data Science badge" />
</a>

## Overview

This project implements an end-to-end machine learning pipeline for the Titanic survival prediction task, with an emphasis on clean structure, reproducibility, and practical MLOps workflows.

The pipeline covers:

- Automated dataset download
- Data loading and preprocessing
- Feature engineering and transformation
- Model training with configurable algorithms
- Evaluation and metric export
- Model artifact persistence
- Centralized configuration management with Hydra

## Features

- **Reproducible environment** managed with `uv`
- **Config-driven experiments** using Hydra configuration groups
- **Modular training code** under `src/training`
- **Multiple model choices** including Logistic Regression and Random Forest
- **Saved outputs** for trained models, processed data, and evaluation metrics

## Tech Stack

- Python `3.11`
- Hydra
- scikit-learn
- pandas
- NumPy
- joblib
- `uv`

## Project Structure

```text
Titanic_project_MLOps_lab_0/
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
|-- .env
|-- .gitignore
|-- Makefile
|-- pyproject.toml
|-- trainer.py
`-- README.md
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Titanic_project_MLOps_lab_0
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

4. Install project dependencies:

```bash
make requirements
```

You can also install dependencies directly with:

```bash
uv sync
```

## Data Access

The pipeline downloads the Titanic dataset automatically through `kagglehub`.

Before running the project, make sure your Kaggle credentials are available. Typically, this means placing `kaggle.json` in one of the following locations:

- The project root
- The default Kaggle credentials directory for your system

## Running the Pipeline

Run the full workflow with:

```bash
python trainer.py
```

This will:

1. Download the dataset into `data/raw/`
2. Load and split the training data
3. Build the preprocessing and modeling pipeline
4. Train the selected model
5. Evaluate performance on the validation split
6. Save metrics to `reports/`
7. Save the trained pipeline to `models/`

## Configuration

Hydra is used to manage configuration files under `conf/`.

Default configuration groups:

- `conf/data/default.yaml`
- `conf/model/logistic.yaml`
- `conf/model/random_forest.yaml`
- `conf/training/default.yaml`

Example: run the pipeline with the Random Forest model:

```bash
python trainer.py model=random_forest
```

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

## Outputs

Typical generated outputs include:

- Processed datasets in `data/processed/`
- Trained model artifacts in `models/`
- Evaluation metrics in `reports/`
- Hydra run outputs in `outputs/`

## Notes

- The repository is structured around a modular training pipeline for maintainability and extension.
- Model behavior can be changed through configuration without modifying source code.
- The current setup is suitable for experimentation, local development, and introductory MLOps practice.
