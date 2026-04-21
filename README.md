# MLOps_lab_0

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# MLOps_Lab0

A simple end-to-end MLOps pipeline built on the Titanic dataset.  
The project demonstrates a full machine learning workflow including data ingestion, preprocessing, model training, evaluation, and report generation.

---

## Pipeline Overview

This project implements a complete ML pipeline:

1. Download dataset from Kaggle (via kagglehub)
2. Preprocess raw data
3. Train multiple machine learning models
4. Evaluate models using multiple metrics
5. Save results into reports

---

## How to Run the Full Pipeline

```bash
python -m src.trainer
```
------------------
## Project Organization
```
├── data
│   ├── external
│   ├── interim
│   ├── processed        <- Cleaned dataset after preprocessing
│   └── raw              <- Raw downloaded Titanic data
│
├── models               <- Saved trained models (.pkl files)
│
├── notebooks            <- Jupyter notebooks for exploration
│
├── reports              <- Evaluation outputs
│   ├── metrics.json     <- Machine-readable results
│   └── metrics.txt      <- Human-readable report
│
├── src
│   ├── config.py        <- Project configuration (paths, constants)
│   ├── dataset.py       <- Kaggle dataset downloader (kagglehub)
│   ├── features.py      <- Data preprocessing & feature engineering
│   │
│   └── modeling
│       ├── train.py     <- Train Logistic Regression & Random Forest
│       ├── evaluate.py  <- Evaluate models and generate reports
│       └── predict.py   <- (optional inference script)
│
├── trainer.py           <- Full pipeline orchestrator
├── requirements.txt
├── pyproject.toml
└── README.md