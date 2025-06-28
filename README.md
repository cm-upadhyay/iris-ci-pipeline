# IRIS CI Pipeline Homework

This repository demonstrates a Continuous Integration (CI) pipeline for an IRIS classification project. It includes:

- **`data/iris.csv`**: The dataset used for training and evaluation.
- **`train.py`**: Script for loading the IRIS dataset, training a Logistic Regression model, and saving the model along with its evaluation metrics.
- **`test_pipeline.py`**: Pytest suite for data validation and model evaluation.
- **`.github/workflows/ci.yaml`**: GitHub Actions workflow that automates the training, testing, and reporting process on every Pull Request to `main` and push to `dev`.
- **CML (Continuous Machine Learning)**: Used in the CI pipeline to report Pytest results and model metrics directly as comments on GitHub Pull Requests.

---

## Getting Started

1.  **Clone the Repository:**
    https://github.com/cm-upadhyay/iris-ci-pipeline.git

2.  Place `iris.csv` inside the `data/` directory.

3.  **Create and Activate Virtual Environment:**
    python -m venv .env
    source .env/bin/activate

4.  **Install Dependencies:**
    pip install -r requirements.txt

5.  **Run Training Locally:**
    python train.py
    This will generate `iris_model.pkl` and `metrics.json`.

6.  **Run Tests Locally:**
    pytest

## CI Pipeline

The `ci.yaml` GitHub Actions workflow will automatically:
1.  Set up the Python environment and install dependencies.
2.  Run `train.py` to train the model and save metrics.
3.  Execute `pytest` for data validation and model evaluation.
4.  Use CML to post the Pytest report and model metrics as a comment on the Pull Request.