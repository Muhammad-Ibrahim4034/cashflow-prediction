## CashFlow Prediction & Forecasting

An end-to-end **Machine Learning and MLOps project** for predicting net cash flow from transaction data and macroeconomic indicators. The project includes automated feature engineering, model training and evaluation, MLflow experiment tracking, and an interactive Streamlit dashboard.

## Features

* Predicts **net cash flow** for individual transactions
* Incorporates macroeconomic indicators such as inflation and GDP
* Automated feature engineering pipeline
* Compares multiple regression models
* 30-day cash flow forecasting
* MLflow experiment tracking
* Interactive Streamlit dashboard
* Scenario analysis and visualizations
* CI/CD workflows with GitHub Actions

## Machine Learning

The project evaluates several regression models:

* Linear Regression
* Ridge Regression
* Lasso Regression
* ElasticNet
* Random Forest

**Random Forest** achieved the best performance:

| Metric    |          Result |
| --------- | --------------: |
| Test MAE  |  **$19,562.95** |
| Test RMSE | **$126,648.35** |

The model uses transaction information, engineered financial features, time-based features, and macroeconomic indicators.

## Pipeline

```text
Raw Transaction Data
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Cross-Validation & Evaluation
        ↓
MLflow Tracking
        ↓
Best Model
        ↓
Prediction / Forecasting
        ↓
Streamlit Dashboard
```

## Project Structure

```text
cashflow-prediction/
│
├── api/
│   ├── app.py
│   └── inference.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_cashflow_model.pkl
│   ├── feature_pipeline.pkl
│   └── liquidity_forecaster.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── forecast.py
│   ├── predict.py
│   ├── train.py
│   └── train_pipeline.py
│
├── mlruns/
├── .github/workflows/
├── evaluation_report.md
└── README.md
```

## Technologies

* Python
* Scikit-learn
* Pandas & NumPy
* MLflow
* Streamlit
* Plotly
* Joblib
* GitHub Actions

## Running the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run api/app.py
```

The dashboard allows users to enter transaction details and receive a predicted net cash flow, risk indication, and scenario-based analysis.

## Forecasting

The project supports forward cash-flow projections using:

* Historical cash-flow values
* Lag features
* Rolling statistics
* Day-of-week patterns
* Inflation
* GDP

The dashboard can generate projections for up to **60 days** and allows users to experiment with inflation, GDP, and volatility scenarios.

## MLOps

MLflow is used to track:

* Model parameters
* Cross-validation metrics
* Test metrics
* Training time
* Model artifacts

GitHub Actions workflows are included for automated ML pipeline and deployment processes.

## Author

**Muhammad Ibrahim**
BS Artificial Intelligence — FAST NUCES

---

If you find this project useful, consider giving the repository a star.

## Output:
<img width="1831" height="862" alt="image" src="https://github.com/user-attachments/assets/e31cf87f-9d16-4626-9f69-e49de3469566" />
<img width="1830" height="860" alt="image" src="https://github.com/user-attachments/assets/2f2c6c79-b545-4ca0-ae73-0dce734ebb04" />
<img width="1832" height="860" alt="image" src="https://github.com/user-attachments/assets/3f9ff29b-e09d-48fb-ba56-aba1c35f6ea6" />
<img width="1830" height="805" alt="image" src="https://github.com/user-attachments/assets/96a91cf0-0ec5-4c15-983b-70316b72050f" />



