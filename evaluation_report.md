# Cashflow Prediction Model Evaluation Report

This report evaluates multiple baseline, penalized linear, and gradient boosting algorithms on predicting the net cash flow. A 5-Fold Cross Validation was performed on the training set to ensure robustness.

| Model | CV MAE | CV RMSE | Test MAE | Test RMSE |
|-------|--------|---------|----------|-----------|
| Linear Regression | $82,230.11 | $268,701.30 | $81,910.01 | $261,750.69 |
| Ridge | $80,468.63 | $268,878.19 | $80,376.52 | $261,772.06 |
| Lasso | $81,036.56 | $270,640.22 | $80,522.43 | $262,986.21 |
| ElasticNet | $73,987.63 | $280,871.86 | $73,664.98 | $275,163.38 |
| Random Forest | $20,626.06 | $128,718.12 | $19,562.95 | $126,648.35 |

## Artifacts Generated
- **Feature Pipeline:** `models/feature_pipeline.pkl`
- **Best Model (Random Forest):** `models/best_cashflow_model.pkl`
