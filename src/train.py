import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Import the feature pipeline from our module
from feature_engineering import feature_pipeline

# -- Paths --
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cashflow_prediction_dataset_100k.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    print("-- 1. Loading Data ------------------------------")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"Data loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: Could not find processed data at {PROCESSED_DATA_PATH}")
        return

    # Stratified split based on risk_score to ensure representation
    strat = df["risk_score"] if "risk_score" in df.columns else None
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=strat)
    print(f"Train rows: {len(df_train):,} | Test rows: {len(df_test):,}")

    print("\n-- 2. Feature Engineering -----------------------")
    result = feature_pipeline(
        df_train=df_train,
        df_infer=df_test,
        scale=True,
        encode_categoricals=True,
        drop_raw_balance_cols=True,
    )

    X_train = result["X_train"]
    X_test = result["X_infer"]
    y_train = result["y_train"]
    y_test = result["y_infer"]
    pipe = result["pipeline"]

    # The ML task is predicting Cash Flow
    # We will use 'net_cash_flow' as our target
    target_col = "net_cash_flow"
    y_train_cf = y_train[target_col]
    y_test_cf = y_test[target_col]

    # Save the fitted pipeline
    pipeline_path = os.path.join(MODELS_DIR, "feature_pipeline.pkl")
    joblib.dump(pipe, pipeline_path)
    print(f"Feature pipeline saved to {pipeline_path}")

    print("\n-- 3. Model Training & Cross-Validation ---------")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42),
        "ElasticNet": ElasticNet(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, 
            max_depth=8, 
            min_samples_leaf=20,
            max_features='sqrt',
            n_jobs=-1, 
            random_state=42
        )
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, verbose=-1)
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, random_seed=42, verbose=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []

    best_test_rmse = float('inf')
    best_model_name = ""
    best_model = None

    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Cross-validation
        cv_mae_list = []
        cv_rmse_list = []
        
        # Convert X_train and y_train_cf to numpy arrays/dataframes suitable for iloc indexing
        X_train_cv = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_cv = y_train_cf.values if isinstance(y_train_cf, pd.Series) else y_train_cf

        for train_idx, val_idx in kf.split(X_train_cv):
            X_tr, X_val = X_train_cv[train_idx], X_train_cv[val_idx]
            y_tr, y_val = y_train_cv[train_idx], y_train_cv[val_idx]
            
            model.fit(X_tr, y_tr)
            val_preds = model.predict(X_val)
            cv_mae_list.append(mean_absolute_error(y_val, val_preds))
            cv_rmse_list.append(root_mean_squared_error(y_val, val_preds))
            
        cv_mae = np.mean(cv_mae_list)
        cv_rmse = np.mean(cv_rmse_list)
        
        # Train on full train set and evaluate on test set
        model.fit(X_train, y_train_cf)
        test_preds = model.predict(X_test)
        test_mae = mean_absolute_error(y_test_cf, test_preds)
        test_rmse = root_mean_squared_error(y_test_cf, test_preds)
        
        print(f"  > CV RMSE: ${cv_rmse:,.2f} | Test RMSE: ${test_rmse:,.2f}")
        
        results.append({
            "Model": name,
            "CV MAE": f"${cv_mae:,.2f}",
            "CV RMSE": f"${cv_rmse:,.2f}",
            "Test MAE": f"${test_mae:,.2f}",
            "Test RMSE": f"${test_rmse:,.2f}"
        })

        # Save best model logic based on Test RMSE
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_model_name = name
            best_model = model

    print("\n-- 4. Saving Artifacts --------------------------")
    best_model_path = os.path.join(MODELS_DIR, "best_cashflow_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"Best model ({best_model_name}) saved to {best_model_path}")

    # Generate Evaluation Report
    report_path = os.path.join(BASE_DIR, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("# Cashflow Prediction Model Evaluation Report\n\n")
        f.write("This report evaluates multiple baseline, penalized linear, and gradient boosting algorithms on predicting the net cash flow. A 5-Fold Cross Validation was performed on the training set to ensure robustness.\n\n")
        
        # Write Markdown Table
        f.write("| Model | CV MAE | CV RMSE | Test MAE | Test RMSE |\n")
        f.write("|-------|--------|---------|----------|-----------|\n")
        for res in results:
            f.write(f"| {res['Model']} | {res['CV MAE']} | {res['CV RMSE']} | {res['Test MAE']} | {res['Test RMSE']} |\n")
        
        f.write("\n## Artifacts Generated\n")
        f.write(f"- **Feature Pipeline:** `models/feature_pipeline.pkl`\n")
        f.write(f"- **Best Model ({best_model_name}):** `models/best_cashflow_model.pkl`\n")

    print(f"\nEvaluation report generated at {report_path}")

if __name__ == "__main__":
    main()
