import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SRC_DIR, "..")
sys.path.insert(0, SRC_DIR)

from feature_engineering import feature_pipeline

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

PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cashflow_prediction_dataset_100k.csv")
MODELS_DIR          = os.path.join(BASE_DIR, "models")
MLRUNS_DIR          = os.path.join(BASE_DIR, "mlruns")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

EXPERIMENT_NAME = "cashflow-prediction"
mlflow.set_tracking_uri(Path(MLRUNS_DIR).as_uri())
mlflow.set_experiment(EXPERIMENT_NAME)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def get_models():
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {"fit_intercept": True}
        },
        "Ridge": {
            "model": Ridge(alpha=1.0, random_state=RANDOM_STATE),
            "params": {"alpha": 1.0}
        },
        "Lasso": {
            "model": Lasso(alpha=0.1, random_state=RANDOM_STATE),
            "params": {"alpha": 0.1}
        },
        "ElasticNet": {
            "model": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE),
            "params": {"alpha": 0.1, "l1_ratio": 0.5}
        },
        "RandomForest": {
            "model": RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=20,
                max_features="sqrt",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 8,
                "min_samples_leaf": 20,
                "max_features": "sqrt"
            }
        },
    }
    if HAS_XGB:
        models["XGBoost"] = {
            "model": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  n_jobs=-1, random_state=RANDOM_STATE),
            "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        }
    if HAS_LGBM:
        models["LightGBM"] = {
            "model": LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   n_jobs=-1, random_state=RANDOM_STATE, verbose=-1),
            "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        }
    if HAS_CATBOOST:
        models["CatBoost"] = {
            "model": CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1,
                                       random_seed=RANDOM_STATE, verbose=0),
            "params": {"iterations": 100, "depth": 6, "learning_rate": 0.1}
        }
    return models


def load_data():
    print("\n── Step 1: Loading processed data ───────────────────────────────")
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Processed data not found at:\n  {PROCESSED_DATA_PATH}\n"
            "Run data_preprocessing.py first."
        )
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

    # Create target columns if missing (raw CSV doesn't include them)
    if "net_cash_flow" not in df.columns:
        df["net_cash_flow"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
        print("  Created net_cash_flow column from balance difference")
    if "risk_score" not in df.columns:
        df["risk_score"] = 0
        print("  Created placeholder risk_score column")

    return df


def engineer_features(df):
    print("\n── Step 2: Feature engineering ──────────────────────────────────")
    strat = df["risk_score"] if "risk_score" in df.columns else None
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=strat
    )
    print(f"  Train rows : {len(df_train):,}")
    print(f"  Test rows  : {len(df_test):,}")

    result = feature_pipeline(
        df_train=df_train,
        df_infer=df_test,
        scale=True,
        encode_categoricals=True,
        drop_raw_balance_cols=True,
    )

    X_train = result["X_train"]
    X_test  = result["X_infer"]
    if result["y_train"] is None or "net_cash_flow" not in result["y_train"].columns:
        raise ValueError(f"net_cash_flow column not found. Available columns: {result['y_train'].columns.tolist() if result['y_train'] is not None else 'None'}")
    y_train = result["y_train"]["net_cash_flow"]
    y_test  = result["y_infer"]["net_cash_flow"]
    pipe = result["pipeline"]

    # Save feature pipeline artifact
    pipeline_path = os.path.join(MODELS_DIR, "feature_pipeline.pkl")
    joblib.dump(pipe, pipeline_path)
    print(f"  Feature pipeline saved → {pipeline_path}")
    print(f"  Feature count : {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, pipe, pipeline_path


def train_and_log(name, model_def, X_train, X_test, y_train, y_test, pipeline_path):
    model  = model_def["model"]
    params = model_def["params"]

    with mlflow.start_run(run_name=name):

        mlflow.log_param("model_name",   name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size",    0.2)
        mlflow.log_param("cv_folds",     5)
        mlflow.log_param("target",       "net_cash_flow")

        for k, v in params.items():
            mlflow.log_param(k, v)

        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        X_cv = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_cv = y_train.values if isinstance(y_train, pd.Series)    else y_train

        cv_maes, cv_rmses = [], []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cv), 1):
            X_tr, X_val = X_cv[tr_idx], X_cv[val_idx]
            y_tr, y_val = y_cv[tr_idx], y_cv[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            cv_maes.append(mean_absolute_error(y_val, preds))
            cv_rmses.append(root_mean_squared_error(y_val, preds))

        cv_mae  = float(np.mean(cv_maes))
        cv_rmse = float(np.mean(cv_rmses))

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        test_preds = model.predict(X_test)
        test_mae   = float(mean_absolute_error(y_test, test_preds))
        test_rmse  = float(root_mean_squared_error(y_test, test_preds))

        mlflow.log_metric("cv_mae",     cv_mae)
        mlflow.log_metric("cv_rmse",    cv_rmse)
        mlflow.log_metric("test_mae",   test_mae)
        mlflow.log_metric("test_rmse",  test_rmse)
        mlflow.log_metric("train_time_seconds", round(train_time, 3))

        mlflow.sklearn.log_model(model, artifact_path="model")

        mlflow.log_artifact(pipeline_path, artifact_path="pipeline")

        run_id = mlflow.active_run().info.run_id
        print(f"  [{name}] CV RMSE: ${cv_rmse:,.2f} | Test RMSE: ${test_rmse:,.2f} | run_id: {run_id[:8]}...")

    return {
        "name":       name,
        "model":      model,
        "cv_mae":     cv_mae,
        "cv_rmse":    cv_rmse,
        "test_mae":   test_mae,
        "test_rmse":  test_rmse,
        "run_id":     run_id,
    }


def save_best_model(results):
    print("\n── Step 4: Saving best model ────────────────────────────────────")
    best = min(results, key=lambda r: r["test_rmse"])

    best_model_path = os.path.join(MODELS_DIR, "best_cashflow_model.pkl")
    joblib.dump(best["model"], best_model_path)
    print(f"  Best model  : {best['name']}")
    print(f"  Test RMSE   : ${best['test_rmse']:,.2f}")
    print(f"  Test MAE    : ${best['test_mae']:,.2f}")
    print(f"  Saved to    : {best_model_path}")

    with mlflow.start_run(run_name="BEST_MODEL"):
        mlflow.log_param("best_model_name", best["name"])
        mlflow.log_metric("best_test_mae",  best["test_mae"])
        mlflow.log_metric("best_test_rmse", best["test_rmse"])
        mlflow.log_metric("best_cv_mae",    best["cv_mae"])
        mlflow.log_metric("best_cv_rmse",   best["cv_rmse"])
        mlflow.log_artifact(best_model_path, artifact_path="best_model")

    return best


def write_report(results, best):
    print("\n── Step 5: Writing evaluation report ────────────────────────────")
    report_path = os.path.join(BASE_DIR, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("# Cashflow Prediction — Model Evaluation Report\n\n")
        f.write("Generated by `train_pipeline.py` (MLOps Engineer)\n\n")
        f.write("## Pipeline Config\n")
        f.write(f"- Random state: `{RANDOM_STATE}`\n")
        f.write(f"- Test split: 20%\n")
        f.write(f"- Cross-validation: 5-Fold\n")
        f.write(f"- Target: `net_cash_flow`\n\n")
        f.write("## Results\n\n")
        f.write("| Model | CV MAE | CV RMSE | Test MAE | Test RMSE | MLflow Run |\n")
        f.write("|-------|--------|---------|----------|-----------|------------|\n")
        for r in results:
            best_flag = " (BEST)" if r["name"] == best["name"] else ""
            f.write(
                f"| {r['name']}{best_flag} "
                f"| ${r['cv_mae']:,.2f} "
                f"| ${r['cv_rmse']:,.2f} "
                f"| ${r['test_mae']:,.2f} "
                f"| ${r['test_rmse']:,.2f} "
                f"| `{r['run_id'][:8]}` |\n"
            )
        f.write(f"\n## Best Model\n")
        f.write(f"**{best['name']}** - Test RMSE: `${best['test_rmse']:,.2f}`\n\n")
        f.write("## Artifacts\n")
        f.write(f"- `models/feature_pipeline.pkl` - fitted feature pipeline\n")
        f.write(f"- `models/best_cashflow_model.pkl` - best trained model\n")
        f.write(f"- `mlruns/` - full MLflow experiment tracking\n")
    print(f"  Report saved → {report_path}")
    return report_path


def main():
    print("=" * 60)
    print("  Cashflow Prediction — MLOps Training Pipeline")
    print(f"  Experiment : {EXPERIMENT_NAME}")
    print(f"  Tracking   : {mlflow.get_tracking_uri()}")
    print("=" * 60)

    df = load_data()

    X_train, X_test, y_train, y_test, pipe, pipeline_path = engineer_features(df)

    print("\n── Step 3: Training & logging all models ────────────────────────")
    all_results = []
    for name, model_def in get_models().items():
        result = train_and_log(
            name, model_def,
            X_train, X_test, y_train, y_test,
            pipeline_path
        )
        all_results.append(result)

    best = save_best_model(all_results)

    write_report(all_results, best)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Best model : {best['name']} (RMSE ${best['test_rmse']:,.2f})")
    print("\n  To view MLflow dashboard run:")
    print("    mlflow ui --backend-store-uri mlruns/")
    print("  Then open: http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()