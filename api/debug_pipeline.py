import joblib
import pandas as pd
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_cashflow_model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "feature_pipeline.pkl")
pipeline = joblib.load(MODEL_PATH)
model = joblib.load(PIPELINE_PATH)

print("=== Pipeline Expected Features ===")
if hasattr(pipeline, 'feature_names_in_'):
    print(list(pipeline.feature_names_in_))
    print("\nTotal features:", len(pipeline.feature_names_in_))
else:
    print("Pipeline does not have feature_names_in_")

print("\n=== Model n_features_in_ ===", model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown")