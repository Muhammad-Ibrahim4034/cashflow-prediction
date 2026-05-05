import os
import pandas as pd
import joblib

# -- Paths --
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "feature_pipeline.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_cashflow_model.pkl")

def run_test():
    print("-- 1. Loading AI Pipeline & Model ---------------")
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        model = joblib.load(MODEL_PATH)
        print("AI System Loaded Successfully!\n")
    except FileNotFoundError:
        print("ERROR: Models not found. Please run src/train.py first.")
        return

    # -- 2. Define Sample Test Scenarios --
    test_cases = [
        {
            "Case": "Regular Daytime Payment",
            "step": 12,
            "amount": 1500.00,
            "oldbalanceOrg": 50000.00,
            "newbalanceOrig": 48500.00,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 1500.00,
            "type": "PAYMENT",
            "day_of_month": 5,
            "week_of_month": 1,
            "hour_of_day": 12,
            "market_inflation_rate": 2.5,
            "market_GDP_USD": 25e12,
            "market_internet_penetration_pct": 85.0,
            "market_trade_services_pct": 12.0,
            "market_gini_index": 35.0,
            "market_tax_revenue_pct": 18.0,
            "market_population": 330e6,
            "market_political_stability": 0.5,
            "market_govt_effectiveness": 0.8,
            "time_of_day_bucket": "afternoon"
        },
        {
            "Case": "Massive Account Drain (Whale)",
            "step": 14,
            "amount": 250000.00,
            "oldbalanceOrg": 250000.00,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 250000.00,
            "type": "TRANSFER",
            "day_of_month": 10,
            "week_of_month": 2,
            "hour_of_day": 14,
            "market_inflation_rate": 2.5,
            "market_GDP_USD": 25e12,
            "market_internet_penetration_pct": 85.0,
            "market_trade_services_pct": 12.0,
            "market_gini_index": 35.0,
            "market_tax_revenue_pct": 18.0,
            "market_population": 330e6,
            "market_political_stability": 0.5,
            "market_govt_effectiveness": 0.8,
            "time_of_day_bucket": "afternoon"
        },
        {
            "Case": "Late Night Suspicious Activity",
            "step": 3,
            "amount": 8000.00,
            "oldbalanceOrg": 10000.00,
            "newbalanceOrig": 2000.00,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 8000.00,
            "type": "CASH_OUT",
            "day_of_month": 15,
            "week_of_month": 3,
            "hour_of_day": 3,
            "market_inflation_rate": 5.0,
            "market_GDP_USD": 20e12,
            "market_internet_penetration_pct": 80.0,
            "market_trade_services_pct": 10.0,
            "market_gini_index": 40.0,
            "market_tax_revenue_pct": 15.0,
            "market_population": 330e6,
            "market_political_stability": -0.5,
            "market_govt_effectiveness": 0.4,
            "time_of_day_bucket": "night"
        }
    ]

    df_raw = pd.DataFrame(test_cases)
    
    # Pre-encode transaction types
    type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
    df_raw["transaction_type_encoded"] = df_raw["type"].map(type_map)
    
    # Add required calculated columns
    df_raw["balance_change_orig"] = df_raw["newbalanceOrig"] - df_raw["oldbalanceOrg"]
    df_raw["balance_change_dest"] = df_raw["newbalanceDest"] - df_raw["oldbalanceDest"]
    df_raw["amount_to_orig_balance_ratio"] = df_raw["amount"] / (df_raw["oldbalanceOrg"] + 1)
    df_raw["is_large_transaction"] = (df_raw["amount"] > 100000).astype(int)
    df_raw["orig_account_drained"] = (df_raw["newbalanceOrig"] == 0).astype(int)

    # Drop non-feature columns
    df_for_pipe = df_raw.drop(columns=["Case", "type"])

    print("-- 2. Running Live Predictions ------------------")
    
    # Run through feature engineering
    X_processed = pipeline.transform(df_for_pipe)
    
    # FORCE ORDER: Ensure processed features match the model's expected order
    if hasattr(model, "feature_names_in_"):
        X_processed = X_processed[model.feature_names_in_]
    
    # Get predictions
    predictions = model.predict(X_processed)
    
    for i, case in enumerate(test_cases):
        print(f"Scenario: {case['Case']}")
        print(f"  Input Amount      : ${case['amount']:,.2f}")
        print(f"  PREDICTED CASHFLOW: ${predictions[i]:,.2f}")
        
        # Flags
        if X_processed.iloc[i]["hour_risk_flag"] > 0:
            print("  [!] ALERT: High-Risk Hour (Late Night).")
        if case["amount"] > 100000:
            print("  [!] ALERT: Large Transaction threshold exceeded.")
        if X_processed.iloc[i]["balance_utilisation"] > 0.9:
            print("  [!] ALERT: Account Drain / High Utilization detected.")
            
        print("-" * 40)

if __name__ == "__main__":
    run_test()
