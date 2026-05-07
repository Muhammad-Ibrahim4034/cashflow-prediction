import pandas as pd
import joblib
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_cashflow_model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "feature_pipeline.pkl")


class CashFlowPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.pipeline = joblib.load(PIPELINE_PATH)

        self.macro_defaults = {
            "market_inflation_rate": 103.78904620758095,
            "market_GDP_USD": 842456608626.4307,
            "market_internet_penetration_pct": 0.14409569461354332,
            "market_trade_services_pct": 19.037649369076057,
            "market_gini_index": 41.83461538461539,
            "market_tax_revenue_pct": 16.421546954908727,
            "market_political_stability": -0.007311884554163131,
            "market_govt_effectiveness": -0.01301692932514891,
            "market_population": 220947620.4528302,
        }

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # === FILL MACROS FIRST ===
        for col, val in self.macro_defaults.items():
            if col not in df.columns:
                df[col] = val

        # === Derived features — compute ALL of them, including ones
        #     that feature_engineering.py's transform() expects internally ===

        # These two are needed by feature_engineering.py internally
        df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

        df['amount_to_orig_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-8)

        df['orig_account_drained'] = (df['newbalanceOrig'] <= 1).astype(int)
        df['is_large_transaction'] = (df['amount'] > 100000).astype(int)

        if 'week_of_month' not in df.columns:
            df['week_of_month'] = ((df['day_of_month'] - 1) // 7 + 1).astype(int)

        # Time of day one-hot
        h = df['hour_of_day']
        df['tod_morning'] = ((h >= 5) & (h <= 11)).astype(int)
        df['tod_afternoon'] = ((h >= 12) & (h <= 17)).astype(int)
        df['tod_evening'] = ((h >= 18) & (h <= 22)).astype(int)
        df['tod_night'] = ((h >= 23) | (h <= 4)).astype(int)
        df['tod_late_night'] = ((h >= 0) & (h <= 4)).astype(int)

        # Log & interaction features
        df['log_amount'] = np.log1p(df['amount'])
        df['log_amount_to_balance_ratio'] = np.log1p(df['amount_to_orig_balance_ratio'])
        df['inflation_x_log_amount'] = df['market_inflation_rate'] * df['log_amount']
        df['log_gdp_scale'] = np.log1p(df['market_GDP_USD'])
        df['amount_gdp_ratio'] = df['amount'] / (df['market_GDP_USD'] + 1e-8)

        # Additional features
        df['hour_risk_flag'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)

        df['monthly_income_est'] = df['oldbalanceOrg'] * 0.05
        df['expense_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-8)
        df['savings_trend'] = 0.0
        df['spending_volatility'] = 0.0
        df['balance_utilisation'] = df['amount'] / (df['oldbalanceOrg'] + 1e-8)
        df['dest_balance_growth'] = df['balance_change_dest'] / (df['oldbalanceDest'] + 1e-8)

        return df

    def predict(self, input_dict: dict):
        df = pd.DataFrame([input_dict])
        df = self._compute_derived_features(df)

        # Columns the pipeline's feature_engineering.py needs internally
        # (balance_change_orig / dest are NOT in the final 35 but must exist
        #  so the pipeline's transform() can reference them without a KeyError)
        pipeline_input_cols = [
            # The 35 the pipeline was trained on
            'step', 'transaction_type_encoded', 'hour_of_day', 'day_of_month', 'week_of_month',
            'amount', 'amount_to_orig_balance_ratio', 'orig_account_drained', 'is_large_transaction',
            'market_inflation_rate', 'market_GDP_USD', 'market_internet_penetration_pct',
            'market_trade_services_pct', 'market_gini_index', 'market_tax_revenue_pct',
            'market_political_stability', 'market_govt_effectiveness', 'market_population',
            'monthly_income_est', 'expense_ratio', 'savings_trend', 'spending_volatility',
            'balance_utilisation', 'dest_balance_growth', 'hour_risk_flag', 'log_amount',
            'log_amount_to_balance_ratio', 'inflation_x_log_amount', 'log_gdp_scale',
            'amount_gdp_ratio', 'tod_afternoon', 'tod_evening', 'tod_late_night',
            'tod_morning', 'tod_night',
            # Extra columns needed internally by feature_engineering.py's transform()
            'balance_change_orig',
            'balance_change_dest',
            # Raw balance columns in case feature_engineering.py references them too
            'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        ]

        # Fill any missing column with 0.0 (safety net)
        for col in pipeline_input_cols:
            if col not in df.columns:
                df[col] = 0.0

        df = df[pipeline_input_cols]  # Pass everything the pipeline might need

        processed = self.pipeline.transform(df)
        prediction = self.model.predict(processed)[0]
        return float(prediction)


predictor = CashFlowPredictor()