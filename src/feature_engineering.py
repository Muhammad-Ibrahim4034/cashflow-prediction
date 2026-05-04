import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def compute_monthly_income(df: pd.DataFrame) -> pd.Series:
    is_inflow = (
        (df["balance_change_orig"] > 0) |
        (df["transaction_type_encoded"] == 4)         
    )
    inflow_amount = np.where(is_inflow, df["amount"], 0.0)

    if "day_of_month" in df.columns:
        days_in_period = df["day_of_month"].clip(lower=1)
        monthly_income = (inflow_amount / days_in_period) * 30
    else:
        monthly_income = inflow_amount

    return pd.Series(monthly_income, index=df.index, name="monthly_income_est")


def compute_expense_ratio(df: pd.DataFrame,
                          monthly_income: pd.Series) -> pd.Series:

    is_outflow = df["balance_change_orig"] < 0
    spend = np.where(is_outflow, df["amount"], 0.0)

    safe_income = monthly_income.replace(0, np.nan)
    ratio = pd.Series(spend, index=df.index) / safe_income
    ratio = ratio.fillna(0.0).clip(upper=10.0)
    ratio.name = "expense_ratio"
    return ratio


def compute_savings_trend(df: pd.DataFrame) -> pd.Series:
    trend = (df["newbalanceOrig"] - df["oldbalanceOrg"]) / (df["oldbalanceOrg"] + 1)
    trend = trend.clip(lower=-1.0, upper=5.0)
    trend.name = "savings_trend"
    return trend


def compute_spending_volatility(df: pd.DataFrame,
                                group_col: str = "day_of_month",
                                vol_mapping: dict = None,
                                global_std: float = None) -> pd.Series:
    if group_col not in df.columns:
        group_col = "week_of_month" if "week_of_month" in df.columns else None

    if group_col and vol_mapping is not None:
        vol = df[group_col].map(vol_mapping)
    elif group_col:
        vol = df.groupby(group_col)["amount"].transform("std")
    else:
        vol = pd.Series(global_std if global_std is not None else df["amount"].std(), index=df.index)
        
    if global_std is None:
        global_std = df["amount"].std()
        
    vol = vol.fillna(global_std)
    vol.name = "spending_volatility"
    return vol


def compute_balance_utilisation(df: pd.DataFrame) -> pd.Series:
    total_capital = df["oldbalanceOrg"] + df["oldbalanceDest"] + 1
    util = df["amount"] / total_capital
    util = util.clip(upper=1.0)
    util.name = "balance_utilisation"
    return util


def compute_dest_balance_growth(df: pd.DataFrame) -> pd.Series:
    """
    dest_balance_growth = balance_change_dest / (oldbalanceDest + 1)
    How much did the destination account grow relative to its prior balance?
    """
    growth = df["balance_change_dest"] / (df["oldbalanceDest"] + 1)
    growth = growth.clip(lower=-1.0, upper=10.0)
    growth.name = "dest_balance_growth"
    return growth


def compute_transaction_hour_risk(df: pd.DataFrame) -> pd.Series:
    if "hour_of_day" in df.columns:
        flag = ((df["hour_of_day"] <= 5) | (df["hour_of_day"] >= 22)).astype(int)
    else:
        flag = pd.Series(0, index=df.index)
    flag.name = "hour_risk_flag"
    return flag


def compute_log_amount(df: pd.DataFrame) -> pd.Series:
    """log1p transform on amount — handles the extreme right skew."""
    return np.log1p(df["amount"]).rename("log_amount")


def compute_log_balance_ratio(df: pd.DataFrame) -> pd.Series:
    """log1p of amount_to_orig_balance_ratio — tames the 386k max outlier."""
    return np.log1p(df["amount_to_orig_balance_ratio"]).rename("log_amount_to_balance_ratio")


def compute_macro_interaction(df: pd.DataFrame) -> pd.DataFrame:

    out = pd.DataFrame(index=df.index)
    if "market_inflation_rate" in df.columns:
        out["inflation_x_log_amount"] = (
            np.log1p(df["market_inflation_rate"].clip(lower=0)) *
            np.log1p(df["amount"])
        )
    if "market_GDP_USD" in df.columns:
        gdp_trillions = df["market_GDP_USD"] / 1e12
        out["log_gdp_scale"] = np.log1p(gdp_trillions)
        out["amount_gdp_ratio"] = df["amount"] / (df["market_GDP_USD"] + 1)
    return out

_SCALE_COLS = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balance_change_orig", "balance_change_dest",
    "market_inflation_rate", "market_GDP_USD",
    "market_internet_penetration_pct", "market_trade_services_pct",
    "market_gini_index", "market_tax_revenue_pct",
    "market_population",
]

_CAT_COLS = ["time_of_day_bucket"]


class FeaturePipeline(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        scale: bool = True,
        encode_categoricals: bool = True,
        drop_raw_balance_cols: bool = False,
    ):
        self.scale = scale
        self.encode_categoricals = encode_categoricals
        self.drop_raw_balance_cols = drop_raw_balance_cols

        self._scaler = StandardScaler()
        self._scale_cols_present: list = []
        self._onehot_categories: list = []
        self._global_std: float = 1.0
        self._vol_mapping: dict = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        self._global_std = float(df["amount"].std())
        
        # Learn standard deviations for spending volatility mapping from training set
        group_col = "day_of_month" if "day_of_month" in df.columns else ("week_of_month" if "week_of_month" in df.columns else None)
        if group_col:
            self._vol_mapping = df.groupby(group_col)["amount"].std().to_dict()
        else:
            self._vol_mapping = {}

        self._scale_cols_present = [c for c in _SCALE_COLS if c in df.columns]
        if self.scale and self._scale_cols_present:
            self._scaler.fit(df[self._scale_cols_present])

        if self.encode_categoricals and "time_of_day_bucket" in df.columns:
            self._onehot_categories = sorted(df["time_of_day_bucket"].unique().tolist())

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() on training data before .transform().")

        df = X.copy()

        monthly_income = compute_monthly_income(df)
        expense_ratio  = compute_expense_ratio(df, monthly_income)
        savings_trend  = compute_savings_trend(df)
        
        # Apply the learned mappings to avoid data leakage
        spending_vol   = compute_spending_volatility(df, vol_mapping=self._vol_mapping, global_std=self._global_std)

        df["monthly_income_est"]   = monthly_income
        df["expense_ratio"]        = expense_ratio
        df["savings_trend"]        = savings_trend
        df["spending_volatility"]  = spending_vol

        df["balance_utilisation"]       = compute_balance_utilisation(df)
        df["dest_balance_growth"]       = compute_dest_balance_growth(df)
        df["hour_risk_flag"]            = compute_transaction_hour_risk(df)
        df["log_amount"]                = compute_log_amount(df)
        df["log_amount_to_balance_ratio"] = compute_log_balance_ratio(df)

        macro_df = compute_macro_interaction(df)
        for col in macro_df.columns:
            df[col] = macro_df[col]

        if self.encode_categoricals and "time_of_day_bucket" in df.columns:
            for cat in self._onehot_categories:
                df[f"tod_{cat}"] = (df["time_of_day_bucket"] == cat).astype(int)
            df.drop(columns=["time_of_day_bucket"], inplace=True)

        if self.scale and self._scale_cols_present:
            present = [c for c in self._scale_cols_present if c in df.columns]
            df[present] = self._scaler.transform(df[present])

        if self.drop_raw_balance_cols:
            raw_bal = [
                "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest",
            ]
            df.drop(columns=[c for c in raw_bal if c in df.columns], inplace=True)

        return df

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def feature_names_out(self) -> list:
        """Returns the list of output column names after last transform call."""
        return self._last_columns if hasattr(self, "_last_columns") else []


def feature_pipeline(
    df_train: pd.DataFrame,
    df_infer: pd.DataFrame = None,
    scale: bool = True,
    encode_categoricals: bool = True,
    drop_raw_balance_cols: bool = False,
    target_cols: list = None,
) -> dict:
   
    if target_cols is None:
        target_cols = ["risk_score", "net_cash_flow"]
    present_targets = [c for c in target_cols if c in df_train.columns]
    y_train = df_train[present_targets].copy() if present_targets else None

    drop_from_features = present_targets + ["simulated_year"]
    X_train_raw = df_train.drop(
        columns=[c for c in drop_from_features if c in df_train.columns]
    )

    pipe = FeaturePipeline(
        scale=scale,
        encode_categoricals=encode_categoricals,
        drop_raw_balance_cols=drop_raw_balance_cols,
    )
    X_train = pipe.fit_transform(X_train_raw)
    feature_names = list(X_train.columns)

    X_infer, y_infer = None, None
    if df_infer is not None:
        present_targets_infer = [c for c in target_cols if c in df_infer.columns]
        y_infer = df_infer[present_targets_infer].copy() if present_targets_infer else None
        X_infer_raw = df_infer.drop(
            columns=[c for c in drop_from_features if c in df_infer.columns]
        )
        X_infer = pipe.transform(X_infer_raw)

    return {
        "X_train":       X_train,
        "y_train":       y_train,
        "X_infer":       X_infer,
        "y_infer":       y_infer,
        "pipeline":      pipe,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    INPUT_PATH = "cashflow_prediction_dataset_100k.csv"
    print(f"Loading dataset from {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    strat = df["risk_score"] if "risk_score" in df.columns else None
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=strat)
    print(f"  Train : {len(df_train):,} rows")
    print(f"  Test  : {len(df_test):,} rows\n")

    result = feature_pipeline(
        df_train=df_train,
        df_infer=df_test,
        scale=True,
        encode_categoricals=True,
        drop_raw_balance_cols=False,
    )

    X_train       = result["X_train"]
    y_train       = result["y_train"]
    X_test        = result["X_infer"]
    y_test        = result["y_infer"]
    pipe          = result["pipeline"]
    feature_names = result["feature_names"]

    print("── Feature Engineering Validation ──────────────────────────────────")
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_test  shape   : {X_test.shape}")

    if y_train is not None:
        print(f"  y_train targets : {list(y_train.columns)}")
        if "risk_score" in y_train.columns:
            fraud_pct = y_train["risk_score"].mean() * 100
            print(f"  Fraud rate      : {fraud_pct:.2f}%")

    print(f"\n  Total features  : {len(feature_names)}")

    core_features = [
        "monthly_income_est",
        "expense_ratio",
        "savings_trend",
        "spending_volatility",
    ]
    print("\n── Core Feature Stats (on training set) ────────────────────────────")
    for feat in core_features:
        if feat in X_train.columns:
            s = X_train[feat]
            print(f"  {feat:<30s}  mean={s.mean():>12.4f}  std={s.std():>12.4f}"
                  f"  min={s.min():>12.4f}  max={s.max():>12.4f}")
        else:
            print(f"  ❌ {feat} NOT FOUND in output")
    train_nulls = X_train.isnull().sum().sum()
    test_nulls  = X_test.isnull().sum().sum()
    print(f"\n  Nulls in X_train : {train_nulls}")
    print(f"  Nulls in X_test  : {test_nulls}")
    assert train_nulls == 0, "❌ Nulls found in training features!"
    assert test_nulls  == 0, "❌ Nulls found in test features!"

    print("\n── All Features ────────────────────────────────────────────────────")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:02d}. {name}")

    print("\n✅ Feature engineering pipeline validated successfully.")
    print(f"   Save the pipeline with: import joblib; joblib.dump(pipe, 'feature_pipeline.pkl')")
