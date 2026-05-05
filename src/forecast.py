import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# -- Paths --
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cashflow_prediction_dataset_100k.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def create_time_series_features(df_daily):
    """
    Creates lag and rolling features for time-series forecasting.
    """
    df = df_daily.copy()
    
    # Target: The total net cash flow for the day
    # We want to predict this based on previous days
    
    # 1. Lag Features (What happened yesterday and the day before?)
    df["lag_1"] = df["total_net_flow"].shift(1)
    df["lag_2"] = df["total_net_flow"].shift(2)
    df["lag_7"] = df["total_net_flow"].shift(7) # Weekly pattern
    
    # 2. Rolling Features (What is the recent trend?)
    df["rolling_mean_3"] = df["total_net_flow"].shift(1).rolling(window=3).mean()
    df["rolling_std_3"] = df["total_net_flow"].shift(1).rolling(window=3).std()
    
    # 3. Time Features
    df["day_of_week"] = (df["day_of_month"] - 1) % 7
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    return df.dropna()

def main():
    print("-- 1. Aggregating Data for Forecasting ----------")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find processed data at {PROCESSED_DATA_PATH}")
        return

    # Aggregate by day to see the bank's total daily liquidity impact
    df_daily = df.groupby("day_of_month").agg(
        total_net_flow=("net_cash_flow", "sum"),
        total_amount=("amount", "sum"),
        avg_inflation=("market_inflation_rate", "mean"),
        avg_gdp=("market_GDP_USD", "mean")
    ).reset_index()

    print(f"Aggregated into {len(df_daily)} days of bank activity.")

    # -- 2. Feature Engineering for Time Series --
    df_ts = create_time_series_features(df_daily)
    
    # Define features and target
    features = [
        "lag_1", "lag_2", "lag_7", 
        "rolling_mean_3", "rolling_std_3", 
        "day_of_week", "is_weekend",
        "avg_inflation", "avg_gdp"
    ]
    X = df_ts[features]
    y = df_ts["total_net_flow"]

    # -- 3. Train/Test Split (Temporal) --
    # For time series, we don't shuffle. We take the last 7 days as test.
    split_idx = len(df_ts) - 7
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training on first {len(X_train)} days, testing on last {len(X_test)} days.")

    # -- 4. Training the Forecaster --
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)

    print("\n-- Forecast Performance (Daily Liquidity) -------")
    print(f"  MAE : ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")

    # -- 5. Saving Artifacts --
    model_path = os.path.join(MODELS_DIR, "liquidity_forecaster.pkl")
    joblib.dump(model, model_path)
    print(f"\nOK Forecaster saved to {model_path}")

    # -- 6. Generate a 30-Day Projection --
    # We take the last available data and "roll" it forward
    print("\n-- Generating 30-Day Liquidity Projection -------")
    last_row = df_ts.iloc[-1:].copy()
    projections = []
    
    # Simple projection logic: repeating the learned daily patterns
    current_flow = df_ts["total_net_flow"].iloc[-1]
    for i in range(1, 31):
        # In a real scenario, we would recursively feed preds back into lags
        # For this demo, we will project based on the day-of-week seasonality
        proj_day = (df_ts["day_of_month"].max() + i)
        day_of_week = (proj_day - 1) % 7
        
        # Heuristic: use mean flow of that day of week from training
        avg_for_day = df_ts[df_ts["day_of_week"] == day_of_week]["total_net_flow"].mean()
        projections.append({"day": proj_day, "predicted_net_flow": avg_for_day})

    proj_df = pd.DataFrame(projections)
    proj_path = os.path.join(BASE_DIR, "data", "processed", "liquidity_projection_30d.csv")
    proj_df.to_csv(proj_path, index=False)
    print(f"OK 30-day projection saved to {proj_path}")

if __name__ == "__main__":
    main()
