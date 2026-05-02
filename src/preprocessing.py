import pandas as pd
import numpy as np

print("Loading PaySim dataset (full)...")
df_paysim = pd.read_csv("PS_20174392719_1491204439457_log.csv")
print(f"  PaySim loaded: {len(df_paysim):,} rows")

print("Loading World Bank dataset...")
df_wb = pd.read_csv("world_bank_development_indicators.csv")
df_wb["year"] = pd.to_datetime(df_wb["date"]).dt.year
print(f"  World Bank loaded: {len(df_wb):,} rows\n")

df_paysim["hour_of_day"] = ((df_paysim["step"] - 1) % 24).astype(int)
df_paysim["day_of_month"] = ((df_paysim["step"] - 1) // 24 + 1).astype(int)
df_paysim["week_of_month"] = ((df_paysim["day_of_month"] - 1) // 7 + 1).astype(int)

N_TOTAL = 100_000
N_PER_DAY = N_TOTAL // 31

print(f"Stratified sampling: {N_PER_DAY} rows/day across 31 days...")
sampled_parts = []
for day, group in df_paysim.groupby("day_of_month"):
    n = min(N_PER_DAY, len(group))
    sampled_parts.append(group.sample(n=n, random_state=42))

df_sampled = pd.concat(sampled_parts, ignore_index=True)

shortfall = N_TOTAL - len(df_sampled)
if shortfall > 0:
    already_used = df_sampled.index
    pool = df_paysim[~df_paysim.index.isin(already_used)]
    top_up = pool.sample(n=shortfall, random_state=99)
    df_sampled = pd.concat([df_sampled, top_up], ignore_index=True)

df_sampled["hour_of_day"] = ((df_sampled["step"] - 1) % 24).astype(int)
df_sampled["day_of_month"] = ((df_sampled["step"] - 1) // 24 + 1).astype(int)
df_sampled["week_of_month"] = ((df_sampled["day_of_month"] - 1) // 7 + 1).astype(int)

print(f"  Sampled rows   : {len(df_sampled):,}")
print(f"  Unique days    : {df_sampled['day_of_month'].nunique()}")
print(f"  Step range     : {df_sampled['step'].min()} – {df_sampled['step'].max()}\n")

day_to_year = {day: 1993 + (day - 1) for day in range(1, 32)}
df_sampled["simulated_year"] = df_sampled["day_of_month"].map(day_to_year)

MACRO_COLS = [
    "inflation_annual%",
    "GDP_current_US",
    "individuals_using_internet%",
    "trade_in_services%",
    "gini_index",
    "tax_revenue%",
    "political_stability_estimate",
    "goverment_effectiveness_estimate",
    "population",
]

global_avg = df_wb.groupby("year")[MACRO_COLS].mean().reset_index()

global_avg = (
    global_avg.set_index("year")
    .reindex(range(1960, 2024))
)
for col in MACRO_COLS:
    global_avg[col] = global_avg[col].interpolate(
        method="linear", limit_direction="both"
    )
global_avg = global_avg.reset_index().rename(columns={"index": "year"})

macro_lookup = (
    global_avg[global_avg["year"].between(1993, 2023)]
    .copy()
    .rename(columns={
        "year": "simulated_year",
        "inflation_annual%": "market_inflation_rate",
        "GDP_current_US": "market_GDP_USD",
        "individuals_using_internet%": "market_internet_penetration_pct",
        "trade_in_services%": "market_trade_services_pct",
        "gini_index": "market_gini_index",
        "tax_revenue%": "market_tax_revenue_pct",
        "political_stability_estimate": "market_political_stability",
        "goverment_effectiveness_estimate": "market_govt_effectiveness",
        "population": "market_population",
    })
)

df = df_sampled.merge(macro_lookup, on="simulated_year", how="left")
print(f"Shape after merge: {df.shape}")

df["time_of_day_bucket"] = (
    pd.cut(
        df["hour_of_day"],
        bins=[-1, 5, 11, 17, 21, 23],
        labels=["night", "morning", "afternoon", "evening", "late_night"],
    )
    .astype(str)
    .fillna("unknown")
)

df["balance_change_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
df["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
df["net_cash_flow_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]

df["amount_to_orig_balance_ratio"] = np.where(
    df["oldbalanceOrg"] > 0,
    df["amount"] / df["oldbalanceOrg"],
    0,
)

df["orig_account_drained"] = (
    (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
).astype(int)

df["is_large_transaction"] = (
    df["amount"] > df["amount"].quantile(0.90)
).astype(int)

df["transaction_type_encoded"] = df["type"].map(
    {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
)

df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud",
                 "isFraud", "net_cash_flow_orig", "type", "simulated_year"], inplace=True)

FINAL_COLS = [
    "step", "transaction_type_encoded",
    "hour_of_day", "day_of_month", "week_of_month", "time_of_day_bucket",
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "balance_change_orig", "balance_change_dest",
    "amount_to_orig_balance_ratio",
    "orig_account_drained", "is_large_transaction",
    "market_inflation_rate", "market_GDP_USD", "market_internet_penetration_pct",
    "market_trade_services_pct", "market_gini_index", "market_tax_revenue_pct",
    "market_political_stability", "market_govt_effectiveness", "market_population",
]

df_final = df[FINAL_COLS]

print("\n── Validation ──────────────────────────────────────")
assert len(df_final) == 100_000, f"Row count mismatch: {len(df_final)}"

nulls = df_final.isnull().sum()
assert nulls.sum() == 0, f"Nulls found:\n{nulls[nulls > 0]}"
print("✅ Rows       : 100,000")
print("✅ Columns    :", len(df_final.columns))
print("✅ Nulls      : 0")

market_cols = [c for c in df_final.columns if c.startswith("market_")]
print("\n── Market column variation (unique values per column) ──")
for c in market_cols:
    print(f"   {c:40s} unique={df_final[c].nunique():3d}  std={df_final[c].std():.4f}")

print("\n── Transaction type distribution ──")
print(df_final["transaction_type_encoded"].value_counts().to_string())

OUTPUT_PATH = "cashflow_prediction_dataset_100k.csv"
df_final.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved → {OUTPUT_PATH}")
print(f"   Shape  : {df_final.shape}")
