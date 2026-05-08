import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import sys
from inference import predictor

st.set_page_config(page_title="CashFlow AI Predictor", layout="wide")
st.title("💰 CashFlow Prediction Dashboard")
st.markdown("**MLOps Project** — Predict net cash flow with ML + Macro Indicators")

# ── Path setup ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

MODEL_PATH          = os.path.join(BASE_DIR, "models", "best_cashflow_model.pkl")
PIPELINE_PATH       = os.path.join(BASE_DIR, "models", "feature_pipeline.pkl")
FORECASTER_PATH     = os.path.join(BASE_DIR, "models", "liquidity_forecaster.pkl")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cashflow_prediction_dataset_100k.csv")
PROJECTION_PATH     = os.path.join(BASE_DIR, "data", "processed", "liquidity_projection_30d.csv")

# ── Load main model & pipeline ─────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline

model, pipeline = load_artifacts()

# ── Time-series helpers ────────────────────────────────────────────────────────
def create_time_series_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    df["lag_1"]          = df["total_net_flow"].shift(1)
    df["lag_2"]          = df["total_net_flow"].shift(2)
    df["lag_7"]          = df["total_net_flow"].shift(7)
    df["rolling_mean_3"] = df["total_net_flow"].shift(1).rolling(window=3).mean()
    df["rolling_std_3"]  = df["total_net_flow"].shift(1).rolling(window=3).std()
    df["day_of_week"]    = (df["day_of_month"] - 1) % 7
    df["is_weekend"]     = df["day_of_week"].isin([5, 6]).astype(int)
    return df.dropna()

FORECAST_FEATURES = [
    "lag_1", "lag_2", "lag_7",
    "rolling_mean_3", "rolling_std_3",
    "day_of_week", "is_weekend",
    "avg_inflation", "avg_gdp",
]

# ── Session-state loaders (avoid stale @st.cache_data issues) ──────────────────
def load_forecaster():
    if "forecaster" not in st.session_state:
        st.session_state["forecaster"] = (
            joblib.load(FORECASTER_PATH) if os.path.exists(FORECASTER_PATH) else None
        )
    return st.session_state["forecaster"]

def load_processed_data():
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = (
            pd.read_csv(PROCESSED_DATA_PATH) if os.path.exists(PROCESSED_DATA_PATH) else None
        )
    return st.session_state["raw_df"]

def build_daily_ts(df: pd.DataFrame) -> tuple:
    df_daily = df.groupby("day_of_month").agg(
        total_net_flow=("net_cash_flow", "sum"),
        total_amount=("amount", "sum"),
        avg_inflation=("market_inflation_rate", "mean"),
        avg_gdp=("market_GDP_USD", "mean"),
    ).reset_index()
    df_ts = create_time_series_features(df_daily)
    return df_daily, df_ts

# ── Projection engine ─────────────────────────────────────────────────────────
def get_projection(
    df_ts: pd.DataFrame,
    forecaster=None,
    horizon: int = 30,
    inflation_delta: float = 0.0,
    gdp_shock: float = 1.0,
    volatility_pct: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a forward projection.

    Parameters
    ----------
    horizon         : number of days to project (1–60)
    inflation_delta : additive shift on avg_inflation  (e.g. +5 = +5 points)
    gdp_shock       : multiplicative factor on avg_gdp  (e.g. 0.9 = −10%)
    volatility_pct  : std of random daily noise as % of rolling mean (0–30)
    """
    rng = np.random.default_rng(seed)

    base_inflation = float(df_ts["avg_inflation"].iloc[-1]) + inflation_delta
    base_gdp       = float(df_ts["avg_gdp"].iloc[-1]) * gdp_shock

    if forecaster is not None:
        history = df_ts.copy()
        projections = []
        for _ in range(horizon):
            proj_day    = int(history["day_of_month"].max()) + 1
            day_of_week = (proj_day - 1) % 7
            is_weekend  = int(day_of_week in [5, 6])

            flows   = history["total_net_flow"].values
            lag_1   = float(flows[-1])
            lag_2   = float(flows[-2]) if len(flows) >= 2 else lag_1
            lag_7   = float(flows[-7]) if len(flows) >= 7 else lag_1
            rm3     = float(np.mean(flows[-3:])) if len(flows) >= 3 else lag_1
            rs3     = float(np.std(flows[-3:]))  if len(flows) >= 3 else 0.0

            row = pd.DataFrame([{
                "lag_1": lag_1, "lag_2": lag_2, "lag_7": lag_7,
                "rolling_mean_3": rm3, "rolling_std_3": rs3,
                "day_of_week": day_of_week, "is_weekend": is_weekend,
                "avg_inflation": base_inflation, "avg_gdp": base_gdp,
            }])
            pred = float(forecaster.predict(row[FORECAST_FEATURES])[0])

            # Apply user-specified volatility noise
            if volatility_pct > 0:
                noise = rng.normal(0, abs(pred) * volatility_pct / 100)
                pred += noise

            projections.append({"day": proj_day, "predicted_net_flow": pred})
            new_row = {
                "day_of_month": proj_day, "total_net_flow": pred,
                "avg_inflation": base_inflation, "avg_gdp": base_gdp,
                "total_amount": 0, "lag_1": lag_1, "lag_2": lag_2,
                "lag_7": lag_7, "rolling_mean_3": rm3, "rolling_std_3": rs3,
                "day_of_week": day_of_week, "is_weekend": is_weekend,
            }
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        return pd.DataFrame(projections)

    else:
        # Heuristic fallback: day-of-week seasonality
        projections = []
        base_day = int(df_ts["day_of_month"].max())
        for i in range(1, horizon + 1):
            proj_day    = base_day + i
            day_of_week = (proj_day - 1) % 7
            avg_for_day = df_ts[df_ts["day_of_week"] == day_of_week]["total_net_flow"].mean()
            noise = 0.0
            if volatility_pct > 0:
                noise = rng.normal(0, abs(avg_for_day) * volatility_pct / 100)
            projections.append({"day": proj_day, "predicted_net_flow": avg_for_day + noise})
        return pd.DataFrame(projections)

# ── Sidebar (main prediction controls) ────────────────────────────────────────
with st.sidebar:
    st.header("Transaction Details")
    amount           = st.number_input("Amount",             value=50000.0)
    oldbalanceOrg    = st.number_input("Old Balance Origin", value=100000.0)
    newbalanceOrig   = st.number_input("New Balance Origin", value=50000.0)
    oldbalanceDest   = st.number_input("Old Balance Dest",   value=0.0)
    newbalanceDest   = st.number_input("New Balance Dest",   value=50000.0)
    transaction_type = st.selectbox("Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
    hour_of_day      = st.slider("Hour of Day",  0, 23, 14)
    day_of_month     = st.slider("Day of Month", 1, 31, 15)
    st.subheader("Macro Economic Inputs (optional)")
    inflation = st.slider("Inflation Rate", 0.0, 200.0, 103.79)

# ── Single-transaction prediction ─────────────────────────────────────────────
if st.button("Predict Net Cash Flow", type="primary"):
    input_data = {
        "step":                     10,
        "amount":                   float(amount),
        "oldbalanceOrg":            float(oldbalanceOrg),
        "newbalanceOrig":           float(newbalanceOrig),
        "oldbalanceDest":           float(oldbalanceDest),
        "newbalanceDest":           float(newbalanceDest),
        "transaction_type_encoded": int(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"].index(transaction_type)),
        "hour_of_day":              int(hour_of_day),
        "day_of_month":             int(day_of_month),
        "market_inflation_rate":    float(inflation),
    }
    prediction = predictor.predict(input_data)
    st.success(f"**Predicted Net Cash Flow: ${prediction:,.2f}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Net Cash Flow", f"${prediction:,.2f}")
    col2.metric("Transaction Size",        f"${amount:,.2f}")
    risk = "🔴 High" if amount > 100000 else ("🟡 Medium" if hour_of_day < 6 else "🟢 Low")
    col3.metric("Risk Level", risk)

    scenario_labels = ["Current", "High Inflation (+2%)", "Large Transfer", "Off-Hours", "Low Balance"]
    scenario_values = [prediction, prediction*0.95, prediction*0.70, prediction*0.88, prediction*1.05]
    scenario_df = pd.DataFrame({"Scenario": scenario_labels, "Predicted Cashflow": scenario_values})

    st.subheader("📊 Prediction vs Amount")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=["Predicted Net Cash Flow"], y=[prediction],
        name="Prediction", marker_color="#2ecc71" if prediction >= 0 else "#e74c3c",
        text=[f"${prediction:,.0f}"], textposition="outside"))
    fig_bar.add_trace(go.Bar(x=["Input Amount"], y=[amount],
        name="Input Amount", marker_color="#3498db",
        text=[f"${amount:,.0f}"], textposition="outside"))
    fig_bar.update_layout(title="Predicted Cash Flow vs Transaction Amount",
        yaxis_title="Value ($)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13), bargap=0.35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_bar.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#888")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📊 Scenario Analysis")
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Bar(x=scenario_df["Scenario"], y=scenario_df["Predicted Cashflow"],
        marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in scenario_values],
        text=[f"${v:,.0f}" for v in scenario_values], textposition="outside"))
    fig_sc.update_layout(title="Cash Flow Across Scenarios",
        xaxis_title="Scenario", yaxis_title="Predicted Cashflow ($)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13))
    fig_sc.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#888")
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Batch Prediction", "📈 Liquidity Forecast", "Model Insights"])

# ─── Tab 1: Batch Prediction ──────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader("Upload CSV for batch prediction", type="csv")
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        processed = pipeline.transform(df_batch)
        df_batch["predicted_net_cashflow"] = model.predict(processed)
        st.dataframe(df_batch)
        fig_batch = go.Figure()
        fig_batch.add_trace(go.Scatter(x=df_batch.index, y=df_batch["predicted_net_cashflow"],
            mode="lines+markers", line=dict(color="#3498db", width=2), marker=dict(size=5),
            name="Predicted Cash Flow"))
        fig_batch.add_hline(y=0, line_dash="dash", line_color="#e74c3c", annotation_text="Break-even")
        fig_batch.update_layout(title="Predicted Net Cash Flow — All Transactions",
            xaxis_title="Transaction Index", yaxis_title="Net Cash Flow ($)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_batch, use_container_width=True)
        st.download_button("Download Predictions", df_batch.to_csv(index=False), "predictions.csv")

# ─── Tab 2: Liquidity Forecast ────────────────────────────────────────────────
with tab2:
    st.header("📈 Liquidity Forecasting")
    st.markdown(
        "Adjust the controls below to see how macro conditions and projection assumptions "
        "change the 30-day cash flow forecast in real time."
    )

    raw_df     = load_processed_data()
    forecaster = load_forecaster()

    if raw_df is None:
        st.warning(
            "⚠️ Processed dataset not found at "
            "`data/processed/cashflow_prediction_dataset_100k.csv`. "
            "Run your data pipeline first."
        )
    else:
        df_daily, df_ts = build_daily_ts(raw_df)

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # INTERACTIVE SCENARIO CONTROLS
        # ══════════════════════════════════════════════════════════════════════
        st.subheader("🎛️ Scenario Controls")
        st.caption("Every slider and toggle updates the forecast chart instantly.")

        ctrl_col1, ctrl_col2 = st.columns(2)

        with ctrl_col1:
            horizon = st.slider(
                "Projection horizon (days)", min_value=7, max_value=60, value=30, step=1,
                help="How many days ahead to project."
            )
            inflation_delta = st.slider(
                "Inflation shift (± points)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5,
                help="Add or subtract from the current inflation rate. Positive = worse macro."
            )

        with ctrl_col2:
            gdp_shock_pct = st.slider(
                "GDP shock (%)", min_value=-30, max_value=30, value=0, step=1,
                help="Simulate a GDP contraction or expansion. -10 = GDP drops 10%."
            )
            volatility_pct = st.slider(
                "Daily volatility (%)", min_value=0, max_value=30, value=0, step=1,
                help="Add random daily noise as % of each predicted value. 0 = deterministic."
            )

        show_confidence = st.toggle("Show ±15% confidence band", value=True)
        show_rolling    = st.toggle("Show 7-day rolling average on projection", value=False)

        # Scenario presets
        st.caption("Or load a preset:")
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
        if preset_col1.button("📉 Recession", use_container_width=True):
            st.session_state["preset"] = {"inflation_delta": 10.0, "gdp_shock_pct": -15, "volatility_pct": 15}
        if preset_col2.button("📈 Boom",      use_container_width=True):
            st.session_state["preset"] = {"inflation_delta": -5.0, "gdp_shock_pct": 12, "volatility_pct": 5}
        if preset_col3.button("😐 Baseline",  use_container_width=True):
            st.session_state["preset"] = {"inflation_delta": 0.0,  "gdp_shock_pct": 0,  "volatility_pct": 0}
        if preset_col4.button("🌪️ High Volatility", use_container_width=True):
            st.session_state["preset"] = {"inflation_delta": 5.0,  "gdp_shock_pct": -5, "volatility_pct": 25}

        # Apply preset values if one was just clicked
        # (Presets override sliders — shown in a callout so user knows)
        active_preset = st.session_state.get("preset", None)
        if active_preset:
            inflation_delta  = active_preset["inflation_delta"]
            gdp_shock_pct    = active_preset["gdp_shock_pct"]
            volatility_pct   = active_preset["volatility_pct"]
            st.info(
                f"Preset active — inflation shift: **{inflation_delta:+.1f}**, "
                f"GDP shock: **{gdp_shock_pct:+d}%**, volatility: **{volatility_pct}%**. "
                "Move any slider above to leave the preset."
            )

        gdp_shock = 1.0 + gdp_shock_pct / 100.0

        # ── Generate projection with current controls ──────────────────────
        proj_df = get_projection(
            df_ts,
            forecaster=forecaster,
            horizon=horizon,
            inflation_delta=inflation_delta,
            gdp_shock=gdp_shock,
            volatility_pct=volatility_pct,
        )

        # Save to disk for external use (non-critical)
        try:
            proj_df.to_csv(PROJECTION_PATH, index=False)
        except Exception:
            pass

        # ── Method label ──────────────────────────────────────────────────
        method = "Random Forest (ML — recursive)" if forecaster is not None else "Day-of-week seasonality (heuristic)"
        st.caption(f"Projection method: **{method}**")

        # ── DYNAMIC KPI strip (updates with every slider change) ──────────
        proj_total     = proj_df["predicted_net_flow"].sum()
        proj_mean      = proj_df["predicted_net_flow"].mean()
        proj_peak      = proj_df["predicted_net_flow"].max()
        proj_worst     = proj_df["predicted_net_flow"].min()
        proj_pos_days  = (proj_df["predicted_net_flow"] > 0).sum()
        hist_mean      = df_daily["total_net_flow"].mean()
        mean_delta_pct = ((proj_mean - hist_mean) / abs(hist_mean) * 100) if hist_mean != 0 else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(
            "Projected total flow",
            f"${proj_total:,.0f}",
            delta="Surplus" if proj_total >= 0 else "Deficit",
            delta_color="normal" if proj_total >= 0 else "inverse",
        )
        k2.metric(
            "Projected daily avg",
            f"${proj_mean:,.0f}",
            delta=f"{mean_delta_pct:+.1f}% vs history",
            delta_color="normal" if mean_delta_pct >= 0 else "inverse",
        )
        k3.metric(
            "Best projected day",
            f"${proj_peak:,.0f}",
            delta="Peak",
        )
        k4.metric(
            "Worst projected day",
            f"${proj_worst:,.0f}",
            delta_color="inverse",
            delta=f"{'Negative' if proj_worst < 0 else 'Positive'}",
        )
        k5.metric(
            "Positive days",
            f"{proj_pos_days} / {horizon}",
            delta=f"{proj_pos_days/horizon*100:.0f}% positive",
            delta_color="normal" if proj_pos_days >= horizon / 2 else "inverse",
        )

        st.divider()

        # ── Chart 1: Historical bar ────────────────────────────────────────
        st.subheader("📊 Historical daily net cash flow")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=df_daily["day_of_month"], y=df_daily["total_net_flow"],
            name="Daily net flow",
            marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in df_daily["total_net_flow"]],
        ))
        fig_hist.add_hline(
            y=df_daily["total_net_flow"].mean(), line_dash="dot", line_color="#3498db",
            annotation_text=f"Mean: ${df_daily['total_net_flow'].mean():,.0f}",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            xaxis_title="Day of month", yaxis_title="Net cash flow ($)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13),
        )
        fig_hist.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#888")
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Chart 2: Combined historical + projection ──────────────────────
        st.subheader(f"🔮 {horizon}-day liquidity projection")

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=df_daily["day_of_month"], y=df_daily["total_net_flow"],
            mode="lines+markers", name="Historical net flow",
            line=dict(color="#3498db", width=2), marker=dict(size=5),
        ))
        fig_fc.add_trace(go.Scatter(
            x=proj_df["day"], y=proj_df["predicted_net_flow"],
            mode="lines+markers", name=f"Projected ({horizon}d)",
            line=dict(color="#f39c12", width=2, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
        ))

        if show_confidence:
            upper = proj_df["predicted_net_flow"] * 1.15
            lower = proj_df["predicted_net_flow"] * 0.85
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([proj_df["day"], proj_df["day"][::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill="toself", fillcolor="rgba(243,156,18,0.10)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip", showlegend=True, name="Confidence band (±15%)",
            ))

        if show_rolling:
            proj_df["rolling_7"] = proj_df["predicted_net_flow"].rolling(window=7, min_periods=1).mean()
            fig_fc.add_trace(go.Scatter(
                x=proj_df["day"], y=proj_df["rolling_7"],
                mode="lines", name="7-day rolling avg",
                line=dict(color="#9b59b6", width=2),
            ))

        fig_fc.add_hline(y=0, line_dash="dash", line_color="#e74c3c",
                         annotation_text="Break-even", annotation_position="bottom right")
        fig_fc.add_vline(
            x=int(df_daily["day_of_month"].max()), line_dash="dot", line_color="#888",
            annotation_text="Forecast →", annotation_position="top right",
        )
        fig_fc.update_layout(
            xaxis_title="Day", yaxis_title="Net cash flow ($)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13), hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_fc.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#aaa")
        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Chart 3: Projection bar (colour-coded) ─────────────────────────
        st.subheader("📊 Projected daily breakdown")
        fig_bar2 = go.Figure()
        fig_bar2.add_trace(go.Bar(
            x=proj_df["day"], y=proj_df["predicted_net_flow"],
            marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in proj_df["predicted_net_flow"]],
            text=[f"${v:,.0f}" for v in proj_df["predicted_net_flow"]],
            textposition="outside",
        ))
        fig_bar2.add_hline(
            y=proj_df["predicted_net_flow"].mean(), line_dash="dot", line_color="#9b59b6",
            annotation_text=f"Projected mean: ${proj_df['predicted_net_flow'].mean():,.0f}",
            annotation_position="top left",
        )
        fig_bar2.update_layout(
            xaxis_title="Projected day", yaxis_title="Predicted net flow ($)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13), showlegend=False,
        )
        fig_bar2.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#888")
        st.plotly_chart(fig_bar2, use_container_width=True)

        st.download_button(
            "📥 Download projection CSV",
            proj_df[["day", "predicted_net_flow"]].to_csv(index=False),
            "liquidity_projection.csv",
        )

        # ── Retrain ────────────────────────────────────────────────────────
        st.divider()
        st.subheader("⚙️ Forecaster controls")
        st.caption("Normally trained by running `src/forecast.py`. Click below to retrain from the app.")
        if st.button("🔄 Re-train forecaster model now"):
            with st.spinner("Training Random Forest forecaster..."):
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                X = df_ts[FORECAST_FEATURES]
                y = df_ts["total_net_flow"]
                split_idx = max(1, len(df_ts) - 7)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                new_forecaster = RandomForestRegressor(n_estimators=100, random_state=42)
                new_forecaster.fit(X_train, y_train)
                preds_test = new_forecaster.predict(X_test)
                mae  = mean_absolute_error(y_test, preds_test)
                rmse = float(np.sqrt(mean_squared_error(y_test, preds_test)))
                os.makedirs(os.path.dirname(FORECASTER_PATH), exist_ok=True)
                joblib.dump(new_forecaster, FORECASTER_PATH)
                st.session_state["forecaster"] = new_forecaster
                new_proj = get_projection(df_ts, forecaster=new_forecaster)
                new_proj.to_csv(PROJECTION_PATH, index=False)
            st.success(f"✅ Retrained! MAE: **${mae:,.2f}** | RMSE: **${rmse:,.2f}**")
            st.rerun()

# ─── Tab 3: Model Insights ─────────────────────────────────────────────────────
with tab3:
    st.info("Model Insights coming soon — feature importance, SHAP values, etc.")

st.caption("Built with Streamlit • Random Forest Model • MLOps Pipeline")