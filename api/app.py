import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
from inference import predictor

st.set_page_config(page_title="CashFlow AI Predictor", layout="wide")
st.title("💰 CashFlow Prediction Dashboard")
st.markdown("**MLOps Project** — Predict net cash flow with ML + Macro Indicators")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_cashflow_model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "feature_pipeline.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline

model, pipeline = load_artifacts()

with st.sidebar:
    st.header("Transaction Details")
    amount = st.number_input("Amount", value=50000.0)
    oldbalanceOrg = st.number_input("Old Balance Origin", value=100000.0)
    newbalanceOrig = st.number_input("New Balance Origin", value=50000.0)
    oldbalanceDest = st.number_input("Old Balance Dest", value=0.0)
    newbalanceDest = st.number_input("New Balance Dest", value=50000.0)
    transaction_type = st.selectbox("Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
    hour_of_day = st.slider("Hour of Day", 0, 23, 14)
    day_of_month = st.slider("Day of Month", 1, 31, 15)
    st.subheader("Macro Economic Inputs (optional)")
    inflation = st.slider("Inflation Rate", 0.0, 200.0, 103.79)

if st.button("Predict Net Cash Flow", type="primary"):
    input_data = {
        "step": 10,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest),
        "transaction_type_encoded": int(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"].index(transaction_type)),
        "hour_of_day": int(hour_of_day),
        "day_of_month": int(day_of_month),
        "market_inflation_rate": float(inflation),
    }

    prediction = predictor.predict(input_data)

    st.success(f"**Predicted Net Cash Flow: ${prediction:,.2f}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Net Cash Flow", f"${prediction:,.2f}")
    with col2:
        st.metric("Transaction Size", f"${amount:,.2f}")
    with col3:
        risk = "🔴 High" if amount > 100000 else ("🟡 Medium" if hour_of_day < 6 else "🟢 Low")
        st.metric("Risk Level", risk)

    # ── Scenario values ──────────────────────────────────────────────────────
    scenario_labels = ["Current", "High Inflation (+2%)", "Large Transfer", "Off-Hours", "Low Balance"]
    scenario_values = [
        prediction,
        prediction * 0.95,
        prediction * 0.70,
        prediction * 0.88,
        prediction * 1.05,
    ]

    scenario_df = pd.DataFrame({
        "Scenario": scenario_labels,
        "Predicted Cashflow": scenario_values,
    })

    # ── Chart 1: Bar — Prediction vs Amount ──────────────────────────────────
    st.subheader("📊 Prediction Visualization")

    bar_colors = ["#2ecc71" if prediction >= 0 else "#e74c3c", "#3498db"]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=["Predicted Net Cash Flow"],
        y=[prediction],
        name="Prediction",
        marker_color=bar_colors[0],
        text=[f"${prediction:,.0f}"],
        textposition="outside",
    ))
    fig_bar.add_trace(go.Bar(
        x=["Input Amount"],
        y=[amount],
        name="Input Amount",
        marker_color=bar_colors[1],
        text=[f"${amount:,.0f}"],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Predicted Cash Flow vs Transaction Amount",
        yaxis_title="Value ($)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.35,
    )
    fig_bar.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#888")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Chart 2: Grouped Bar — Scenario Analysis ──────────────────────────────
    st.subheader("📊 Scenario Analysis — Bar Chart")

    bar_scenario_colors = [
        "#2ecc71" if v >= 0 else "#e74c3c" for v in scenario_values
    ]
    fig_scenario_bar = go.Figure()
    fig_scenario_bar.add_trace(go.Bar(
        x=scenario_df["Scenario"],
        y=scenario_df["Predicted Cashflow"],
        marker_color=bar_scenario_colors,
        text=[f"${v:,.0f}" for v in scenario_values],
        textposition="outside",
        name="Cashflow",
    ))
    fig_scenario_bar.update_layout(
        title="Cash Flow Across Different Scenarios",
        xaxis_title="Scenario",
        yaxis_title="Predicted Cashflow ($)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        showlegend=False,
    )
    fig_scenario_bar.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#888")
    st.plotly_chart(fig_scenario_bar, use_container_width=True)

    # ── Chart 3: Line Graph — Scenario Trend ─────────────────────────────────
    st.subheader("📈 Scenario Analysis — Line Trend")

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=scenario_df["Scenario"],
        y=scenario_df["Predicted Cashflow"],
        mode="lines+markers+text",
        line=dict(color="#9b59b6", width=3),
        marker=dict(size=10, color=[
            "#2ecc71" if v >= 0 else "#e74c3c" for v in scenario_values
        ], line=dict(width=2, color="white")),
        text=[f"${v:,.0f}" for v in scenario_values],
        textposition="top center",
        name="Predicted Cashflow",
    ))
    # Zero reference line
    fig_line.add_hline(
        y=0,
        line_dash="dash",
        line_color="#888",
        annotation_text="Break-even",
        annotation_position="bottom right",
    )
    # Shade negative region
    fig_line.add_hrect(
        y0=min(scenario_values) * 1.15, y1=0,
        fillcolor="#e74c3c", opacity=0.07,
        layer="below", line_width=0,
    )
    fig_line.update_layout(
        title="Cash Flow Trend Across Scenarios",
        xaxis_title="Scenario",
        yaxis_title="Predicted Cashflow ($)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        hovermode="x unified",
    )
    st.plotly_chart(fig_line, use_container_width=True)

# ── Tabs: Batch Prediction & Model Insights ───────────────────────────────────
tab1, tab2 = st.tabs(["Batch Prediction", "Model Insights"])

with tab1:
    uploaded = st.file_uploader("Upload CSV for batch prediction", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        processed = pipeline.transform(df)
        df["predicted_net_cashflow"] = model.predict(processed)
        st.dataframe(df)

        # Line chart for batch results
        if "predicted_net_cashflow" in df.columns:
            st.subheader("📈 Batch Prediction Trend")
            fig_batch = go.Figure()
            fig_batch.add_trace(go.Scatter(
                x=df.index,
                y=df["predicted_net_cashflow"],
                mode="lines+markers",
                line=dict(color="#3498db", width=2),
                marker=dict(size=5),
                name="Predicted Cash Flow",
            ))
            fig_batch.add_hline(y=0, line_dash="dash", line_color="#e74c3c",
                                annotation_text="Break-even")
            fig_batch.update_layout(
                title="Predicted Net Cash Flow — All Transactions",
                xaxis_title="Transaction Index",
                yaxis_title="Net Cash Flow ($)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_batch, use_container_width=True)

        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")

with tab2:
    st.info("Model Insights coming soon — feature importance, SHAP values, etc.")

st.caption("Built with Streamlit • Random Forest Model • DevOps MLOps Pipeline testing")