"""
========================================================================================
Electricity Demand–Supply Gap Forecasting & RAG-Based Energy Planning Web Application
========================================================================================
A Confidence-Driven Probabilistic Machine Learning Framework for Electricity Demand–Supply
Gap Forecasting and RAG-Based Energy Planning.

Authors: Final Year Engineering Project Team
========================================================================================
"""

import sys
from pathlib import Path

# Add webapp directory to sys.path
WEBAPP_DIR = Path(__file__).resolve().parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

import streamlit as st
import numpy as np
import pandas as pd

import config
from services.demand_forecasting import DemandForecaster
from services.supply_forecasting import SupplyForecaster
from services.gap_analysis import perform_gap_analysis
from services.risk_assessment import assess_risk
from services.rag_planning import generate_energy_planning_recommendation
from utils.visualization import (
    create_forecast_comparison_chart,
    create_prediction_interval_chart,
    create_gap_distribution_chart,
    create_risk_gauge,
    create_validation_comparison_chart,
)
from utils.preprocessing import encode_cyclical_month


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Electricity Demand–Supply Gap Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Professional Data-Science Theme
st.markdown(
    """
    <style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 4px;
    }
    .metric-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }
    
    /* Risk Badges */
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Policy recommendation box */
    .rec-box {
        background-color: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Model Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Demand Forecasting LSTM Model & Scalers...")
def get_demand_forecaster() -> DemandForecaster:
    return DemandForecaster()


@st.cache_resource(show_spinner="Loading Supply Forecasting LSTM Model & Scalers...")
def get_supply_forecaster() -> SupplyForecaster:
    return SupplyForecaster()


# -----------------------------------------------------------------------------
# App Header
# -----------------------------------------------------------------------------
st.title("⚡ Electricity Demand–Supply Gap Forecasting")
st.subheader("Confidence-Driven Probabilistic Forecasting and Energy Planning")
st.caption(
    "A Confidence-Driven Probabilistic Machine Learning Framework with Independent LSTM Pipelines, "
    "Residual-Based Uncertainty Quantification (90%, 95%, 99% Prediction Intervals), and RAG-Grounded Energy Planning."
)
st.markdown("---")


# -----------------------------------------------------------------------------
# Check Model Availability
# -----------------------------------------------------------------------------
try:
    demand_forecaster = get_demand_forecaster()
    supply_forecaster = get_supply_forecaster()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(f"⚠️ **Model Initialization Error**: {e}")
    st.info(
        "Please ensure the trained models and scalers are located in their respective directories:\n"
        f"- Demand Model: `{config.DEMAND_MODEL_PATH}`\n"
        f"- Supply Model: `{config.SUPPLY_MODEL_PATH}`\n"
    )


# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab_forecast, tab_validation, tab_models, tab_rag, tab_methodology = st.tabs([
    "🔮 Forecast Dashboard",
    "📊 Validation Results (Jan–Mar 2026)",
    "🧠 Model Architecture & Comparison",
    "🏛️ RAG Energy Planning & Policies",
    "📐 Statistical Methodology",
])


# =============================================================================
# TAB 1: FORECAST DASHBOARD
# =============================================================================
with tab_forecast:
    if not models_ready:
        st.warning("Forecasting models are currently unavailable. Check the error message above.")
    else:
        # Sidebar Controls
        st.sidebar.header("⚙️ Forecast Scenario Configuration")
        
        input_mode = st.sidebar.radio(
            "Select Forecasting Mode:",
            ["Validation Benchmark (Jan–Mar 2026)", "Custom Forecasting Scenario"],
            help="Choose a pre-validated benchmark month with authentic project inputs or customize your own input scenario.",
        )

        forecast_triggered = False

        if input_mode == "Validation Benchmark (Jan–Mar 2026)":
            selected_month = st.sidebar.selectbox(
                "Select Target Month:",
                ["Jan 2026", "Feb 2026", "Mar 2026"],
                index=0,
            )

            st.sidebar.markdown("### 📌 Scenario Description")
            st.sidebar.info(
                f"**Target Month:** `{selected_month}`\n\n"
                "- Uses authentic chronological feature sequences from project validation sets (`Book1.xlsx` and `supply_dataset.csv`).\n"
                "- Demand lookback: 3 months sequence (14 features).\n"
                "- Supply lookback: 6 months sequence (18 features).\n"
                "- Zero cross-dependency between Demand and Supply."
            )

            if st.sidebar.button("🚀 Run Forecast", type="primary", use_container_width=True):
                forecast_triggered = True

        else:
            # Custom Scenario Inputs
            st.sidebar.markdown("### 🗓️ Target Date & Calendar Features")
            custom_year = st.sidebar.number_input("Forecast Year:", min_value=2024, max_value=2030, value=2026, step=1)
            custom_month = st.sidebar.slider("Forecast Month (1–12):", min_value=1, max_value=12, value=4)
            is_festival = st.sidebar.selectbox("Festival Month?", [0, 1], index=0, format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            
            # Demand Weather & Lags
            st.sidebar.markdown("### 🌡️ Demand Inputs (Weather & History)")
            c_temp = st.sidebar.number_input("Avg Temperature (°C):", value=29.5, step=0.1)
            c_rain = st.sidebar.number_input("Rainfall (mm):", value=35.0, step=1.0)
            c_humid = st.sidebar.number_input("Humidity (%):", value=68.0, step=0.5)
            c_solar = st.sidebar.number_input("Solar Irradiance (W/m²):", value=220.0, step=5.0)
            
            c_d_lag1 = st.sidebar.number_input("Demand Lag 1 (t-1 MU):", value=12233.0, step=50.0)
            c_d_lag2 = st.sidebar.number_input("Demand Lag 2 (t-2 MU):", value=10125.0, step=50.0)
            c_d_lag3 = st.sidebar.number_input("Demand Lag 3 (t-3 MU):", value=10067.0, step=50.0)

            # Supply Generation Lags
            st.sidebar.markdown("### ⚡ Supply Inputs (Generation Lags in MU)")
            c_coal = st.sidebar.number_input("Coal Generation (t-1 MU):", value=6000.0, step=50.0)
            c_hydro = st.sidebar.number_input("Hydro Generation (t-1 MU):", value=200.0, step=10.0)
            c_nuclear = st.sidebar.number_input("Nuclear Generation (t-1 MU):", value=1500.0, step=20.0)
            c_wind = st.sidebar.number_input("Wind Generation (t-1 MU):", value=800.0, step=20.0)
            c_sol_gen = st.sidebar.number_input("Solar Generation (t-1 MU):", value=1800.0, step=20.0)
            c_oil_gas = st.sidebar.number_input("Oil & Gas (t-1 MU):", value=90.0, step=5.0)
            c_small_hydro = st.sidebar.number_input("Small-Hydro (t-1 MU):", value=15.0, step=2.0)
            c_bio = st.sidebar.number_input("Bio Power (t-1 MU):", value=10.0, step=2.0)
            c_tot_lag1 = st.sidebar.number_input("Total Supply Lag 1 (t-1 MU):", value=11247.5, step=50.0)

            if st.sidebar.button("🚀 Run Custom Forecast", type="primary", use_container_width=True):
                forecast_triggered = True

        # Default run on initial load
        if not forecast_triggered and "forecast_run" not in st.session_state:
            forecast_triggered = True
            selected_month = "Jan 2026"
            input_mode = "Validation Benchmark (Jan–Mar 2026)"

        # Execute Forecasting Pipeline
        if forecast_triggered or "forecast_results" in st.session_state:
            if forecast_triggered:
                with st.spinner("Executing independent Demand & Supply probabilistic inference..."):
                    if input_mode == "Validation Benchmark (Jan–Mar 2026)":
                        d_res = demand_forecaster.predict_benchmark_month(selected_month)
                        s_res = supply_forecaster.predict_benchmark_month(selected_month)
                        month_display = selected_month
                    else:
                        # Construct 3-month Demand sequence for custom scenario
                        m_sin, m_cos = encode_cyclical_month(custom_month)
                        roll_3 = (c_d_lag1 + c_d_lag2 + c_d_lag3) / 3.0
                        roll_6 = roll_3 * 1.02
                        roll_12 = roll_3 * 1.05

                        d_row = [
                            c_humid, c_rain, c_solar, c_temp, custom_year,
                            m_sin, m_cos, is_festival,
                            c_d_lag1, c_d_lag2, c_d_lag3,
                            roll_3, roll_6, roll_12
                        ]
                        d_seq = np.tile(d_row, (3, 1))
                        d_res = demand_forecaster.predict(d_seq)

                        # Construct 6-month Supply sequence for custom scenario
                        quarter = (custom_month - 1) // 3 + 1
                        s_row = [
                            c_coal, c_oil_gas, c_nuclear, c_hydro,
                            c_sol_gen, c_wind, c_small_hydro, c_bio,
                            custom_year, custom_month, quarter,
                            c_tot_lag1, c_tot_lag1 * 0.98, c_tot_lag1 * 0.95, c_tot_lag1 * 0.92,
                            c_tot_lag1 * 0.99, c_tot_lag1 * 0.97, c_tot_lag1 * 0.96
                        ]
                        s_seq = np.tile(s_row, (6, 1))
                        s_res = supply_forecaster.predict(s_seq)
                        month_display = f"Custom Scenario ({custom_year}-{custom_month:02d})"

                    # Gap analysis & Risk classification
                    gap_res = perform_gap_analysis(d_res, s_res)
                    risk_res = assess_risk(gap_res["predicted_gap"], gap_res["condition"])
                    rag_res = generate_energy_planning_recommendation(gap_res, risk_res)

                    st.session_state["forecast_results"] = {
                        "demand": d_res,
                        "supply": s_res,
                        "gap": gap_res,
                        "risk": risk_res,
                        "rag": rag_res,
                        "month_display": month_display,
                    }

            # Retrieve saved results
            res = st.session_state["forecast_results"]
            d_res = res["demand"]
            s_res = res["supply"]
            gap_res = res["gap"]
            risk_res = res["risk"]
            rag_res = res["rag"]
            month_display = res["month_display"]

            # -----------------------------------------------------------------
            # 1. SUMMARY CARDS
            # -----------------------------------------------------------------
            st.markdown(f"### 📋 Forecast Results Summary — `{month_display}`")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            
            with c1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Demand</div>
                        <div class="metric-value">{d_res['predicted_demand']:,.2f} <span style="font-size:1rem;color:#94A3B8">MU</span></div>
                        <div class="metric-subtext">σ_D = {d_res['sigma']:.2f} MU</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with c2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Supply</div>
                        <div class="metric-value">{s_res['predicted_supply']:,.2f} <span style="font-size:1rem;color:#94A3B8">MU</span></div>
                        <div class="metric-subtext">σ_S = {s_res['sigma']:.2f} MU</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                gap_val = gap_res['predicted_gap']
                gap_color = "#EF4444" if gap_val > 0 else "#10B981"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Demand–Supply Gap</div>
                        <div class="metric-value" style="color:{gap_color}">{gap_val:,.2f} <span style="font-size:1rem;color:#94A3B8">MU</span></div>
                        <div class="metric-subtext">σ_Gap = {gap_res['gap_sigma']:.2f} MU</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c4:
                condition_label = gap_res['condition'].upper()
                cond_badge_class = "badge-high" if condition_label == "SHORTAGE" else "badge-low"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Grid Condition</div>
                        <div style="margin-top:8px;">
                            <span class="{cond_badge_class}">{condition_label}</span>
                        </div>
                        <div class="metric-subtext" style="margin-top:10px;">Demand vs Supply State</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c5:
                risk_lvl = risk_res['risk_level'].upper()
                badge_class = (
                    "badge-low" if risk_lvl == "LOW" else ("badge-moderate" if risk_lvl == "MODERATE" else "badge-high")
                )
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Risk Classification</div>
                        <div style="margin-top:8px;">
                            <span class="{badge_class}">{risk_lvl} RISK</span>
                        </div>
                        <div class="metric-subtext" style="margin-top:10px;">Deterministic Threshold</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 95% Confidence Interval Highlight Banner
            p95 = gap_res["gap_intervals"]["pi_95"]
            st.info(
                f"🎯 **95% Gap Prediction Interval**: **[{p95['lower']:,.2f} MU  to  {p95['upper']:,.2f} MU]** "
                f"(Margin: ±{p95['margin']:,.2f} MU at z = 1.960). "
                f"{risk_res['description']}"
            )

            # -----------------------------------------------------------------
            # 2. INTERACTIVE CHARTS (Comparison & Distribution)
            # -----------------------------------------------------------------
            st.markdown("---")
            col_left, col_right = st.columns([1, 1])

            with col_left:
                fig_comp = create_forecast_comparison_chart(
                    d_res["predicted_demand"],
                    s_res["predicted_supply"],
                    gap_res["predicted_gap"],
                    month_label=month_display,
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            with col_right:
                fig_dist = create_gap_distribution_chart(
                    gap_res["predicted_gap"],
                    gap_res["gap_sigma"],
                    gap_res["gap_intervals"],
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            # -----------------------------------------------------------------
            # 3. PROBABILISTIC PREDICTION INTERVALS (Demand & Supply)
            # -----------------------------------------------------------------
            st.markdown("### 📐 Probabilistic Prediction Interval Breakdowns")
            
            d_col, s_col = st.columns(2)

            with d_col:
                st.markdown("#### 🔵 Demand Forecast Intervals")
                d_intervals = d_res["intervals"]
                
                # Table of intervals
                d_table_data = [
                    {"Confidence": "Point Forecast", "Lower Bound": "—", "Upper Bound": "—", "Margin": "—", "Value (MU)": f"{d_res['predicted_demand']:,.2f}"},
                    {"Confidence": "90% Interval (z=1.645)", "Lower Bound": f"{d_intervals['pi_90']['lower']:,.2f}", "Upper Bound": f"{d_intervals['pi_90']['upper']:,.2f}", "Margin": f"±{d_intervals['pi_90']['margin']:,.2f}", "Value (MU)": f"[{d_intervals['pi_90']['lower']:,.2f}, {d_intervals['pi_90']['upper']:,.2f}]"},
                    {"Confidence": "95% Interval (z=1.960)", "Lower Bound": f"{d_intervals['pi_95']['lower']:,.2f}", "Upper Bound": f"{d_intervals['pi_95']['upper']:,.2f}", "Margin": f"±{d_intervals['pi_95']['margin']:,.2f}", "Value (MU)": f"[{d_intervals['pi_95']['lower']:,.2f}, {d_intervals['pi_95']['upper']:,.2f}]"},
                    {"Confidence": "99% Interval (z=2.576)", "Lower Bound": f"{d_intervals['pi_99']['lower']:,.2f}", "Upper Bound": f"{d_intervals['pi_99']['upper']:,.2f}", "Margin": f"±{d_intervals['pi_99']['margin']:,.2f}", "Value (MU)": f"[{d_intervals['pi_99']['lower']:,.2f}, {d_intervals['pi_99']['upper']:,.2f}]"},
                ]
                st.dataframe(pd.DataFrame(d_table_data), hide_index=True, use_container_width=True)
                
                fig_d_int = create_prediction_interval_chart(
                    "Demand",
                    d_intervals,
                    theme_color=config.DEMAND_FEATURES,
                )
                st.plotly_chart(fig_d_int, use_container_width=True)

            with s_col:
                st.markdown("#### 🟢 Supply Forecast Intervals")
                s_intervals = s_res["intervals"]
                
                # Table of intervals
                s_table_data = [
                    {"Confidence": "Point Forecast", "Lower Bound": "—", "Upper Bound": "—", "Margin": "—", "Value (MU)": f"{s_res['predicted_supply']:,.2f}"},
                    {"Confidence": "90% Interval (z=1.645)", "Lower Bound": f"{s_intervals['pi_90']['lower']:,.2f}", "Upper Bound": f"{s_intervals['pi_90']['upper']:,.2f}", "Margin": f"±{s_intervals['pi_90']['margin']:,.2f}", "Value (MU)": f"[{s_intervals['pi_90']['lower']:,.2f}, {s_intervals['pi_90']['upper']:,.2f}]"},
                    {"Confidence": "95% Interval (z=1.960)", "Lower Bound": f"{s_intervals['pi_95']['lower']:,.2f}", "Upper Bound": f"{s_intervals['pi_95']['upper']:,.2f}", "Margin": f"±{s_intervals['pi_95']['margin']:,.2f}", "Value (MU)": f"[{s_intervals['pi_95']['lower']:,.2f}, {s_intervals['pi_95']['upper']:,.2f}]"},
                    {"Confidence": "99% Interval (z=2.576)", "Lower Bound": f"{s_intervals['pi_99']['lower']:,.2f}", "Upper Bound": f"{s_intervals['pi_99']['upper']:,.2f}", "Margin": f"±{s_intervals['pi_99']['margin']:,.2f}", "Value (MU)": f"[{s_intervals['pi_99']['lower']:,.2f}, {s_intervals['pi_99']['upper']:,.2f}]"},
                ]
                st.dataframe(pd.DataFrame(s_table_data), hide_index=True, use_container_width=True)

                fig_s_int = create_prediction_interval_chart(
                    "Supply",
                    s_intervals,
                    theme_color="#10B981",
                )
                st.plotly_chart(fig_s_int, use_container_width=True)

            # -----------------------------------------------------------------
            # 4. RISK ASSESSMENT GAUGE & DECISION THRESHOLDS
            # -----------------------------------------------------------------
            st.markdown("---")
            st.markdown("### 🛡️ Risk Assessment & Decision Framework")
            
            rg_col, meta_col = st.columns([1, 1])

            with rg_col:
                fig_gauge = create_risk_gauge(gap_res["predicted_gap"], risk_res["risk_level"])
                st.plotly_chart(fig_gauge, use_container_width=True)

            with meta_col:
                st.markdown("#### 📋 Deterministic Risk Threshold Definitions")
                st.markdown(
                    """
                    | Risk Tier | Gap Condition | Operational Protocol |
                    | :--- | :--- | :--- |
                    | **🟢 Low Risk** | $\\text{Gap} < 3,000\\text{ MU}$ | Manageable with routine thermal spinning reserves & bilateral procurement. |
                    | **🟡 Moderate Risk** | $3,000\\text{ MU} \\le \\text{Gap} \\le 4,500\\text{ MU}$ | Peaking plant pre-warming, DAM exchange trades, industrial demand response. |
                    | **🔴 High Risk** | $\\text{Gap} > 4,500\\text{ MU}$ | Emergency interstate power imports, BESS full discharge, graded load management. |
                    """
                )
                st.caption(
                    "⚠️ **Notice:** These thresholds are project-defined deterministic decision boundaries. "
                    "They are NOT calculated by RAG or language models."
                )


# =============================================================================
# TAB 2: VALIDATION RESULTS (Jan - Mar 2026)
# =============================================================================
with tab_validation:
    st.header("📊 Out-of-Sample Validation Benchmarks (Jan – Mar 2026)")
    st.markdown(
        "Comparison of model point forecasts against unseen actual observations recorded for "
        "January, February, and March 2026."
    )

    # Detailed Table
    val_rows = []
    for m in ["Jan 2026", "Feb 2026", "Mar 2026"]:
        b = config.VALIDATION_BENCHMARKS[m]
        d_err = abs(b["demand_actual"] - b["demand_predicted"])
        d_ape = (d_err / b["demand_actual"]) * 100
        s_err = abs(b["supply_actual"] - b["supply_predicted"])
        s_ape = (s_err / b["supply_actual"]) * 100

        val_rows.append({
            "Month": m,
            "Actual Demand (MU)": f"{b['demand_actual']:,.2f}",
            "Predicted Demand (MU)": f"{b['demand_predicted']:,.2f}",
            "Demand Error (MU)": f"{d_err:,.2f} ({d_ape:.2f}%)",
            "Actual Supply (MU)": f"{b['supply_actual']:,.2f}",
            "Predicted Supply (MU)": f"{b['supply_predicted']:,.2f}",
            "Supply Error (MU)": f"{s_err:,.2f} ({s_ape:.2f}%)",
            "Actual Gap (MU)": f"{b['gap_actual']:,.2f}",
            "Predicted Gap (MU)": f"{b['gap_predicted']:,.2f}",
            "Condition": "Shortage",
        })

    df_val_table = pd.DataFrame(val_rows)
    st.dataframe(df_val_table, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.plotly_chart(create_validation_comparison_chart(), use_container_width=True)


# =============================================================================
# TAB 3: MODEL ARCHITECTURE & COMPARISON
# =============================================================================
with tab_models:
    st.header("🧠 Model Architecture & Development Benchmarks")
    st.markdown(
        "During model development, four machine learning and deep learning architectures "
        "were trained and evaluated using chronological time-series splits."
    )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("Demand Forecasting Models ($R^2$ Scores)")
        d_scores = pd.DataFrame([
            {"Model": "Random Forest", "R² Score": config.MODEL_DEVELOPMENT_SCORES["Demand"]["Random Forest"], "Status": "Evaluated"},
            {"Model": "XGBoost", "R² Score": config.MODEL_DEVELOPMENT_SCORES["Demand"]["XGBoost"], "Status": "Evaluated"},
            {"Model": "LightGBM", "R² Score": config.MODEL_DEVELOPMENT_SCORES["Demand"]["LightGBM"], "Status": "Evaluated"},
            {"Model": "Stacked LSTM (Selected)", "R² Score": config.MODEL_DEVELOPMENT_SCORES["Demand"]["LSTM"], "Status": "⭐ Selected Final Model"},
        ])
        st.dataframe(d_scores, hide_index=True, use_container_width=True)

        st.info(
            "**Demand LSTM Architecture:**\n"
            "- Lookback sequence length: `3 months`\n"
            "- Input features: `14 features` (Weather, Calendar, Lags 1-3, Rolling 3/6/12)\n"
            "- Layer 1: LSTM (64/96 units, tanh activation)\n"
            "- Layer 2: LSTM (32 units, tanh activation)\n"
            "- Dropout: 0.0 – 0.3\n"
            "- Output: Dense(1)"
        )

    with col_m2:
        st.subheader("Supply Forecasting Models ($R^2$ Scores)")
        s_scores = pd.DataFrame([
            {"Model": "Random Forest", "R² Score": config.MODEL_DEVELOPMENT_SCORES["Supply"]["Random Forest"], "Status": "Evaluated"},
            {"Model": "XGBoost", "R² Score": config.MODEL_DEVELOPMENT_SCORES["Supply"]["XGBoost"], "Status": "Evaluated"},
            {"Model": "LightGBM", "R² Score": config.MODEL_DEVELOPMENT_SCORES["Supply"]["LightGBM"], "Status": "Evaluated"},
            {"Model": "Tuned LSTM (Selected)", "R² Score": config.MODEL_DEVELOPMENT_SCORES["Supply"]["LSTM"], "Status": "⭐ Selected Final Model"},
        ])
        st.dataframe(s_scores, hide_index=True, use_container_width=True)

        st.info(
            "**Supply LSTM Architecture:**\n"
            "- Lookback sequence length: `6 months`\n"
            "- Input features: `18 features` (Fuel-specific generation lags, Calendar, Lags 1-12, Rolling 3/6/12)\n"
            "- Layer 1: LSTM (64 units, tanh activation)\n"
            "- Layer 2: LSTM (32 units, tanh activation)\n"
            "- Output: Dense(1)"
        )

    st.markdown("---")
    st.warning(
        "📌 **Clear Distinction:** The $R^2$ scores above represent the initial development comparison on the historical "
        "test set (up to Dec 2025). The out-of-sample validation table in Tab 2 reflects real performance on unseen 2026 data."
    )


# =============================================================================
# TAB 4: RAG ENERGY PLANNING & POLICIES
# =============================================================================
with tab_rag:
    st.header("🏛️ RAG-Based Energy Planning & Policy Recommendation System")
    st.markdown(
        "Retrieval-Augmented Generation (RAG) grounds operational energy planning recommendations "
        "in official state and national electricity regulations, guidelines, and SLDC operating norms."
    )

    st.markdown(
        """
        ```
        +-------------------------+       +-------------------------+
        | Demand Forecast (LSTM)  |       | Supply Forecast (LSTM)  |
        +-------------------------+       +-------------------------+
                     \\                               /
                      \\                             /
                       v                           v
                      +-----------------------------+
                      |    Gap Analysis (D - S)     |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Deterministic Risk Ranking  |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Knowledge Retrieval (FAISS) |
                      | - CEA Thermal Norms         |
                      | - TNERC DSM Directives      |
                      | - National BESS Policy      |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Evidence-Grounded Planning  |
                      | Action Recommendations      |
                      +-----------------------------+
        ```
        """
    )

    if "forecast_results" in st.session_state:
        rag_data = st.session_state["forecast_results"]["rag"]
        gap_data = st.session_state["forecast_results"]["gap"]
        risk_data = st.session_state["forecast_results"]["risk"]

        st.markdown(f"### 📋 Current Operational Recommendations ({res['month_display']})")
        st.markdown(
            f"**Condition:** `{gap_data['condition']}` | **Risk:** `{risk_data['risk_level']} Risk` | "
            f"**Gap:** `{gap_data['predicted_gap']:,.2f} MU` (95% CI: `{rag_data['ci_95_range'][0]:,.2f}` to `{rag_data['ci_95_range'][1]:,.2f}` MU)"
        )

        for rec in rag_data["recommendations"]:
            st.markdown(
                f"""
                <div class="rec-box">
                    <div style="font-weight:700;font-size:1.05rem;color:#F8FAFC">{rec['category']}</div>
                    <div style="font-weight:600;font-size:0.95rem;color:#38BDF8;margin-top:4px;">{rec['action']}</div>
                    <div style="color:#CBD5E1;font-size:0.9rem;margin-top:6px;">{rec['details']}</div>
                    <div style="color:#94A3B8;font-size:0.8rem;margin-top:6px;font-style:italic">Grounded In: {rec['policy_grounding']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 📚 Retrieved Regulatory Knowledge Base Excerpts")
        for doc in rag_data["retrieved_knowledge"]:
            with st.expander(f"📖 {doc['id']}: {doc['title']}"):
                st.markdown(f"**Regulatory Reference:** {doc['reference']}")
                st.markdown(f"**Policy Directive:** {doc['content']}")

    st.markdown("---")
    st.info(
        "💡 **RAG Architecture Disclosure:** RAG is utilized as an evidence-grounded recommendation engine. "
        "It retrieves verified domain policies and does not alter numerical machine learning predictions."
    )


# =============================================================================
# TAB 5: STATISTICAL METHODOLOGY
# =============================================================================
with tab_methodology:
    st.header("📐 Statistical & Mathematical Methodology")
    st.markdown(
        "The web application faithfully implements the exact probabilistic confidence-interval estimation "
        "equations from `CI_estimation.ipynb`."
    )

    st.markdown(
        r"""
        ### 1. Residual Standard Error ($\sigma_e$)
        The standard error of forecast residuals is estimated from model validation predictions up to December 2025:
        $$\sigma_e = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} \left( y_{\text{act}, i} - \hat{y}_{\text{pred}, i} \right)^2}$$

        - **Demand Standard Error ($\sigma_D$):** `391.30 MU`
        - **Supply Standard Error ($\sigma_S$):** `996.83 MU`
        
        ### 2. Combined Demand–Supply Gap Standard Error ($\sigma_{\text{Gap}}$)
        Assuming independent residual distributions between the demand and supply forecasting models:
        $$\sigma_{\text{Gap}} = \sqrt{\sigma_D^2 + \sigma_S^2} = \sqrt{(391.30)^2 + (996.83)^2} = 1070.88\text{ MU}$$

        ### 3. Prediction Intervals ($\text{PI}_{1-\alpha}$)
        Symmetric prediction intervals are centered on the point forecast $\hat{y}$ using standard normal multipliers ($z_{1-\alpha/2}$):
        $$\text{PI}_{1-\alpha} = \left[ \hat{y} - z_{1-\alpha/2} \cdot \sigma_e, \quad \hat{y} + z_{1-\alpha/2} \cdot \sigma_e \right]$$

        - **90% Interval ($\alpha = 0.10$):** $z_{0.95} = 1.645 \implies \hat{y} \pm 1.645 \cdot \sigma_e$
        - **95% Interval ($\alpha = 0.05$):** $z_{0.975} = 1.960 \implies \hat{y} \pm 1.960 \cdot \sigma_e$
        - **99% Interval ($\alpha = 0.01$):** $z_{0.995} = 2.576 \implies \hat{y} \pm 2.576 \cdot \sigma_e$

        ### 4. Deterministic Risk Classification
        - $\text{Gap} < 3000\text{ MU} \implies \text{Low Risk}$
        - $3000\text{ MU} \le \text{Gap} \le 4500\text{ MU} \implies \text{Moderate Risk}$
        - $\text{Gap} > 4500\text{ MU} \implies \text{High Risk}$
        """
    )
