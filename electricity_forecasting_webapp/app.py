"""
========================================================================================
Electricity Demand–Supply Gap Forecasting & RAG-Based Energy Planning Dashboard
========================================================================================
Confidence-Driven Probabilistic Machine Learning Framework for Electricity Demand–Supply
Gap Forecasting and RAG-Based Energy Planning.

Authors: Final Year Engineering Project Team
Domain: Deep Learning, Probabilistic Uncertainty Quantification & Applied RAG Planning
========================================================================================
"""

import os
import sys
from pathlib import Path

# Disable GPU initialization, limit OpenMP threads, and set single-thread mode for Linux containers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    create_multi_month_timeline_chart,
    COLOR_DEMAND,
    COLOR_SUPPLY,
)
from utils.preprocessing import encode_cyclical_month


def render_html(html_str: str):
    """
    Renders pure HTML in Streamlit by stripping all leading and trailing indentation
    from every line. This completely prevents Markdown from ever treating HTML as code blocks.
    """
    clean_html = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(clean_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Modern Engineering Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Electricity Demand–Supply Gap Forecasting | IEEE Project",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Professional Energy-Tech CSS
render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .dashboard-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.95rem;
        font-weight: 500;
        color: #94A3B8;
        margin-top: 4px;
    }
    .header-badges {
        margin-top: 10px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .pill-badge {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 9999px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .kpi-card {
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.85), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        border-color: rgba(96, 165, 250, 0.4);
    }
    .kpi-card.gap-card {
        border: 1.5px solid rgba(139, 92, 246, 0.4);
        background: linear-gradient(145deg, rgba(23, 15, 38, 0.9), rgba(15, 23, 42, 0.95));
    }
    .kpi-tag {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
        display: inline-block;
    }
    .tag-demand { color: #60A5FA; }
    .tag-supply { color: #34D399; }
    .tag-gap { color: #C084FC; }

    .kpi-hero-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-hero-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.45rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 4px 0 8px 0;
        letter-spacing: -0.02em;
    }
    .hero-demand { color: #93C5FD; text-shadow: 0 0 12px rgba(59, 130, 246, 0.3); }
    .hero-supply { color: #6EE7B7; text-shadow: 0 0 12px rgba(16, 185, 129, 0.3); }
    .hero-gap { color: #FCA5A5; text-shadow: 0 0 12px rgba(239, 68, 68, 0.3); }

    .range-bar-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 8px 10px;
        margin: 10px 0;
    }
    .range-line-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #94A3B8;
    }
    .range-track {
        flex-grow: 1;
        height: 4px;
        background: rgba(255, 255, 255, 0.15);
        margin: 0 8px;
        border-radius: 2px;
        position: relative;
    }
    .range-fill-demand { background: linear-gradient(90deg, #3B82F6, #60A5FA); }
    .range-fill-supply { background: linear-gradient(90deg, #059669, #34D399); }
    .range-fill-gap { background: linear-gradient(90deg, #D97706, #EF4444); }
    
    .range-marker {
        position: absolute;
        top: -4px;
        left: 50%;
        width: 12px;
        height: 12px;
        background: #F59E0B;
        border: 2px solid #FFFFFF;
        border-radius: 50%;
        transform: translateX(-50%);
        box-shadow: 0 0 6px #F59E0B;
    }

    .kpi-subrow {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 8px;
        margin-top: 8px;
        font-size: 0.8rem;
    }
    .subrow-label { color: #94A3B8; }
    .subrow-val {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: #F8FAFC;
    }

    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .risk-low {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid #10B981;
    }
    .risk-mod {
        background: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid #F59E0B;
    }
    .risk-high {
        background: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid #EF4444;
    }

    .summary-alert-box {
        background: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #3B82F6;
        border-radius: 0 10px 10px 0;
        padding: 12px 18px;
        margin: 16px 0 24px 0;
        font-size: 0.88rem;
        color: #E2E8F0;
    }

    .rec-panel {
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .rec-cat {
        font-size: 0.8rem;
        font-weight: 700;
        color: #60A5FA;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .rec-action {
        font-size: 1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 4px 0;
    }
    .rec-desc {
        font-size: 0.88rem;
        color: #CBD5E1;
        line-height: 1.45;
    }
    .rec-cite {
        font-size: 0.78rem;
        color: #94A3B8;
        font-style: italic;
        margin-top: 6px;
    }

    .formula-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    </style>
    """
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
# Header Banner
# -----------------------------------------------------------------------------
render_html(
    """
    <div class="dashboard-header">
        <div class="header-title">⚡ ELECTRICITY DEMAND–SUPPLY GAP FORECASTING DASHBOARD</div>
        <div class="header-subtitle">Confidence-Driven Probabilistic Machine Learning Framework & RAG-Based Energy Planning</div>
        <div class="header-badges">
            <span class="pill-badge">Decoupled Dual-Stream LSTM</span>
            <span class="pill-badge">Multi-Tier Prediction Intervals (90%, 95%, 99%)</span>
            <span class="pill-badge">Deterministic Risk Assessment</span>
            <span class="pill-badge">FAISS Regulatory RAG Planning</span>
            <span class="pill-badge">Tamil Nadu Power Grid Scope</span>
        </div>
    </div>
    """
)


# -----------------------------------------------------------------------------
# Top Control Bar (Month Selection & Confidence Tier Toggle)
# -----------------------------------------------------------------------------
col_ctl1, col_ctl2, col_ctl3 = st.columns([2.2, 1.8, 1.0])

with col_ctl1:
    selected_month = st.selectbox(
        "🎯 Select Operational Planning Target:",
        ["Jan 2026", "Feb 2026", "Mar 2026", "Custom Scenario (Configure in Sidebar)"],
        index=0,
        help="Select a benchmark hold-out validation month or switch to a custom forecasting scenario.",
    )

with col_ctl2:
    selected_ci = st.radio(
        "📐 Active Prediction Interval Tier:",
        ["90% Interval", "🎯 95% Interval (Default / IEEE Benchmark)", "99% Interval"],
        index=1,
        horizontal=True,
        help="95% is the primary academic benchmark. You can also inspect 90% and 99% confidence margins.",
    )
    ci_label = "95%" if "95%" in selected_ci else ("90%" if "90%" in selected_ci else "99%")
    ci_key = "pi_95" if ci_label == "95%" else ("pi_90" if ci_label == "90%" else "pi_99")

with col_ctl3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    run_btn = st.button("🔄 Recalculate", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Sidebar Configuration (Custom Scenarios)
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Scenario Configuration")
input_mode = "Custom Forecasting Scenario" if "Custom" in selected_month else "Validation Benchmark (Jan–Mar 2026)"

if input_mode == "Validation Benchmark (Jan–Mar 2026)":
    st.sidebar.markdown(f"**Target Period:** `{selected_month}`")
    st.sidebar.info(
        "📌 **Validation Benchmark Mode Active**\n\n"
        "- Uses authentic chronological sequence matrices from `Book1.xlsx` and `supply_dataset.csv`.\n"
        "- **Demand Lookback:** 3 months historical sequence (14 features).\n"
        "- **Supply Lookback:** 6 months historical sequence (18 features).\n"
        "- **Decoupled Stream:** Zero cross-dependency between Demand and Supply modules."
    )
else:
    st.sidebar.markdown("### 🗓️ Target Calendar Features")
    custom_year = st.sidebar.number_input("Forecast Year:", min_value=2024, max_value=2030, value=2026, step=1)
    custom_month = st.sidebar.slider("Forecast Month (1–12):", min_value=1, max_value=12, value=4)
    is_festival = st.sidebar.selectbox("Festival Month?", [0, 1], index=0, format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    
    st.sidebar.markdown("### 🌡️ Demand Inputs (Weather & History)")
    c_temp = st.sidebar.number_input("Avg Temperature (°C):", value=29.5, step=0.1)
    c_rain = st.sidebar.number_input("Rainfall (mm):", value=35.0, step=1.0)
    c_humid = st.sidebar.number_input("Humidity (%):", value=68.0, step=0.5)
    c_solar = st.sidebar.number_input("Solar Irradiance (W/m²):", value=220.0, step=5.0)
    
    c_d_lag1 = st.sidebar.number_input("Demand Lag 1 (t-1 MU):", value=12233.0, step=50.0)
    c_d_lag2 = st.sidebar.number_input("Demand Lag 2 (t-2 MU):", value=10125.0, step=50.0)
    c_d_lag3 = st.sidebar.number_input("Demand Lag 3 (t-3 MU):", value=10067.0, step=50.0)

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

st.sidebar.markdown("---")
st.sidebar.caption("⚡ **Tamil Nadu State Load Despatch Centre (SLDC) Analytics System**")


# -----------------------------------------------------------------------------
# Execute Inference Pipeline
# -----------------------------------------------------------------------------
if models_ready:
    with st.spinner("Executing independent Demand & Supply probabilistic inference..."):
        if input_mode == "Validation Benchmark (Jan–Mar 2026)":
            d_res = demand_forecaster.predict_benchmark_month(selected_month)
            s_res = supply_forecaster.predict_benchmark_month(selected_month)
            month_display = selected_month
        else:
            # Custom 3-month Demand sequence
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

            # Custom 6-month Supply sequence
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

        gap_res = perform_gap_analysis(d_res, s_res)
        risk_res = assess_risk(gap_res["predicted_gap"], gap_res["condition"])


    # =========================================================================
    # LEVEL 1 & 2: EXECUTIVE UNCERTAINTY-FIRST KPI DECK
    # =========================================================================
    st.markdown("### 📊 Operational Forecast & Uncertainty Deck")
    
    d_pi = d_res["intervals"][ci_key]
    s_pi = s_res["intervals"][ci_key]
    g_pi = gap_res["gap_intervals"][ci_key]

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    # 1. Demand KPI Card
    with kpi_col1:
        render_html(
            f"""
            <div class="kpi-card">
            <div class="kpi-tag tag-demand">⚡ ELECTRICITY DEMAND</div>
            <div class="kpi-hero-label">{ci_label} Prediction Interval (Dominant)</div>
            <div class="kpi-hero-value hero-demand">[{d_pi['lower']:,.2f} – {d_pi['upper']:,.2f}] <span style="font-size:0.9rem;color:#94A3B8">MU</span></div>
            <div class="range-bar-box">
            <div class="range-line-container">
            <span>{d_pi['lower']:,.0f}</span>
            <div class="range-track range-fill-demand"><div class="range-marker"></div></div>
            <span>{d_pi['upper']:,.0f}</span>
            </div>
            </div>
            <div class="kpi-subrow">
            <span class="subrow-label">Point Forecast (LSTM):</span>
            <span class="subrow-val">{d_res['predicted_demand']:,.2f} MU</span>
            </div>
            <div class="kpi-subrow">
            <span class="subrow-label">Uncertainty Margin:</span>
            <span class="subrow-val">±{d_pi['margin']:,.2f} MU (σ_D = {d_res['sigma']:.2f})</span>
            </div>
            </div>
            """
        )

    # 2. Supply KPI Card
    with kpi_col2:
        render_html(
            f"""
            <div class="kpi-card">
            <div class="kpi-tag tag-supply">🌱 TOTAL GENERATION (SUPPLY)</div>
            <div class="kpi-hero-label">{ci_label} Prediction Interval (Dominant)</div>
            <div class="kpi-hero-value hero-supply">[{s_pi['lower']:,.2f} – {s_pi['upper']:,.2f}] <span style="font-size:0.9rem;color:#94A3B8">MU</span></div>
            <div class="range-bar-box">
            <div class="range-line-container">
            <span>{s_pi['lower']:,.0f}</span>
            <div class="range-track range-fill-supply"><div class="range-marker"></div></div>
            <span>{s_pi['upper']:,.0f}</span>
            </div>
            </div>
            <div class="kpi-subrow">
            <span class="subrow-label">Point Forecast (LSTM):</span>
            <span class="subrow-val">{s_res['predicted_supply']:,.2f} MU</span>
            </div>
            <div class="kpi-subrow">
            <span class="subrow-label">Uncertainty Margin:</span>
            <span class="subrow-val">±{s_pi['margin']:,.2f} MU (σ_S = {s_res['sigma']:.2f})</span>
            </div>
            </div>
            """
        )

    # 3. Gap & Risk KPI Card (Central Analytical KPI)
    with kpi_col3:
        gap_val = gap_res["predicted_gap"]
        risk_lvl = risk_res["risk_level"].upper()
        risk_badge_cls = "risk-low" if risk_lvl == "LOW" else ("risk-mod" if risk_lvl == "MODERATE" else "risk-high")
        cond_label = gap_res["condition"].upper()
        
        render_html(
            f"""
            <div class="kpi-card gap-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
            <span class="kpi-tag tag-gap">⚖️ NET DEMAND–SUPPLY GAP</span>
            <span class="risk-badge {risk_badge_cls}">{risk_lvl} RISK</span>
            </div>
            <div class="kpi-hero-label">{ci_label} Gap Prediction Interval (Dominant)</div>
            <div class="kpi-hero-value hero-gap">[{g_pi['lower']:,.2f} – {g_pi['upper']:,.2f}] <span style="font-size:0.9rem;color:#94A3B8">MU</span></div>
            <div class="range-bar-box">
            <div class="range-line-container">
            <span>{g_pi['lower']:,.0f}</span>
            <div class="range-track range-fill-gap"><div class="range-marker"></div></div>
            <span>{g_pi['upper']:,.0f}</span>
            </div>
            </div>
            <div class="kpi-subrow">
            <span class="subrow-label">Expected Net Deficit (Gap):</span>
            <span class="subrow-val" style="color:#EF4444">+{gap_val:,.2f} MU ({cond_label})</span>
            </div>
            <div class="kpi-subrow">
            <span class="subrow-label">Combined Standard Error:</span>
            <span class="subrow-val">σ_Gap = {gap_res['gap_sigma']:,.2f} MU</span>
            </div>
            </div>
            """
        )

    # Summary Alert Banner
    render_html(
        f"""
        <div class="summary-alert-box">
        🎯 <b>Operational Insight ({month_display}):</b> 
        The system forecasts a net <b>{cond_label.lower()}</b> of <b>{gap_val:,.2f} MU</b> with a 
        <b>{ci_label} confidence boundary of [{g_pi['lower']:,.2f} to {g_pi['upper']:,.2f}] MU</b> (Margin: ±{g_pi['margin']:,.2f} MU).
        Classified as <b>{risk_lvl} RISK</b> under project deterministic thresholds. 
        {risk_res['description']}
        </div>
        """
    )


    # =========================================================================
    # MAIN NAVIGATION TABS
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Probabilistic Forecast & 95% Intervals",
        "📊 Out-of-Sample Validation Benchmarks",
        "🧠 Model Architecture & Comparison",
        "🏛️ RAG-Based Energy Planning",
        "📐 Statistical Methodology",
    ])


    # -------------------------------------------------------------------------
    # TAB 1: PROBABILISTIC FORECAST & UNCERTAINTY VISUALIZATION
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("#### 📈 Interactive Probabilistic Visualizations")
        
        c_left, c_right = st.columns([1.1, 1.0])

        with c_left:
            fig_comp = create_forecast_comparison_chart(
                d_res["predicted_demand"],
                s_res["predicted_supply"],
                gap_res["predicted_gap"],
                d_intervals=d_res["intervals"],
                s_intervals=s_res["intervals"],
                gap_intervals=gap_res["gap_intervals"],
                month_label=month_display,
                selected_ci=ci_label,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with c_right:
            fig_dist = create_gap_distribution_chart(
                gap_res["predicted_gap"],
                gap_res["gap_sigma"],
                gap_res["gap_intervals"],
                selected_ci=ci_label,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📐 Dedicated Prediction Interval Ranges (Multi-Tier Inspection)")

        d_col, s_col = st.columns(2)

        with d_col:
            st.markdown("##### 🔵 Demand Prediction Intervals")
            fig_d = create_prediction_interval_chart(
                "Demand Forecast",
                d_res["intervals"],
                theme_color=COLOR_DEMAND,
                selected_ci=ci_label,
            )
            st.plotly_chart(fig_d, use_container_width=True)

            d_tbl = [
                {"Confidence Tier": "Point Forecast", "Lower Bound": "—", "Upper Bound": "—", "Margin": "—", "Forecast Span": f"{d_res['predicted_demand']:,.2f} MU"},
                {"Confidence Tier": "90% Interval (z=1.645)", "Lower Bound": f"{d_res['intervals']['pi_90']['lower']:,.2f}", "Upper Bound": f"{d_res['intervals']['pi_90']['upper']:,.2f}", "Margin": f"±{d_res['intervals']['pi_90']['margin']:,.2f}", "Forecast Span": f"[{d_res['intervals']['pi_90']['lower']:,.2f}, {d_res['intervals']['pi_90']['upper']:,.2f}]"},
                {"Confidence Tier": "🎯 95% Interval (z=1.960)", "Lower Bound": f"{d_res['intervals']['pi_95']['lower']:,.2f}", "Upper Bound": f"{d_res['intervals']['pi_95']['upper']:,.2f}", "Margin": f"±{d_res['intervals']['pi_95']['margin']:,.2f}", "Forecast Span": f"[{d_res['intervals']['pi_95']['lower']:,.2f}, {d_res['intervals']['pi_95']['upper']:,.2f}]"},
                {"Confidence Tier": "99% Interval (z=2.576)", "Lower Bound": f"{d_res['intervals']['pi_99']['lower']:,.2f}", "Upper Bound": f"{d_res['intervals']['pi_99']['upper']:,.2f}", "Margin": f"±{d_res['intervals']['pi_99']['margin']:,.2f}", "Forecast Span": f"[{d_res['intervals']['pi_99']['lower']:,.2f}, {d_res['intervals']['pi_99']['upper']:,.2f}]"},
            ]
            st.dataframe(pd.DataFrame(d_tbl), hide_index=True, use_container_width=True)

        with s_col:
            st.markdown("##### 🟢 Supply Prediction Intervals")
            fig_s = create_prediction_interval_chart(
                "Supply Forecast",
                s_res["intervals"],
                theme_color=COLOR_SUPPLY,
                selected_ci=ci_label,
            )
            st.plotly_chart(fig_s, use_container_width=True)

            s_tbl = [
                {"Confidence Tier": "Point Forecast", "Lower Bound": "—", "Upper Bound": "—", "Margin": "—", "Forecast Span": f"{s_res['predicted_supply']:,.2f} MU"},
                {"Confidence Tier": "90% Interval (z=1.645)", "Lower Bound": f"{s_res['intervals']['pi_90']['lower']:,.2f}", "Upper Bound": f"{s_res['intervals']['pi_90']['upper']:,.2f}", "Margin": f"±{s_res['intervals']['pi_90']['margin']:,.2f}", "Forecast Span": f"[{s_res['intervals']['pi_90']['lower']:,.2f}, {s_res['intervals']['pi_90']['upper']:,.2f}]"},
                {"Confidence Tier": "🎯 95% Interval (z=1.960)", "Lower Bound": f"{s_res['intervals']['pi_95']['lower']:,.2f}", "Upper Bound": f"{s_res['intervals']['pi_95']['upper']:,.2f}", "Margin": f"±{s_res['intervals']['pi_95']['margin']:,.2f}", "Forecast Span": f"[{s_res['intervals']['pi_95']['lower']:,.2f}, {s_res['intervals']['pi_95']['upper']:,.2f}]"},
                {"Confidence Tier": "99% Interval (z=2.576)", "Lower Bound": f"{s_res['intervals']['pi_99']['lower']:,.2f}", "Upper Bound": f"{s_res['intervals']['pi_99']['upper']:,.2f}", "Margin": f"±{s_res['intervals']['pi_99']['margin']:,.2f}", "Forecast Span": f"[{s_res['intervals']['pi_99']['lower']:,.2f}, {s_res['intervals']['pi_99']['upper']:,.2f}]"},
            ]
            st.dataframe(pd.DataFrame(s_tbl), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🛡️ Risk Gauge & Threshold Analysis")
        
        rg_c1, rg_c2 = st.columns([1, 1.2])
        with rg_c1:
            fig_g = create_risk_gauge(gap_res["predicted_gap"], risk_res["risk_level"])
            st.plotly_chart(fig_g, use_container_width=True)

        with rg_c2:
            st.markdown("##### 📋 Project-Defined Risk Decision Boundaries")
            st.markdown(
                """
                | Risk Classification | Demand–Supply Gap Threshold | Grid Operating Directive |
                | :--- | :--- | :--- |
                | **🟢 Low Risk** | $\\text{Gap} < 3,000\\text{ MU}$ | Manageable via thermal spinning reserves & bilateral procurement. |
                | **🟡 Moderate Risk** | $3,000\\text{ MU} \\le \\text{Gap} \\le 4,500\\text{ MU}$ | Peaking plant dispatch, DAM exchange purchase, demand response. |
                | **🔴 High Risk** | $\\text{Gap} > 4,500\\text{ MU}$ | Emergency power wheeling, BESS full discharge, graded load curtailment. |
                """
            )
            st.caption("ℹ️ Deterministic decision layer based on SLDC capacity margins (NOT generated by LLM).")


    # -------------------------------------------------------------------------
    # TAB 2: OUT-OF-SAMPLE VALIDATION BENCHMARKS
    # -------------------------------------------------------------------------
    with tab2:
        st.header("📊 Out-of-Sample Hold-Out Validation (Jan – Mar 2026)")
        st.markdown(
            "Empirical evaluation of model point predictions against unseen, hold-out ground-truth records "
            "for the first quarter of 2026."
        )

        val_rows = []
        for m in ["Jan 2026", "Feb 2026", "Mar 2026"]:
            b = config.VALIDATION_BENCHMARKS[m]
            d_err = abs(b["demand_actual"] - b["demand_predicted"])
            d_ape = (d_err / b["demand_actual"]) * 100
            s_err = abs(b["supply_actual"] - b["supply_predicted"])
            s_ape = (s_err / b["supply_actual"]) * 100

            # 95% PI for gap
            z = config.Z_95
            g_low = b["gap_predicted"] - z * config.BASE_SIGMA_GAP
            g_high = b["gap_predicted"] + z * config.BASE_SIGMA_GAP

            val_rows.append({
                "Month": m,
                "Actual Demand (MU)": f"{b['demand_actual']:,.2f}",
                "Predicted Demand (MU)": f"{b['demand_predicted']:,.2f}",
                "Demand Error": f"{d_err:,.2f} MU ({d_ape:.2f}%)",
                "Actual Supply (MU)": f"{b['supply_actual']:,.2f}",
                "Predicted Supply (MU)": f"{b['supply_predicted']:,.2f}",
                "Supply Error": f"{s_err:,.2f} MU ({s_ape:.2f}%)",
                "Actual Gap (MU)": f"{b['gap_actual']:,.2f}",
                "Predicted Gap (MU)": f"{b['gap_predicted']:,.2f}",
                "95% Gap PI [Lower, Upper]": f"[{g_low:,.2f}, {g_high:,.2f}] MU",
                "Condition": "Shortage",
            })

        st.dataframe(pd.DataFrame(val_rows), hide_index=True, use_container_width=True)

        st.markdown("---")
        vc_left, vc_right = st.columns(2)
        with vc_left:
            st.plotly_chart(create_validation_comparison_chart(), use_container_width=True)
        with vc_right:
            st.plotly_chart(create_multi_month_timeline_chart(), use_container_width=True)

        st.info(
            "📌 **Data Integrity Note:** Validation benchmarks represent true out-of-sample forward testing. "
            "They are completely held out from model training and hyperparameter selection."
        )


    # -------------------------------------------------------------------------
    # TAB 3: MODEL ARCHITECTURE & COMPARISON
    # -------------------------------------------------------------------------
    with tab3:
        st.header("🧠 Model Development & Comparative Benchmarks")
        st.markdown(
            "Comprehensive evaluation of candidate machine learning algorithms and deep recurrent architectures "
            "trained on chronological time-series splits (129 monthly records: Apr 2015 – Dec 2025)."
        )

        m_col1, m_col2 = st.columns(2)

        with m_col1:
            st.subheader("🔵 Demand Forecasting Model Comparison")
            d_perf = pd.DataFrame([
                {"Model": "Random Forest", "MAE (MU)": 741.63, "RMSE (MU)": 909.69, "MAPE (%)": "6.56%", "R² Score": 0.31, "Status": "Baseline"},
                {"Model": "XGBoost", "MAE (MU)": 619.18, "RMSE (MU)": 724.51, "MAPE (%)": "5.56%", "R² Score": 0.34, "Status": "Baseline"},
                {"Model": "LightGBM", "MAE (MU)": 634.18, "RMSE (MU)": 748.24, "MAPE (%)": "5.66%", "R² Score": 0.29, "Status": "Baseline"},
                {"Model": "Stacked LSTM", "MAE (MU)": 368.72, "RMSE (MU)": 412.93, "MAPE (%)": "3.41%", "R² Score": 0.78, "Status": "⭐ Selected Final Model"},
            ])
            st.dataframe(d_perf, hide_index=True, use_container_width=True)

            render_html(
                """
                <div class="formula-card">
                <b>Demand LSTM Architecture Blueprint:</b><br>
                • <b>Lookback Sequence:</b> 3 months (3 × 14 feature matrix)<br>
                • <b>Input Features:</b> Weather (Temp, Rain, Humid, Solar), Calendar, Lags 1-3, Rolling 3/6/12<br>
                • <b>Recurrent Layers:</b> Layer 1 (64/96 units, tanh) → Layer 2 (32 units, tanh)<br>
                • <b>Residual Standard Error:</b> σ_D = 391.30 MU (Validation standard deviation)
                </div>
                """
            )

        with m_col2:
            st.subheader("🟢 Supply Forecasting Model Comparison")
            s_perf = pd.DataFrame([
                {"Model": "Random Forest", "MAE (MU)": 823.30, "RMSE (MU)": 1220.90, "MAPE (%)": "6.89%", "R² Score": 0.31, "Status": "Baseline"},
                {"Model": "XGBoost", "MAE (MU)": 847.76, "RMSE (MU)": 1215.30, "MAPE (%)": "7.13%", "R² Score": 0.31, "Status": "Baseline"},
                {"Model": "LightGBM", "MAE (MU)": 802.74, "RMSE (MU)": 1157.40, "MAPE (%)": "6.74%", "R² Score": 0.38, "Status": "Baseline"},
                {"Model": "Tuned LSTM", "MAE (MU)": 731.56, "RMSE (MU)": 973.93, "MAPE (%)": "6.30%", "R² Score": 0.56, "Status": "⭐ Selected Final Model"},
            ])
            st.dataframe(s_perf, hide_index=True, use_container_width=True)

            render_html(
                """
                <div class="formula-card">
                <b>Supply LSTM Architecture Blueprint:</b><br>
                • <b>Lookback Sequence:</b> 6 months (6 × 18 feature matrix)<br>
                • <b>Input Features:</b> 8 Source Lags (Coal, Hydro, Nuclear, Solar, Wind, etc.), Rolling 3/6/12<br>
                • <b>Recurrent Layers:</b> Layer 1 (64 units, tanh) → Layer 2 (32 units, tanh)<br>
                • <b>Residual Standard Error:</b> σ_S = 996.83 MU (Validation standard deviation)
                </div>
                """
            )


    # -------------------------------------------------------------------------
    # TAB 4: RAG-BASED ENERGY PLANNING
    # -------------------------------------------------------------------------
    with tab4:
        st.header("⚡ Energy Recommendations")
        st.markdown(
            f"**Forecast Month:** `{month_display}` | **Demand-Supply Shortage:** `{gap_res['predicted_gap']:,.2f} MU` | **Risk Level:** `{risk_res['risk_level']} Risk`"
        )
        st.markdown("---")

        with st.spinner("Retrieving knowledge base evidence & generating recommendations..."):
            rag_res = generate_energy_planning_recommendation(gap_res, risk_res, month_name=month_display)

        # Display ONLY the 5 simple recommendation points
        rec_text = rag_res.get("recommendation_text", "")
        if rec_text:
            st.markdown(rec_text)

        st.markdown("---")

        # Optional Collapsible Knowledge Base Chunks
        retrieved_chunks = rag_res.get("retrieved_chunks", [])
        if retrieved_chunks:
            with st.expander("📚 Supporting Document Evidence"):
                for chunk in retrieved_chunks:
                    rank = chunk.get("rank", 1)
                    source = chunk.get("source", "Document.pdf")
                    page = chunk.get("page", "N/A")
                    text = chunk.get("text", "")
                    st.markdown(f"**Point #{rank} Source:** {source} (Page {page})")
                    st.caption(text[:200] + "...")
                    st.markdown("---")


    # -------------------------------------------------------------------------
    # TAB 5: STATISTICAL METHODOLOGY
    # -------------------------------------------------------------------------
    with tab5:
        st.header("📐 Statistical & Mathematical Methodology")
        st.markdown(
            "Mathematical formulations implemented in the confidence-interval and gap uncertainty engine."
        )

        st.markdown(
            r"""
            ### 1. Residual Standard Error ($\sigma_e$)
            $$\sigma_e = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} \left( y_{\text{act}, i} - \hat{y}_{\text{pred}, i} \right)^2}$$
            - **Demand Standard Error ($\sigma_D$):** `391.30 MU`
            - **Supply Standard Error ($\sigma_S$):** `996.83 MU`

            ### 2. Combined Demand–Supply Gap Standard Error ($\sigma_{\text{Gap}}$)
            Assuming orthogonal forecasting error distributions:
            $$\sigma_{\text{Gap}} = \sqrt{\sigma_D^2 + \sigma_S^2} = \sqrt{(391.30)^2 + (996.83)^2} = 1070.88\text{ MU}$$

            ### 3. Symmetric Prediction Intervals ($\text{PI}_{1-\alpha}$)
            $$\text{PI}_{1-\alpha} = \left[ \hat{y} - z_{1-\alpha/2} \cdot \sigma_e, \quad \hat{y} + z_{1-\alpha/2} \cdot \sigma_e \right]$$
            - **90% Interval ($z_{0.95} = 1.645$):** $\hat{y} \pm 1.645 \cdot \sigma_e$
            - **95% Interval ($z_{0.975} = 1.960$):** $\hat{y} \pm 1.960 \cdot \sigma_e$ *(IEEE Project Benchmark)*
            - **99% Interval ($z_{0.995} = 2.576$):** $\hat{y} \pm 2.576 \cdot \sigma_e$

            ### 4. Deterministic Risk Classification
            - $\text{Gap} < 3000\text{ MU} \implies \text{Low Risk}$
            - $3000\text{ MU} \le \text{Gap} \le 4500\text{ MU} \implies \text{Moderate Risk}$
            - $\text{Gap} > 4500\text{ MU} \implies \text{High Risk}$
            """
        )
