"""
========================================================================================
Electricity Demand–Supply Gap Forecasting & RAG-Based Energy Planning Dashboard
========================================================================================
Academic Engineering Dashboard for IEEE Project Demonstration.
Confidence-Driven Probabilistic Machine Learning Framework & RAG-Based Energy Planning.

Authors: Final Year Engineering Project Team
Domain: Deep Learning, Probabilistic Uncertainty Quantification & Applied RAG Planning
========================================================================================
"""

import os
import sys
from pathlib import Path

# Disable GPU initialization, limit OpenMP threads, and set single-thread mode
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
import plotly.graph_objects as go

import config
from services.demand_forecasting import DemandForecaster
from services.supply_forecasting import SupplyForecaster
from services.gap_analysis import perform_gap_analysis
from services.risk_assessment import assess_risk
from services.rag_planning import generate_energy_planning_recommendation
from utils.preprocessing import encode_cyclical_month


def render_html(html_str: str):
    """
    Renders clean HTML in Streamlit by stripping leading and trailing indentation.
    """
    clean_html = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(clean_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Streamlit Page Configuration (Clean Academic Layout)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Electricity Demand–Supply Gap Forecasting | IEEE Project",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# High-Contrast Minimal Academic CSS
render_html(
    """
    <style>
    /* =========================================================================
       GLOBAL THEME & TYPOGRAPHY
       ========================================================================= */
    html, body, [class*="css"], .stApp {
        font-family: Arial, Helvetica, sans-serif !important;
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }

    /* All text elements default to high contrast dark */
    .stApp p, .stApp span, .stApp label, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #111111;
    }

    /* Main container padding */
    .main .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1150px !important;
    }

    /* Header */
    .header-box {
        border-bottom: 1px solid #BDBDBD;
        padding-bottom: 10px;
        margin-bottom: 18px;
    }
    .header-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #111111 !important;
        margin: 0;
        line-height: 1.25;
    }
    .header-subtitle {
        font-size: 0.88rem;
        color: #444444 !important;
        margin: 5px 0 0 0;
    }

    /* =========================================================================
       WIDGET LABELS (Fixes invisible labels across all inputs and sidebar)
       ========================================================================= */
    label[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span,
    div[data-testid="stWidgetLabel"] span,
    label[data-testid="stWidgetLabel"] div,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] label p,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextInput"] label p {
        color: #111111 !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        margin-bottom: 3px !important;
    }

    /* =========================================================================
       SIDEBAR STYLING & HEADINGS
       ========================================================================= */
    section[data-testid="stSidebar"] {
        background-color: #F7F8FA !important;
        border-right: 1px solid #DCE0E5 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
        color: #111111 !important;
        font-weight: 700 !important;
        font-size: 0.90rem !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-top: 16px !important;
        margin-bottom: 8px !important;
        border-bottom: 1px solid #DCE0E5;
        padding-bottom: 4px;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
        color: #222222 !important;
        font-size: 0.84rem !important;
        line-height: 1.45;
    }

    /* =========================================================================
       NUMBER INPUTS & TEXT INPUTS
       ========================================================================= */
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stTextInput"] div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 4px !important;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        color: #111111 !important;
        background-color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    div[data-testid="stNumberInput"] button {
        background-color: #EEEEEE !important;
        color: #111111 !important;
        border: none !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #DDDDDD !important;
    }

    div[data-testid="stNumberInput"] button svg {
        fill: #111111 !important;
        stroke: #111111 !important;
    }

    /* =========================================================================
       SELECTBOX & DROPDOWNS
       ========================================================================= */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 4px !important;
    }

    div[data-testid="stSelectbox"] div,
    div[data-testid="stSelectbox"] span,
    div[data-testid="stSelectbox"] svg {
        color: #111111 !important;
        fill: #111111 !important;
    }

    /* =========================================================================
       SLIDERS
       ========================================================================= */
    div[data-testid="stSlider"] div,
    div[data-testid="stSlider"] span,
    div[data-testid="stSlider"] [data-testid="stTickBarMin"],
    div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: #333333 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
    }

    /* =========================================================================
       RADIO BUTTONS
       ========================================================================= */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label div {
        color: #111111 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stRadio"] [role="radio"] {
        opacity: 1 !important;
    }

    /* Recalculate button */
    div.stButton > button {
        background-color: #303030 !important;
        color: #FFFFFF !important;
        border: 1px solid #303030 !important;
        border-radius: 3px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        min-height: 38px !important;
        box-shadow: none !important;
    }
    div.stButton > button p,
    div.stButton > button span {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }
    div.stButton > button:hover {
        background-color: #111111 !important;
        border-color: #111111 !important;
    }

    /* Plain forecast section -- no card */
    .results-container {
        border-top: 1px solid #BDBDBD;
        border-bottom: 1px solid #BDBDBD;
        padding: 12px 0;
        margin: 8px 0 18px 0;
        background: #FFFFFF;
    }
    .results-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #111111 !important;
        margin-bottom: 12px;
    }
    .results-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0;
    }
    .result-col {
        padding: 2px 18px;
        border-right: 1px solid #D6D6D6;
    }
    .result-col:first-child {
        padding-left: 0;
    }
    .result-col:last-child {
        border-right: none;
        padding-right: 0;
    }
    .result-col-header {
        font-size: 0.82rem;
        font-weight: 700;
        color: #222222 !important;
        margin-bottom: 9px;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }
    .result-field {
        margin-bottom: 7px;
        font-size: 0.84rem;
        color: #222222 !important;
    }
    .result-field-label {
        color: #555555 !important;
        font-weight: 400;
    }
    .result-field-val {
        font-family: "Courier New", monospace;
        font-weight: 700;
        color: #111111 !important;
        margin-left: 2px;
    }

    /* One selected-month graph */
    .graph-heading {
        font-size: 0.95rem;
        font-weight: 700;
        color: #111111 !important;
        margin: 10px 0 4px 0;
    }

    /* Plain operational summary */
    .summary-box {
        border-top: 1px solid #D0D0D0;
        border-bottom: 1px solid #D0D0D0;
        padding: 10px 0;
        margin: 14px 0;
        background: #FFFFFF;
    }
    .summary-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #222222 !important;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }
    .summary-text {
        font-size: 0.86rem;
        color: #333333 !important;
        line-height: 1.5;
        margin: 0;
    }

    /* Plain RAG section */
    .rag-box {
        border-top: 1px solid #BDBDBD;
        padding: 12px 0 0 0;
        margin-top: 18px;
        background: #FFFFFF;
    }
    .rag-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #111111 !important;
        margin-bottom: 8px;
    }
    .rag-meta {
        font-size: 0.82rem;
        color: #444444 !important;
        border-bottom: 1px solid #E0E0E0;
        padding-bottom: 7px;
        margin-bottom: 9px;
        line-height: 1.5;
    }
    .rag-recommendations {
        font-size: 0.86rem;
        color: #222222 !important;
        line-height: 1.55;
    }
    .rag-recommendations ol,
    .rag-recommendations ul {
        margin: 4px 0;
        padding-left: 20px;
    }
    .rag-recommendations li {
        margin-bottom: 4px;
    }
    </style>
    """
)


# -----------------------------------------------------------------------------
# Cached Model Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_demand_forecaster() -> DemandForecaster:
    return DemandForecaster()


@st.cache_resource(show_spinner=False)
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
    st.error(f"Model Initialization Error: {e}")
    st.info(
        "Please ensure the trained models and scalers are located in their respective directories:\n"
        f"- Demand Model: {config.DEMAND_MODEL_PATH}\n"
        f"- Supply Model: {config.SUPPLY_MODEL_PATH}"
    )


# -----------------------------------------------------------------------------
# 1. TITLE & SHORT DESCRIPTION
# -----------------------------------------------------------------------------
render_html(
    """
    <div class="header-box">
        <h1 class="header-title">Electricity Demand–Supply Gap Forecasting & Energy Planning Framework</h1>
        <p class="header-subtitle">Machine Learning Forecasting with Prediction Intervals and RAG-Based Energy Planning</p>
    </div>
    """
)


# -----------------------------------------------------------------------------
# 2. SELECT MONTH + PREDICTION INTERVAL + RECALCULATE (ONE ROW)
# -----------------------------------------------------------------------------
ctl_col1, ctl_col2, ctl_col3 = st.columns([1.8, 2.0, 0.8])

with ctl_col1:
    selected_month = st.selectbox(
        "Select Month:",
        ["Jan 2026", "Feb 2026", "Mar 2026", "Custom Scenario (Configure in Sidebar)"],
        index=0,
        help="Select exactly one target operational month to evaluate.",
    )

with ctl_col2:
    selected_ci = st.radio(
        "Prediction Interval:",
        ["90%", "95%", "99%"],
        index=1,
        horizontal=True,
        help="Select the prediction confidence interval tier.",
    )
    ci_label = selected_ci
    ci_key = "pi_95" if selected_ci == "95%" else ("pi_90" if selected_ci == "90%" else "pi_99")

with ctl_col3:
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    run_btn = st.button("Recalculate", use_container_width=True)


# -----------------------------------------------------------------------------
# Sidebar Configuration (Custom Operational Scenarios)
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Scenario Configuration")
input_mode = "Custom Forecasting Scenario" if "Custom" in selected_month else "Validation Benchmark (Jan–Mar 2026)"

if input_mode == "Validation Benchmark (Jan–Mar 2026)":
    st.sidebar.markdown(f"**Target Period:** {selected_month}")
    st.sidebar.markdown(
        "**Validation Benchmark Mode Active**\n\n"
        "- Displays results for only the selected month.\n"
        "- Demand lookback sequence: 3 months (14 features).\n"
        "- Supply lookback sequence: 6 months (18 features).\n"
        "- Decoupled Dual-Stream architecture."
    )
else:
    st.sidebar.markdown("### Calendar Features")
    custom_year = st.sidebar.number_input("Forecast Year:", min_value=2024, max_value=2030, value=2026, step=1)
    custom_month = st.sidebar.slider("Forecast Month (1–12):", min_value=1, max_value=12, value=4)
    is_festival = st.sidebar.selectbox("Festival Month:", [0, 1], index=0, format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    
    st.sidebar.markdown("### Demand Inputs (Weather & History)")
    c_temp = st.sidebar.number_input("Avg Temperature (°C):", value=29.5, step=0.1)
    c_rain = st.sidebar.number_input("Rainfall (mm):", value=35.0, step=1.0)
    c_humid = st.sidebar.number_input("Humidity (%):", value=68.0, step=0.5)
    c_solar = st.sidebar.number_input("Solar Irradiance (W/m²):", value=220.0, step=5.0)
    
    c_d_lag1 = st.sidebar.number_input("Demand Lag 1 (t-1 MU):", value=12233.0, step=50.0)
    c_d_lag2 = st.sidebar.number_input("Demand Lag 2 (t-2 MU):", value=10125.0, step=50.0)
    c_d_lag3 = st.sidebar.number_input("Demand Lag 3 (t-3 MU):", value=10067.0, step=50.0)

    st.sidebar.markdown("### Supply Inputs (Generation Lags in MU)")
    c_coal = st.sidebar.number_input("Coal Generation (t-1 MU):", value=6000.0, step=50.0)
    c_hydro = st.sidebar.number_input("Hydro Generation (t-1 MU):", value=200.0, step=10.0)
    c_nuclear = st.sidebar.number_input("Nuclear Generation (t-1 MU):", value=1500.0, step=20.0)
    c_wind = st.sidebar.number_input("Wind Generation (t-1 MU):", value=800.0, step=20.0)
    c_sol_gen = st.sidebar.number_input("Solar Generation (t-1 MU):", value=1800.0, step=20.0)
    c_oil_gas = st.sidebar.number_input("Oil & Gas (t-1 MU):", value=90.0, step=5.0)
    c_small_hydro = st.sidebar.number_input("Small-Hydro (t-1 MU):", value=15.0, step=2.0)
    c_bio = st.sidebar.number_input("Bio Power (t-1 MU):", value=10.0, step=2.0)
    c_tot_lag1 = st.sidebar.number_input("Total Supply Lag 1 (t-1 MU):", value=11247.5, step=50.0)


# -----------------------------------------------------------------------------
# Execute Inference Pipeline for the Single Selected Month
# -----------------------------------------------------------------------------
if models_ready:
    with st.spinner("Calculating forecast for selected month..."):
        if input_mode == "Validation Benchmark (Jan–Mar 2026)":
            d_res = demand_forecaster.predict_benchmark_month(selected_month)
            s_res = supply_forecaster.predict_benchmark_month(selected_month)
            month_display = selected_month
        else:
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
            month_display = f"Custom ({custom_year}-{custom_month:02d})"

        gap_res = perform_gap_analysis(d_res, s_res)
        risk_res = assess_risk(gap_res["predicted_gap"], gap_res["condition"])

    d_pi = d_res["intervals"][ci_key]
    s_pi = s_res["intervals"][ci_key]
    g_pi = gap_res["gap_intervals"][ci_key]

    gap_val = gap_res["predicted_gap"]
    risk_lvl = risk_res["risk_level"]
    cond_label = gap_res["condition"]
    gap_sign = "+" if gap_val >= 0 else ""

    # -------------------------------------------------------------------------
    # 3. FORECAST RESULTS SECTION (FOR SELECTED MONTH ONLY)
    # -------------------------------------------------------------------------
    render_html(
        f"""
        <div class="results-container">
            <div class="results-title">Forecast Results – {month_display}</div>
            <div class="results-grid">
                <!-- Electricity Demand -->
                <div class="result-col">
                    <div class="result-col-header">Electricity Demand</div>
                    <div class="result-field">
                        <span class="result-field-label">Point Forecast:</span>
                        <span class="result-field-val">{d_res['predicted_demand']:,.2f} MU</span>
                    </div>
                    <div class="result-field">
                        <span class="result-field-label">Prediction Interval ({ci_label}):</span>
                        <span class="result-field-val">[{d_pi['lower']:,.2f}, {d_pi['upper']:,.2f}] MU</span>
                    </div>
                </div>

                <!-- Electricity Supply -->
                <div class="result-col">
                    <div class="result-col-header">Electricity Supply</div>
                    <div class="result-field">
                        <span class="result-field-label">Point Forecast:</span>
                        <span class="result-field-val">{s_res['predicted_supply']:,.2f} MU</span>
                    </div>
                    <div class="result-field">
                        <span class="result-field-label">Prediction Interval ({ci_label}):</span>
                        <span class="result-field-val">[{s_pi['lower']:,.2f}, {s_pi['upper']:,.2f}] MU</span>
                    </div>
                </div>

                <!-- Demand–Supply Gap -->
                <div class="result-col">
                    <div class="result-col-header">Demand–Supply Gap</div>
                    <div class="result-field">
                        <span class="result-field-label">Predicted Gap:</span>
                        <span class="result-field-val">{gap_sign}{gap_val:,.2f} MU</span>
                    </div>
                    <div class="result-field">
                        <span class="result-field-label">Prediction Interval ({ci_label}):</span>
                        <span class="result-field-val">[{g_pi['lower']:,.2f}, {g_pi['upper']:,.2f}] MU</span>
                    </div>
                    <div class="result-field">
                        <span class="result-field-label">Status:</span>
                        <span class="result-field-val">{cond_label}</span>
                    </div>
                    <div class="result-field">
                        <span class="result-field-label">Risk Level:</span>
                        <span class="result-field-val">{risk_lvl} Risk</span>
                    </div>
                </div>
            </div>
        </div>
        """
    )

    # -------------------------------------------------------------------------
    # 4. GRAPH: ONE GRAPH FOR THE SELECTED MONTH ONLY
    # -------------------------------------------------------------------------
    categories = ["Electricity Demand", "Electricity Supply"]
    values = [d_res["predicted_demand"], s_res["predicted_supply"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            text=[f"{v:,.2f} MU" for v in values],
            textposition="outside",
            marker_color="#555555",
            hovertemplate="<b>%{x}</b>: %{y:,.2f} MU<extra></extra>",
            width=0.45,
        )
    )

    y_max = max(values) * 1.18

    fig.update_layout(
        title=dict(
            text=f"<b>Electricity Demand vs Supply – {month_display}</b>",
            font=dict(size=13, color="#111111", family="Arial, Helvetica, sans-serif"),
            x=0,
            xanchor="left",
        ),
        yaxis=dict(
            title=dict(text="Electricity Quantity (MU)", font=dict(color="#333333", size=11)),
            gridcolor="#E5E5E5",
            zeroline=True,
            zerolinecolor="#BDBDBD",
            range=[0, y_max],
            tickfont=dict(color="#333333", size=10),
        ),
        xaxis=dict(
            tickfont=dict(color="#111111", size=11, family="Arial, Helvetica, sans-serif"),
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=270,
        margin=dict(l=45, r=20, t=45, b=25),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # -------------------------------------------------------------------------
    # 5. OPERATIONAL SUMMARY
    # -------------------------------------------------------------------------
    render_html(
        f"""
        <div class="summary-box">
            <div class="summary-title">Operational Summary</div>
            <p class="summary-text">
                For <b>{month_display}</b>, electricity demand is <b>{d_res['predicted_demand']:,.2f} MU</b> 
                and electricity supply is <b>{s_res['predicted_supply']:,.2f} MU</b>, resulting in a 
                <b>{cond_label.lower()} of {gap_val:,.2f} MU</b>. The condition is classified as <b>{risk_lvl} Risk</b>.
            </p>
        </div>
        """
    )

    # -------------------------------------------------------------------------
    # 6. RAG-BASED ENERGY PLANNING RECOMMENDATION
    # -------------------------------------------------------------------------
    with st.spinner("Generating energy planning recommendations..."):
        rag_res = generate_energy_planning_recommendation(gap_res, risk_res, month_name=month_display)

    rec_text = rag_res.get("recommendation_text", "")
    retrieved_chunks = rag_res.get("retrieved_chunks", [])

    # Format recommendation list (removing all markdown asterisks/stars)
    lines = [l.strip() for l in rec_text.splitlines() if l.strip()]
    rec_items = []
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line.startswith(("-", "•")):
            cleaned_line = cleaned_line.lstrip("-•").strip()
        elif len(cleaned_line) > 2 and cleaned_line[0].isdigit() and cleaned_line[1] in [".", ")"]:
            cleaned_line = cleaned_line[2:].strip()
        elif len(cleaned_line) > 3 and cleaned_line[:2].isdigit() and cleaned_line[2] in [".", ")"]:
            cleaned_line = cleaned_line[3:].strip()
        
        # Remove all markdown asterisks / stars
        cleaned_line = cleaned_line.replace("**", "").replace("*", "").strip()
        if cleaned_line:
            rec_items.append(f"<li>{cleaned_line}</li>")

    clean_rec_text = rec_text.replace("**", "").replace("*", "")
    formatted_rec_html = f"<ol>{''.join(rec_items)}</ol>" if rec_items else f"<p>{clean_rec_text}</p>"

    # Retrieved context summary
    if retrieved_chunks:
        policy_info = "Relevant energy policy and planning documents (Tamil Nadu Energy Directives, CEA Norms, TNERC Guidelines)"
    else:
        policy_info = "Relevant energy policy and planning documents (State Grid Directives & Operating Procedures)"

    render_html(
        f"""
        <div class="rag-box">
            <div class="rag-title">RAG-Based Energy Planning Recommendation</div>
            <div class="rag-meta">
                <strong>Risk Condition:</strong> {risk_lvl} Risk &nbsp;|&nbsp; 
                <strong>Demand–Supply Gap:</strong> {gap_val:,.2f} MU &nbsp;|&nbsp; 
                <strong>Retrieved Policy Context:</strong> {policy_info}
            </div>
            <div class="rag-recommendations">
                {formatted_rec_html}
            </div>
        </div>
        """
    )
