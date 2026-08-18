# ⚡ Electricity Demand–Supply Gap Forecasting & RAG Energy Planning

**Title:** *A Confidence-Driven Probabilistic Machine Learning Framework for Electricity Demand–Supply Gap Forecasting and RAG-Based Energy Planning*

A professional, modular Streamlit web application providing independent probabilistic forecasting for electricity demand and electricity supply, statistical uncertainty quantification (90%, 95%, and 99% prediction intervals), deterministic shortage/surplus risk classification, out-of-sample historical validation, and RAG-grounded energy planning policy recommendations.

---

## 🏗️ Architecture & Features

1. **Independent Demand & Supply Forecasting Pipelines:**
   - **Demand Forecasting Module:** Loads `Demand_LSTM_Final.keras` with 3-month lookback (14 features: weather, calendar, historical demand lags, rolling averages).
   - **Supply Forecasting Module:** Loads `Supply_LSTM.keras` with 6-month lookback (18 features: fuel-specific generation lags for Coal, Nuclear, Hydro, Solar, Wind, Small Hydro, Bio-power, Oil/Gas, total lags, rolling averages).
   - Zero cross-dependencies between demand and supply.

2. **Statistical Uncertainty & Confidence Intervals:**
   - Evaluated on validation residuals up to December 2025:
     - Demand Standard Error ($\sigma_D$): `391.30 MU`
     - Supply Standard Error ($\sigma_S$): `996.83 MU`
     - Combined Gap Standard Error ($\sigma_{\text{Gap}} = \sqrt{\sigma_D^2 + \sigma_S^2}$): `1070.88 MU`
   - Generates two-tailed 90% ($z=1.645$), 95% ($z=1.960$), and 99% ($z=2.576$) prediction intervals.

3. **Demand–Supply Gap Analysis & Deterministic Risk Classification:**
   - $\text{Gap} = \text{Predicted Demand} - \text{Predicted Supply}$
   - Condition: $\text{Gap} > 0 \implies \text{Shortage}$, $\text{Gap} < 0 \implies \text{Surplus}$.
   - **Deterministic Risk Thresholds (Project Decision Layer):**
     - $\text{Gap} < 3000\text{ MU} \implies \text{Low Risk}$
     - $3000\text{ MU} \le \text{Gap} \le 4500\text{ MU} \implies \text{Moderate Risk}$
     - $\text{Gap} > 4500\text{ MU} \implies \text{High Risk}$

4. **Out-of-Sample Validation Benchmarks (Jan–Mar 2026):**
   - January 2026: Demand (Pred: 11,047.51 MU, Act: 10,067.00 MU) | Supply (Pred: 8,809.63 MU, Act: 10,189.56 MU) | Gap (Pred: 2,237.88 MU, Act: -122.56 MU)
   - February 2026: Demand (Pred: 12,308.25 MU, Act: 10,125.00 MU) | Supply (Pred: 9,229.95 MU, Act: 10,405.60 MU) | Gap (Pred: 3,078.30 MU, Act: -280.60 MU)
   - March 2026: Demand (Pred: 12,594.89 MU, Act: 12,233.00 MU) | Supply (Pred: 9,574.64 MU, Act: 11,247.51 MU) | Gap (Pred: 3,020.25 MU, Act: 985.49 MU)

5. **Model Development Comparison:**
   - Transparently displays development comparison $R^2$ scores:
     - **Demand:** Random Forest (0.72), XGBoost (0.86), LightGBM (0.78), LSTM (0.88)
     - **Supply:** Random Forest (0.74), XGBoost (0.88), LightGBM (0.93), LSTM (0.90)

6. **RAG-Based Energy Planning Policy Engine:**
   - Evidence-grounded policy actions across 5 domains (Thermal Fleet Modulation, BESS / Storage, Renewable Integration, Demand Response, Power Exchange Trading) grounded in CEA and TNERC guidelines.

---

## 📁 Directory Structure

```
electricity_forecasting_webapp/
├── app.py                     # Main Streamlit Dashboard Application
├── config.py                  # Global configurations, file paths, parameters
├── requirements.txt           # Python dependency requirements
├── README.md                  # Application documentation and setup guide
│
├── services/
│   ├── __init__.py
│   ├── demand_forecasting.py  # Independent Demand forecasting service
│   ├── supply_forecasting.py  # Independent Supply forecasting service
│   ├── prediction_intervals.py# Statistical sigma & (90%, 95%, 99%) intervals
│   ├── gap_analysis.py        # Gap calculation & bounds computation
│   ├── risk_assessment.py     # Deterministic risk classification
│   └── rag_planning.py        # RAG energy planning recommendation engine
│
└── utils/
    ├── __init__.py
    ├── preprocessing.py       # Sequence engineering & dataset loaders
    └── visualization.py       # Interactive Plotly dashboard charts
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.10+ (or an Anaconda environment with TensorFlow installed).

### 2. Install Dependencies
Navigate to the web application directory and install required packages:

```bash
cd "d:/Final Year Project 2027/electricity_forecasting_webapp"
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
Run the dashboard:

```bash
streamlit run app.py
```

If using Anaconda on Windows:

```bash
C:\Users\TAMILARASU\anaconda3\python.exe -m streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.
