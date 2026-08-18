"""
Configuration file for Electricity Demand–Supply Gap Forecasting & Energy Planning Web Application.
Contains file paths, feature column specifications, statistical interval constants, and risk thresholds.
"""

import os
from pathlib import Path

# Base Paths
WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent

# Model & Scaler Directories
DEMAND_MODEL_DIR = PROJECT_ROOT / "Electricity Demand and Supply Forecasting" / "Model Training" / "Demand forecasting"
DEMAND_VAL_DIR = PROJECT_ROOT / "Electricity Demand and Supply Forecasting" / "Model Training" / "Demand_Forecasting_Validation"
SUPPLY_MODEL_DIR = PROJECT_ROOT / "Electricity Demand and Supply Forecasting" / "Model Training" / "Supply Forecasting"

# Demand Model Files
DEMAND_MODEL_PATH = DEMAND_MODEL_DIR / "Demand_LSTM_Final.keras"
DEMAND_X_SCALER_PATH = DEMAND_MODEL_DIR / "Demand_LSTM_X_scaler.pkl"
DEMAND_Y_SCALER_PATH = DEMAND_MODEL_DIR / "Demand_LSTM_y_scaler.pkl"
DEMAND_X_TRAIN_PATH = DEMAND_MODEL_DIR / "Demand_X_train.pkl"
DEMAND_X_TEST_PATH = DEMAND_MODEL_DIR / "Demand_X_test.pkl"
DEMAND_Y_TEST_PATH = DEMAND_MODEL_DIR / "Demand_y_test.pkl"
DEMAND_VAL_DATA_PATH = DEMAND_VAL_DIR / "Book1.xlsx"

# Supply Model Files
SUPPLY_MODEL_PATH = SUPPLY_MODEL_DIR / "Supply_LSTM.keras"
SUPPLY_X_SCALER_PATH = SUPPLY_MODEL_DIR / "Supply_LSTM_X_scaler.pkl"
SUPPLY_Y_SCALER_PATH = SUPPLY_MODEL_DIR / "Supply_LSTM_y_scaler.pkl"
SUPPLY_X_TRAIN_PATH = SUPPLY_MODEL_DIR / "Supply_X_train.pkl"
SUPPLY_X_TEST_PATH = SUPPLY_MODEL_DIR / "Supply_X_test.pkl"
SUPPLY_Y_TEST_PATH = SUPPLY_MODEL_DIR / "Supply_y_test.pkl"
SUPPLY_DATASET_PATH = SUPPLY_MODEL_DIR / "supply_dataset.csv"

# Demand Feature Configuration (Lookback = 3 months)
DEMAND_LOOKBACK = 3
DEMAND_FEATURES = [
    "Humidity",
    "Rainfall",
    "Solar_Irradiance",
    "Temperature",
    "Year",
    "Month_sin",
    "Month_cos",
    "Festival",
    "Demand_Lag_1",
    "Demand_Lag_2",
    "Demand_Lag_3",
    "Demand_Rolling_3",
    "Demand_Rolling_6",
    "Demand_Rolling_12",
]

# Supply Feature Configuration (Lookback = 6 months)
SUPPLY_LOOKBACK = 6
SUPPLY_FEATURES = [
    "Coal_Lag_1",
    "Oil_Gas_Lag_1",
    "Nuclear_Lag_1",
    "Hydro_Lag_1",
    "Solar_Lag_1",
    "Wind_Lag_1",
    "Small_Hydro_Lag_1",
    "Bio_Power_Lag_1",
    "Year",
    "Month",
    "Quarter",
    "Total_Lag_1",
    "Total_Lag_3",
    "Total_Lag_6",
    "Total_Lag_12",
    "Total_Rolling_3",
    "Total_Rolling_6",
    "Total_Rolling_12",
]

# Standard Normal Multipliers (Two-tailed confidence intervals)
# 90% confidence -> alpha=0.10 -> z_0.95
# 95% confidence -> alpha=0.05 -> z_0.975
# 99% confidence -> alpha=0.01 -> z_0.995
Z_90 = 1.6448536269514722  # scipy.stats.norm.ppf(0.95)
Z_95 = 1.959963984540054   # scipy.stats.norm.ppf(0.975)
Z_99 = 2.5758293035489004  # scipy.stats.norm.ppf(0.995)

# Known Baseline Residual Standard Errors (sigma_e in MU up to Dec 2025)
# Used as fallback or verified dynamically against training residuals
BASE_SIGMA_DEMAND = 391.30
BASE_SIGMA_SUPPLY = 996.83
BASE_SIGMA_GAP = 1070.88

# Project-Defined Risk Thresholds (in MU)
# Note: Deterministic project decision layer, not calculated by RAG
RISK_THRESHOLD_LOW = 3000.0
RISK_THRESHOLD_MODERATE = 4500.0

# Benchmark Validation Values (Jan - Mar 2026)
VALIDATION_BENCHMARKS = {
    "Jan 2026": {
        "demand_actual": 10067.00,
        "demand_predicted": 11047.51,
        "supply_actual": 10189.56,
        "supply_predicted": 8809.63,
        "gap_actual": -122.56,
        "gap_predicted": 2237.88,
    },
    "Feb 2026": {
        "demand_actual": 10125.00,
        "demand_predicted": 12308.25,
        "supply_actual": 10405.60,
        "supply_predicted": 9229.95,
        "gap_actual": -280.60,
        "gap_predicted": 3078.30,
    },
    "Mar 2026": {
        "demand_actual": 12233.00,
        "demand_predicted": 12594.89,
        "supply_actual": 11247.51,
        "supply_predicted": 9574.64,
        "gap_actual": 985.49,
        "gap_predicted": 3020.25,
    },
}

# Model Development R2 Comparison Scores (From Model Training phase)
MODEL_DEVELOPMENT_SCORES = {
    "Demand": {
        "Random Forest": 0.72,
        "XGBoost": 0.86,
        "LightGBM": 0.78,
        "LSTM": 0.88,
    },
    "Supply": {
        "Random Forest": 0.74,
        "XGBoost": 0.88,
        "LightGBM": 0.93,
        "LSTM": 0.90,
    },
}
