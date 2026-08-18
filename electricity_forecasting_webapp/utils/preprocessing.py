"""
Preprocessing & Feature Engineering Utilities.
Handles feature scaling, cyclical encoding, lag/rolling features,
and 3D tensor sequence shaping for Demand and Supply LSTM models.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import config


def encode_cyclical_month(month: int) -> Tuple[float, float]:
    """
    Compute cyclical sine and cosine representations for a month (1-12).
    """
    month_sin = float(np.sin(2 * np.pi * month / 12))
    month_cos = float(np.cos(2 * np.pi * month / 12))
    return month_sin, month_cos


def prepare_demand_sequence(
    raw_feature_matrix: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Scale and reshape a 3-month Demand feature matrix (3 timesteps, 14 features)
    into a 3D tensor suitable for Demand LSTM inference: (1, 3, 14).
    """
    if raw_feature_matrix.shape != (config.DEMAND_LOOKBACK, len(config.DEMAND_FEATURES)):
        raise ValueError(
            f"Demand input matrix must have shape ({config.DEMAND_LOOKBACK}, {len(config.DEMAND_FEATURES)}), "
            f"but got {raw_feature_matrix.shape}."
        )

    # Scale with the pre-fitted feature scaler
    scaled_matrix = scaler.transform(raw_feature_matrix)
    
    # Reshape to (1, 3, 14)
    tensor_input = scaled_matrix.reshape(1, config.DEMAND_LOOKBACK, len(config.DEMAND_FEATURES))
    return tensor_input


def prepare_supply_sequence(
    raw_feature_matrix: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Scale and reshape a 6-month Supply feature matrix (6 timesteps, 18 features)
    into a 3D tensor suitable for Supply LSTM inference: (1, 6, 18).
    """
    if raw_feature_matrix.shape != (config.SUPPLY_LOOKBACK, len(config.SUPPLY_FEATURES)):
        raise ValueError(
            f"Supply input matrix must have shape ({config.SUPPLY_LOOKBACK}, {len(config.SUPPLY_FEATURES)}), "
            f"but got {raw_feature_matrix.shape}."
        )

    # Scale with the pre-fitted feature scaler
    scaled_matrix = scaler.transform(raw_feature_matrix)
    
    # Reshape to (1, 6, 18)
    tensor_input = scaled_matrix.reshape(1, config.SUPPLY_LOOKBACK, len(config.SUPPLY_FEATURES))
    return tensor_input


def load_demand_validation_dataset() -> pd.DataFrame:
    """
    Load and preprocess the Demand validation dataset (Book1.xlsx)
    with the exact feature pipeline from the project notebooks.
    """
    if not config.DEMAND_VAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Demand validation data file not found: {config.DEMAND_VAL_DATA_PATH}")

    df = pd.read_excel(config.DEMAND_VAL_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Demand_Lag_1"] = df["Electricity_Requirement"].shift(1)
    df["Demand_Lag_2"] = df["Electricity_Requirement"].shift(2)
    df["Demand_Lag_3"] = df["Electricity_Requirement"].shift(3)
    df["Demand_Rolling_3"] = df["Electricity_Requirement"].shift(1).rolling(3).mean()
    df["Demand_Rolling_6"] = df["Electricity_Requirement"].shift(1).rolling(6).mean()
    df["Demand_Rolling_12"] = df["Electricity_Requirement"].shift(1).rolling(12).mean()

    # Apply rolling 12 adjustments for recent months (from notebook)
    df.loc[df["Date"] == "2025-11-01", "Demand_Rolling_12"] = 10853.67
    df.loc[df["Date"] == "2025-12-01", "Demand_Rolling_12"] = 10903.83
    df.loc[df["Date"] == "2026-01-01", "Demand_Rolling_12"] = 10964.75

    return df


def load_supply_validation_dataset() -> pd.DataFrame:
    """
    Load and prepare the Supply dataset (supply_dataset.csv).
    """
    if not config.SUPPLY_DATASET_PATH.exists():
        raise FileNotFoundError(f"Supply dataset file not found: {config.SUPPLY_DATASET_PATH}")

    df = pd.read_csv(config.SUPPLY_DATASET_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df
