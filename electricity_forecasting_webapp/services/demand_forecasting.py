"""
Independent Demand Forecasting Service.
Handles loading of trained Demand LSTM model & scalers, input preprocessing,
point prediction in MU, and confidence interval estimation.
"""

from typing import Dict, Any, Optional
import os
import joblib
import numpy as np
import pandas as pd
try:
    from tensorflow.keras.models import load_model
except (ImportError, ModuleNotFoundError):
    from keras.models import load_model

import config
from services.prediction_intervals import get_prediction_intervals, calculate_residual_sigma
from utils.preprocessing import prepare_demand_sequence, load_demand_validation_dataset


class DemandForecaster:
    """
    Independent forecasting pipeline for Electricity Demand.
    """

    def __init__(self):
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.sigma = config.BASE_SIGMA_DEMAND
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Load trained Demand LSTM model, feature scaler, and target scaler.
        """
        if not config.DEMAND_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Demand model file missing at: {config.DEMAND_MODEL_PATH}. "
                "Ensure trained 'Demand_LSTM_Final.keras' is present."
            )
        if not config.DEMAND_X_SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Demand feature scaler missing at: {config.DEMAND_X_SCALER_PATH}."
            )
        if not config.DEMAND_Y_SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Demand target scaler missing at: {config.DEMAND_Y_SCALER_PATH}."
            )

        self.model = load_model(str(config.DEMAND_MODEL_PATH))
        self.x_scaler = joblib.load(str(config.DEMAND_X_SCALER_PATH))
        self.y_scaler = joblib.load(str(config.DEMAND_Y_SCALER_PATH))

        # Calculate or verify sigma_D dynamically from historical validation residuals
        self._compute_historical_sigma()
        self.is_loaded = True

    def _compute_historical_sigma(self):
        """
        Calculates the exact historical standard error sigma_D on test set up to Dec 2025.
        """
        try:
            if (
                config.DEMAND_X_TRAIN_PATH.exists()
                and config.DEMAND_X_TEST_PATH.exists()
                and config.DEMAND_Y_TEST_PATH.exists()
            ):
                d_X_train = joblib.load(str(config.DEMAND_X_TRAIN_PATH))
                d_X_test = joblib.load(str(config.DEMAND_X_TEST_PATH))
                d_y_test = joblib.load(str(config.DEMAND_Y_TEST_PATH))

                d_X_tr_sc = self.x_scaler.transform(d_X_train.values)
                d_X_te_sc = self.x_scaler.transform(d_X_test.values)
                d_comb = np.vstack([d_X_tr_sc[-3:], d_X_te_sc])
                d_X_seq = np.array([d_comb[i - 3 : i] for i in range(3, len(d_comb))])

                d_test_pred = self.y_scaler.inverse_transform(
                    self.model.predict(d_X_seq, verbose=0)
                ).flatten()
                d_test_act = d_y_test.values.flatten()
                self.sigma = calculate_residual_sigma(d_test_act, d_test_pred)
            else:
                self.sigma = config.BASE_SIGMA_DEMAND
        except Exception:
            self.sigma = config.BASE_SIGMA_DEMAND

    def predict(self, raw_sequence_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Generate point forecast and 90%, 95%, 99% prediction intervals
        for a 3-month raw input matrix (shape: 3x14).
        """
        if not self.is_loaded:
            self._load_artifacts()

        tensor_input = prepare_demand_sequence(raw_sequence_matrix, self.x_scaler)
        scaled_pred = self.model.predict(tensor_input, verbose=0)
        point_pred_mu = float(self.y_scaler.inverse_transform(scaled_pred)[0, 0])

        interval_data = get_prediction_intervals(point_pred_mu, self.sigma)
        return {
            "predicted_demand": point_pred_mu,
            "sigma": self.sigma,
            "intervals": interval_data,
        }

    def predict_benchmark_month(self, month_label: str) -> Dict[str, Any]:
        """
        Run inference for the authentic Jan, Feb, or Mar 2026 sequence.
        """
        date_ranges = {
            "Jan 2026": ["2025-11-01", "2025-12-01", "2026-01-01"],
            "Feb 2026": ["2025-12-01", "2026-01-01", "2026-02-01"],
            "Mar 2026": ["2026-01-01", "2026-02-01", "2026-03-01"],
        }
        if month_label not in date_ranges:
            raise ValueError(f"Unknown benchmark month: {month_label}. Choose from {list(date_ranges.keys())}")

        df_val = load_demand_validation_dataset()
        seq_dates = pd.to_datetime(date_ranges[month_label])
        seq_df = df_val[df_val["Date"].isin(seq_dates)].sort_values("Date")
        raw_matrix = seq_df[config.DEMAND_FEATURES].values

        res = self.predict(raw_matrix)
        res["month"] = month_label
        res["actual_demand"] = config.VALIDATION_BENCHMARKS[month_label]["demand_actual"]
        return res
