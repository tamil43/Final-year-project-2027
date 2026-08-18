"""
Independent Supply Forecasting Service.
Handles loading of trained Supply LSTM model & scalers, input preprocessing,
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
from utils.preprocessing import prepare_supply_sequence, load_supply_validation_dataset


class SupplyForecaster:
    """
    Independent forecasting pipeline for Electricity Supply.
    """

    def __init__(self):
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.sigma = config.BASE_SIGMA_SUPPLY
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Load trained Supply LSTM model, feature scaler, and target scaler.
        """
        if not config.SUPPLY_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Supply model file missing at: {config.SUPPLY_MODEL_PATH}. "
                "Ensure trained 'Supply_LSTM.keras' is present."
            )
        if not config.SUPPLY_X_SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Supply feature scaler missing at: {config.SUPPLY_X_SCALER_PATH}."
            )
        if not config.SUPPLY_Y_SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Supply target scaler missing at: {config.SUPPLY_Y_SCALER_PATH}."
            )

        self.model = load_model(str(config.SUPPLY_MODEL_PATH))
        self.x_scaler = joblib.load(str(config.SUPPLY_X_SCALER_PATH))
        self.y_scaler = joblib.load(str(config.SUPPLY_Y_SCALER_PATH))

        # Calculate or verify sigma_S dynamically from historical validation residuals
        self._compute_historical_sigma()
        self.is_loaded = True

    def _compute_historical_sigma(self):
        """
        Calculates the exact historical standard error sigma_S on test set up to Dec 2025.
        """
        try:
            if (
                config.SUPPLY_X_TRAIN_PATH.exists()
                and config.SUPPLY_X_TEST_PATH.exists()
                and config.SUPPLY_Y_TEST_PATH.exists()
            ):
                s_X_train = joblib.load(str(config.SUPPLY_X_TRAIN_PATH))
                s_X_test = joblib.load(str(config.SUPPLY_X_TEST_PATH))
                s_y_test = joblib.load(str(config.SUPPLY_Y_TEST_PATH))

                s_X_tr_sc = self.x_scaler.transform(s_X_train.values)
                s_X_te_sc = self.x_scaler.transform(s_X_test.values)
                s_comb = np.vstack([s_X_tr_sc[-6:], s_X_te_sc])
                s_X_seq = np.array([s_comb[i - 6 : i] for i in range(6, len(s_comb))])

                s_test_pred = self.y_scaler.inverse_transform(
                    self.model.predict(s_X_seq, verbose=0)
                ).flatten()
                s_test_act = s_y_test.values.flatten()
                self.sigma = calculate_residual_sigma(s_test_act, s_test_pred)
            else:
                self.sigma = config.BASE_SIGMA_SUPPLY
        except Exception:
            self.sigma = config.BASE_SIGMA_SUPPLY

    def predict(self, raw_sequence_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Generate point forecast and 90%, 95%, 99% prediction intervals
        for a 6-month raw input matrix (shape: 6x18).
        """
        if not self.is_loaded:
            self._load_artifacts()

        tensor_input = prepare_supply_sequence(raw_sequence_matrix, self.x_scaler)
        scaled_pred = self.model.predict(tensor_input, verbose=0)
        point_pred_mu = float(self.y_scaler.inverse_transform(scaled_pred)[0, 0])

        interval_data = get_prediction_intervals(point_pred_mu, self.sigma)
        return {
            "predicted_supply": point_pred_mu,
            "sigma": self.sigma,
            "intervals": interval_data,
        }

    def predict_benchmark_month(self, month_label: str) -> Dict[str, Any]:
        """
        Run inference for the authentic Jan, Feb, or Mar 2026 sequence.
        """
        target_dates = {
            "Jan 2026": "2026-01-01",
            "Feb 2026": "2026-02-01",
            "Mar 2026": "2026-03-01",
        }
        if month_label not in target_dates:
            raise ValueError(f"Unknown benchmark month: {month_label}. Choose from {list(target_dates.keys())}")

        df_s = load_supply_validation_dataset()
        target_date = target_dates[month_label]
        idx = df_s[df_s["Date"] == target_date].index[0]
        seq_df = df_s.iloc[idx - config.SUPPLY_LOOKBACK + 1 : idx + 1]
        raw_matrix = seq_df[config.SUPPLY_FEATURES].values

        res = self.predict(raw_matrix)
        res["month"] = month_label
        res["actual_supply"] = config.VALIDATION_BENCHMARKS[month_label]["supply_actual"]
        return res
