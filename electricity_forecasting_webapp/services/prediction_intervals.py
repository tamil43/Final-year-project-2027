"""
Prediction Intervals & Uncertainty Quantification Service.
Calculates historical standard error (sigma_e) from validation residuals
and constructs two-tailed 90%, 95%, and 99% prediction intervals centered
on point forecasts using standard normal critical values.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import norm
import config


def calculate_residual_sigma(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculate the sample standard deviation of residuals (ddof=1).
    sigma_e = sqrt( sum( (y_act - y_pred)^2 ) / (N - 1) )
    """
    residuals = np.array(y_actual).flatten() - np.array(y_predicted).flatten()
    sigma = float(np.std(residuals, ddof=1))
    return sigma


def calculate_gap_sigma(sigma_demand: float, sigma_supply: float) -> float:
    """
    Calculate combined standard error for the Demand - Supply gap.
    sigma_Gap = sqrt(sigma_D^2 + sigma_S^2)
    Assumes independent forecasting error distributions.
    """
    return float(np.sqrt(sigma_demand**2 + sigma_supply**2))


def get_prediction_intervals(point_forecast: float, sigma: float) -> Dict[str, Any]:
    """
    Generate 90%, 95%, and 99% symmetric prediction intervals for a given point forecast.
    
    Formula:
      PI_{1-alpha} = [ y_hat - z_{1-alpha/2} * sigma, y_hat + z_{1-alpha/2} * sigma ]
      
    Critical values:
      90% -> z = 1.644853 (norm.ppf(0.95))
      95% -> z = 1.959964 (norm.ppf(0.975))
      99% -> z = 2.575829 (norm.ppf(0.995))
    """
    z90 = config.Z_90
    z95 = config.Z_95
    z99 = config.Z_99

    margin_90 = z90 * sigma
    margin_95 = z95 * sigma
    margin_99 = z99 * sigma

    return {
        "point_forecast": float(point_forecast),
        "sigma": float(sigma),
        "pi_90": {
            "lower": float(point_forecast - margin_90),
            "upper": float(point_forecast + margin_90),
            "margin": float(margin_90),
            "z": float(z90),
        },
        "pi_95": {
            "lower": float(point_forecast - margin_95),
            "upper": float(point_forecast + margin_95),
            "margin": float(margin_95),
            "z": float(z95),
        },
        "pi_99": {
            "lower": float(point_forecast - margin_99),
            "upper": float(point_forecast + margin_99),
            "margin": float(margin_99),
            "z": float(z99),
        },
    }
