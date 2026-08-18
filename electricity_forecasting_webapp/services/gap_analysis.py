"""
Electricity Demand–Supply Gap Analysis Service.
Calculates the forecasted gap:
  Gap = Predicted Demand - Predicted Supply
Classifies the condition as Shortage, Surplus, or Balanced,
and calculates combined uncertainty and prediction intervals for the gap.
"""

from typing import Dict, Any
import numpy as np
import config
from services.prediction_intervals import calculate_gap_sigma, get_prediction_intervals


def perform_gap_analysis(
    demand_result: Dict[str, Any],
    supply_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform comprehensive Demand - Supply gap analysis.
    
    Parameters:
      demand_result: Result dict from DemandForecaster.predict()
      supply_result: Result dict from SupplyForecaster.predict()
      
    Returns:
      Dict with predicted gap, condition, gap sigma, and gap prediction intervals.
    """
    pred_demand = demand_result["predicted_demand"]
    pred_supply = supply_result["predicted_supply"]
    sigma_demand = demand_result.get("sigma", config.BASE_SIGMA_DEMAND)
    sigma_supply = supply_result.get("sigma", config.BASE_SIGMA_SUPPLY)

    # Calculate point gap
    predicted_gap = pred_demand - pred_supply

    # Condition classification
    if predicted_gap > 0:
        condition = "Shortage"
        condition_description = "Electricity Demand exceeds Supply. Generation ramp-up, storage discharge, or grid import required."
    elif predicted_gap < 0:
        condition = "Surplus"
        condition_description = "Electricity Supply exceeds Demand. Excess generation available for storage, EV charging, or export."
    else:
        condition = "Balanced"
        condition_description = "Electricity Demand and Supply are in exact equilibrium."

    # Combined Gap standard error: sigma_Gap = sqrt(sigma_D^2 + sigma_S^2)
    gap_sigma = calculate_gap_sigma(sigma_demand, sigma_supply)

    # Generate 90%, 95%, 99% prediction intervals for the gap
    gap_intervals = get_prediction_intervals(predicted_gap, gap_sigma)

    actual_demand = demand_result.get("actual_demand")
    actual_supply = supply_result.get("actual_supply")
    actual_gap = None
    if actual_demand is not None and actual_supply is not None:
        actual_gap = float(actual_demand - actual_supply)

    return {
        "predicted_demand": pred_demand,
        "predicted_supply": pred_supply,
        "predicted_gap": predicted_gap,
        "actual_demand": actual_demand,
        "actual_supply": actual_supply,
        "actual_gap": actual_gap,
        "condition": condition,
        "condition_description": condition_description,
        "gap_sigma": gap_sigma,
        "sigma_demand": sigma_demand,
        "sigma_supply": sigma_supply,
        "gap_intervals": gap_intervals,
        "demand_intervals": demand_result["intervals"],
        "supply_intervals": supply_result["intervals"],
    }
