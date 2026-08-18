"""
Deterministic Risk Classification Service.
Evaluates forecasted electricity gap against project-defined thresholds:
  - Gap < 3000 MU   -> Low Risk / Lower Shortage
  - 3000 - 4500 MU  -> Moderate Risk
  - Gap > 4500 MU   -> High Risk

IMPORTANT:
These thresholds are project-defined decision criteria.
Risk classification is a deterministic decision layer, NOT calculated by RAG.
"""

from typing import Dict, Any
import config


def assess_risk(predicted_gap: float, condition: str) -> Dict[str, Any]:
    """
    Perform deterministic risk classification based on forecasted gap magnitude.
    
    Parameters:
      predicted_gap: Forecasted Demand - Supply in MU
      condition: 'Shortage', 'Surplus', or 'Balanced'
      
    Returns:
      Dict with risk_level, severity_color, badge, description, and threshold_meta.
    """
    # Deterministic rule evaluation
    if condition == "Surplus":
        risk_level = "Low"
        risk_category = "Surplus (No Deficit Risk)"
        severity_color = "#10B981"  # Emerald Green
        badge = "🟢 LOW RISK / SURPLUS"
        description = (
            "System supply exceeds forecasted demand. No deficit risk observed. "
            "Grid management should prioritize renewable curtailment minimization and energy storage charging."
        )
    elif predicted_gap < config.RISK_THRESHOLD_LOW:
        risk_level = "Low"
        risk_category = "Low Shortage Risk"
        severity_color = "#3B82F6"  # Blue / Info
        badge = "🔵 LOW RISK"
        description = (
            f"Forecasted gap ({predicted_gap:,.2f} MU) is below the {config.RISK_THRESHOLD_LOW:,.0f} MU threshold. "
            "Manageable via routine thermal spinning reserve dispatch and short-term bilateral power procurement."
        )
    elif config.RISK_THRESHOLD_LOW <= predicted_gap <= config.RISK_THRESHOLD_MODERATE:
        risk_level = "Moderate"
        risk_category = "Moderate Shortage Risk"
        severity_color = "#F59E0B"  # Amber / Warning
        badge = "🟡 MODERATE RISK"
        description = (
            f"Forecasted gap ({predicted_gap:,.2f} MU) lies within the moderate band "
            f"({config.RISK_THRESHOLD_LOW:,.0f} - {config.RISK_THRESHOLD_MODERATE:,.0f} MU). "
            "Requires proactive peaking plant activation, energy exchange trading, and industrial demand response scheduling."
        )
    else:  # predicted_gap > 4500 MU
        risk_level = "High"
        risk_category = "High Shortage Risk"
        severity_color = "#EF4444"  # Red / Danger
        badge = "🔴 HIGH RISK"
        description = (
            f"Forecasted gap ({predicted_gap:,.2f} MU) exceeds the critical {config.RISK_THRESHOLD_MODERATE:,.0f} MU threshold. "
            "Urgent grid reliability measures required: emergency power imports, battery energy storage system discharge, "
            "and active load curtailment protocols."
        )

    return {
        "risk_level": risk_level,
        "risk_category": risk_category,
        "severity_color": severity_color,
        "badge": badge,
        "description": description,
        "thresholds": {
            "low_limit": config.RISK_THRESHOLD_LOW,
            "moderate_limit": config.RISK_THRESHOLD_MODERATE,
            "nature": "Project-Defined Deterministic Decision Layer",
        },
    }
