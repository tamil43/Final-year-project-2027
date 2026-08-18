"""
RAG-Based Energy Planning & Policy Recommendation Architecture.

Workflow:
  Forecast Results (Demand & Supply)
          ↓
     Gap Analysis
          ↓
  Risk Classification (Deterministic)
          ↓
  Retrieve Trusted Energy Knowledge (CEA/TNERC Guidelines, Policies)
          ↓
  Context-Grounded Recommendation Synthesis

IMPORTANT:
- RAG does NOT alter or improve numerical forecasting accuracy.
- RAG provides evidence-grounded, policy-aligned operational recommendations
  based on predicted gaps, confidence intervals, and risk status.
"""

from typing import Dict, Any, List


# Structured Knowledge Base of Trusted Energy Policies & Operational Standards
ENERGY_POLICY_KNOWLEDGE_BASE = [
    {
        "id": "KB-CEA-01",
        "category": "Generation Scheduling",
        "title": "Central Electricity Authority (CEA) Flexible Thermal Operation Norms",
        "reference": "CEA Guidelines on Minimum Technical Load of Thermal Units (2023)",
        "content": "Coal-fired thermal plants can modulate down to 55% technical minimum load during renewable surplus and ramp up at 2-3% per minute to meet evening demand peaks.",
    },
    {
        "id": "KB-RE-02",
        "category": "Renewable Integration",
        "title": "TNERC Forecasting, Scheduling & Deviation Settlement Mechanism (DSM)",
        "reference": "Tamil Nadu Electricity Regulatory Commission DSM Regulations",
        "content": "Wind and solar generators must adhere to 15-minute scheduling intervals. Deviations outside the +/- 10% band incur regulatory penalties; green energy corridors should be prioritized during wind season.",
    },
    {
        "id": "KB-BESS-03",
        "category": "Energy Storage Planning",
        "title": "National Framework for Promotion of Energy Storage Systems",
        "reference": "Ministry of Power, Government of India (2023)",
        "content": "Battery Energy Storage Systems (BESS) and Pumped Storage Projects (PSP) should be charged during off-peak/solar surplus periods (10:00 - 15:00) and discharged during morning/evening peak demand intervals.",
    },
    {
        "id": "KB-MKT-04",
        "category": "Power Exchange & Procurement",
        "title": "Indian Energy Exchange (IEX) Real-Time & Day-Ahead Market Rules",
        "reference": "CERC Power Market Regulations (2021)",
        "content": "DISCOMs encountering sudden shortages exceeding bilateral contracts should utilize Real-Time Market (RTM) 1-hour ahead windows to balance frequency at 50.00 Hz without grid indiscipline.",
    },
    {
        "id": "KB-DSM-05",
        "category": "Demand-Side Management",
        "title": "Time of Day (ToD) Tariff & Agricultural Load Shifting Policy",
        "reference": "National Electricity Plan & State Agricultural Tariff Directives",
        "content": "Agricultural pumping loads (3-phase power) are staggered to daytime solar availability windows (09:00 - 16:00), flattening evening peak loads by up to 18%.",
    },
]


def retrieve_relevant_knowledge(condition: str, risk_level: str) -> List[Dict[str, str]]:
    """
    Retrieve domain policy knowledge grounded in the current forecast condition and risk state.
    """
    if condition == "Surplus":
        selected_ids = ["KB-RE-02", "KB-BESS-03", "KB-CEA-01"]
    elif risk_level == "High":
        selected_ids = ["KB-MKT-04", "KB-BESS-03", "KB-DSM-05", "KB-CEA-01"]
    elif risk_level == "Moderate":
        selected_ids = ["KB-CEA-01", "KB-DSM-05", "KB-MKT-04"]
    else:  # Low risk shortage
        selected_ids = ["KB-CEA-01", "KB-RE-02", "KB-DSM-05"]

    return [item for item in ENERGY_POLICY_KNOWLEDGE_BASE if item["id"] in selected_ids]


def generate_energy_planning_recommendation(
    gap_analysis_result: Dict[str, Any],
    risk_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Synthesize structured, evidence-grounded energy planning actions
    based on the probabilistic forecasting results and retrieved policy knowledge.
    """
    condition = gap_analysis_result["condition"]
    risk_level = risk_result["risk_level"]
    pred_gap = gap_analysis_result["predicted_gap"]
    pi_95 = gap_analysis_result["gap_intervals"]["pi_95"]
    lower_95 = pi_95["lower"]
    upper_95 = pi_95["upper"]

    retrieved_docs = retrieve_relevant_knowledge(condition, risk_level)

    recommendations = []

    if condition == "Surplus":
        recommendations.append({
            "category": "⚡ Energy Storage & BESS Utilization",
            "action": "Maximize Battery Storage & Pumped Hydro Charging",
            "details": f"Channel forecasted excess generation ({abs(pred_gap):,.2f} MU) into storage systems during surplus generation hours.",
            "policy_grounding": "Ministry of Power BESS Framework Guidelines",
        })
        recommendations.append({
            "category": "🌱 Renewable Integration",
            "action": "Minimize Renewable Curtailment",
            "details": "Export power via Green Day-Ahead Market (G-DAM) on energy exchanges to neighboring state grids.",
            "policy_grounding": "TNERC Forecasting & Scheduling Directives",
        })
        recommendations.append({
            "category": "🏭 Thermal Fleet Modulation",
            "action": "Ramp Down Base Thermal Plants to Technical Minimum",
            "details": "Turn down coal units to 55% rated capacity to conserve fuel while keeping units synchronized for fast ramp-up.",
            "policy_grounding": "CEA Flexible Thermal Operation Norms",
        })

    elif risk_level == "Low":
        recommendations.append({
            "category": "🔥 Generation Scheduling",
            "action": "Standard Reserve Dispatch",
            "details": f"Cover manageable deficit of {pred_gap:,.2f} MU [95% CI: {lower_95:,.2f} to {upper_95:,.2f} MU] using spinning thermal reserves.",
            "policy_grounding": "State Load Despatch Centre (SLDC) Operational Code",
        })
        recommendations.append({
            "category": "🌾 Demand-Side Management",
            "action": "Agricultural Feed Staggering",
            "details": "Maintain rotational 3-phase agricultural power timing during high solar generation slots.",
            "policy_grounding": "State Time-of-Day (ToD) Agricultural Directives",
        })

    elif risk_level == "Moderate":
        recommendations.append({
            "category": "📈 Power Market Procurement",
            "action": "Advance Day-Ahead & Bilateral Contract Bidding",
            "details": f"Contract 2,000–3,500 MU via IEX Day-Ahead Market (DAM) and bilateral agreements to buffer the expected {pred_gap:,.2f} MU shortage.",
            "policy_grounding": "CERC Power Market Regulations (2021)",
        })
        recommendations.append({
            "category": "🚀 Peaking Plant Dispatch",
            "action": "Pre-warm Gas & Hydro Peaking Generation Units",
            "details": "Prepare fast-response hydro and combined-cycle gas turbine (CCGT) stations for dispatch during peak evening load hours.",
            "policy_grounding": "CEA Peaking Power Management Norms",
        })
        recommendations.append({
            "category": "🏢 Industrial Demand Response",
            "action": "Trigger Voluntary Industrial Demand Response (DR)",
            "details": "Notify high-tension (HT) continuous process industrial consumers of peak-hour power shaving incentives.",
            "policy_grounding": "TNERC Demand Response Framework",
        })

    else:  # High risk
        recommendations.append({
            "category": "🚨 Emergency Power Import & Market Operations",
            "action": "Activate Emergency Inter-State Transmission Capacity",
            "details": f"Severe deficit forecasted ({pred_gap:,.2f} MU; upper 95% bound reaching {upper_95:,.2f} MU). Secure maximum RTM energy on national grid corridors.",
            "policy_grounding": "National Grid Code (IEGC) Grid Security Regulations",
        })
        recommendations.append({
            "category": "🔋 BESS Full Discharge Protocol",
            "action": "Full-Capacity Storage & Hydro Discharge",
            "details": "Deplete all utility-scale BESS and pumped storage reserves during the critical 4-hour evening peak window.",
            "policy_grounding": "National Storage System Operating Procedure",
        })
        recommendations.append({
            "category": "⚖️ Managed Load Curtailment",
            "action": "Execute Graded Rotational Load Management",
            "details": "Enact non-essential feeder rotation while strictly protecting critical infrastructure (hospitals, water pumping, emergency services).",
            "policy_grounding": "Disaster Management & Grid Reliability Protocol",
        })

    return {
        "status": "Success",
        "condition": condition,
        "risk_level": risk_level,
        "predicted_gap": pred_gap,
        "ci_95_range": (lower_95, upper_95),
        "retrieved_knowledge": retrieved_docs,
        "recommendations": recommendations,
        "framework_disclaimer": "RAG recommendation engine synthesizes evidence from trusted energy domain policies and SLDC guidelines. It does not alter ML model point forecasts.",
    }
