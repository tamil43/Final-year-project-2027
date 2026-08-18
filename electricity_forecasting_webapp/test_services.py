"""
Verification script for backend services of electricity_forecasting_webapp.
"""
import sys
import io

# Ensure UTF-8 output on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

import config
from services.demand_forecasting import DemandForecaster
from services.supply_forecasting import SupplyForecaster
from services.gap_analysis import perform_gap_analysis
from services.risk_assessment import assess_risk
from services.rag_planning import generate_energy_planning_recommendation

print("========================================")
print("TESTING WEB APPLICATION BACKEND SERVICES")
print("========================================")

# 1. Demand Forecaster
d_fc = DemandForecaster()
print(f"Demand Forecaster Loaded! Sigma: {d_fc.sigma:.2f} MU")
assert abs(d_fc.sigma - 391.30) < 0.1, f"Unexpected Demand sigma: {d_fc.sigma}"

# 2. Supply Forecaster
s_fc = SupplyForecaster()
print(f"Supply Forecaster Loaded! Sigma: {s_fc.sigma:.2f} MU")
assert abs(s_fc.sigma - 996.83) < 0.1, f"Unexpected Supply sigma: {s_fc.sigma}"

# 3. Validation Months
expected_results = {
    "Jan 2026": {"demand": 11047.51, "supply": 8809.63, "gap": 2237.88, "risk": "Low"},
    "Feb 2026": {"demand": 12308.25, "supply": 9229.95, "gap": 3078.30, "risk": "Moderate"},
    "Mar 2026": {"demand": 12594.89, "supply": 9574.64, "gap": 3020.25, "risk": "Moderate"},
}

for m in ["Jan 2026", "Feb 2026", "Mar 2026"]:
    d_res = d_fc.predict_benchmark_month(m)
    s_res = s_fc.predict_benchmark_month(m)
    gap_res = perform_gap_analysis(d_res, s_res)
    risk_res = assess_risk(gap_res["predicted_gap"], gap_res["condition"])
    rag_res = generate_energy_planning_recommendation(gap_res, risk_res)

    print(f"\n--- {m} ---")
    print(f"Demand Pred: {d_res['predicted_demand']:.2f} MU (Actual: {d_res['actual_demand']:.2f} MU)")
    print(f"Supply Pred: {s_res['predicted_supply']:.2f} MU (Actual: {s_res['actual_supply']:.2f} MU)")
    print(f"Gap Pred   : {gap_res['predicted_gap']:.2f} MU (Condition: {gap_res['condition']})")
    print(f"Risk Level : {risk_res['risk_level']} ({risk_res['badge']})")
    print(f"95% Gap CI : [{gap_res['gap_intervals']['pi_95']['lower']:.2f}, {gap_res['gap_intervals']['pi_95']['upper']:.2f}] MU")
    print(f"Recommendations Generated: {len(rag_res['recommendations'])}")

    # Verify numerical match with project benchmark
    exp = expected_results[m]
    assert abs(d_res["predicted_demand"] - exp["demand"]) < 0.1, f"Demand mismatch for {m}"
    assert abs(s_res["predicted_supply"] - exp["supply"]) < 0.1, f"Supply mismatch for {m}"
    assert abs(gap_res["predicted_gap"] - exp["gap"]) < 0.1, f"Gap mismatch for {m}"
    assert risk_res["risk_level"] == exp["risk"], f"Risk mismatch for {m}"

print("\n========================================")
print("ALL VERIFICATION TESTS PASSED PERFECTLY!")
print("========================================")
