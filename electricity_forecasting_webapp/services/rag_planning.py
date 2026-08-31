"""
RAG-Based Energy Planning & Policy Recommendation Architecture.
Connected to FAISS Vector Database & Grounded Recommendation Engine.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st

logger = logging.getLogger("RAG_Webapp_Bridge")

# Ensure project root and RAG module directories are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = CURRENT_DIR.parent
PROJECT_ROOT = WEBAPP_DIR.parent
RAG_DIR = PROJECT_ROOT / "RAG"

for path_entry in [str(PROJECT_ROOT), str(RAG_DIR), str(WEBAPP_DIR)]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)


@st.cache_resource(show_spinner=False)
def get_rag_retriever():
    from forecast_retrieval import RAGForecastRetriever
    vector_db_path = RAG_DIR / "vector_db"
    if not vector_db_path.exists():
        vector_db_path = PROJECT_ROOT / "RAG" / "vector_db"
    if not vector_db_path.exists():
        vector_db_path = Path("RAG/vector_db")
        
    logger.info(f"Initializing FAISS RAG Retriever from {vector_db_path.resolve()}")
    return RAGForecastRetriever(vector_db_path)


def generate_energy_planning_recommendation(
    gap_analysis_result: Dict[str, Any],
    risk_result: Dict[str, Any],
    month_name: str = "Target Period"
) -> Dict[str, Any]:
    from generate_recommendation import generate_recommendation as run_gemini_recommendation

    pred_demand = gap_analysis_result.get("predicted_demand", 0.0)
    pred_supply = gap_analysis_result.get("predicted_supply", 0.0)
    pred_gap = gap_analysis_result.get("predicted_gap", pred_demand - pred_supply)
    condition = gap_analysis_result.get("condition", "Shortage")
    risk_level = risk_result.get("risk_level", "Moderate")
    
    pi_95 = gap_analysis_result.get("gap_intervals", {}).get("pi_95", {"lower": pred_gap - 1000, "upper": pred_gap + 1000})
    lower_95 = pi_95["lower"]
    upper_95 = pi_95["upper"]

    forecast_payload = {
        "month": month_name,
        "predicted_demand": float(pred_demand),
        "predicted_supply": float(pred_supply),
        "gap": float(pred_gap),
        "risk_level": risk_level
    }

    try:
        retriever = get_rag_retriever()
        retrieval_res = retriever.retrieve(forecast_payload, top_k=5)
        generated_query = retrieval_res.get("generated_query", "")
        retrieved_chunks = retrieval_res.get("retrieved_chunks", [])
        
        rec_res = run_gemini_recommendation(forecast_payload, retrieved_chunks)
        recommendation_text = rec_res.get("recommendation", "")
        execution_mode = rec_res.get("execution_mode", "Dynamic Grounded Engine")
        
        return {
            "status": "Success",
            "condition": condition,
            "risk_level": risk_level,
            "predicted_gap": pred_gap,
            "ci_95_range": (lower_95, upper_95),
            "generated_query": generated_query,
            "retrieved_chunks": retrieved_chunks,
            "recommendation_text": recommendation_text,
            "execution_mode": execution_mode,
            "framework_disclaimer": "RAG recommendations are grounded in verified state energy policies and CEA/TNERC planning guidelines."
        }
        
    except Exception as e:
        logger.error(f"Error executing RAG retrieval & recommendation: {e}", exc_info=True)
        return {
            "status": "Error",
            "condition": condition,
            "risk_level": risk_level,
            "predicted_gap": pred_gap,
            "ci_95_range": (lower_95, upper_95),
            "generated_query": f"Tamil Nadu grid planning for {month_name}",
            "retrieved_chunks": [],
            "recommendation_text": f"1. Purchase peak electricity from daily power exchanges.\n\n2. Shift farm pumping to solar hours.\n\n3. Keep backup thermal generation ready.",
            "execution_mode": "Resilient Fallback",
            "framework_disclaimer": "RAG recommendation fallback active."
        }
