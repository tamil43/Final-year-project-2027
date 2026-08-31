"""
RAG-Based Energy Planning & Policy Recommendation Architecture.
Connected to FAISS Vector Database & Gemini Grounded Recommendation Engine.

Workflow:
  Forecast Results (Demand & Supply from web application models)
          ↓
     Gap Analysis
          ↓
  Risk Classification (Deterministic: <3000 Low, 3000-4500 Moderate, >4500 High)
          ↓
  FAISS Vector Similarity Search (sentence-transformers/all-MiniLM-L6-v2)
          ↓
  Context-Grounded Gemini Recommendation Engine with Document & Page Citations
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("RAG_Webapp_Bridge")

# Ensure project root and RAG module directories are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = CURRENT_DIR.parent
PROJECT_ROOT = WEBAPP_DIR.parent
RAG_DIR = PROJECT_ROOT / "RAG"

for path_entry in [str(PROJECT_ROOT), str(RAG_DIR), str(WEBAPP_DIR)]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

# Import RAG pipeline modules directly
from forecast_retrieval import RAGForecastRetriever
from generate_recommendation import generate_recommendation as run_gemini_recommendation


import streamlit as st

@st.cache_resource(show_spinner=False)
def get_rag_retriever() -> RAGForecastRetriever:
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
    """
    Synthesize structured, evidence-grounded energy planning actions
    by passing REAL forecasting web application results to the FAISS + Gemini RAG pipeline.
    """
    pred_demand = gap_analysis_result.get("predicted_demand", 0.0)
    pred_supply = gap_analysis_result.get("predicted_supply", 0.0)
    pred_gap = gap_analysis_result.get("predicted_gap", pred_demand - pred_supply)
    condition = gap_analysis_result.get("condition", "Shortage")
    risk_level = risk_result.get("risk_level", "Moderate")
    
    # 95% Confidence Interval range
    pi_95 = gap_analysis_result.get("gap_intervals", {}).get("pi_95", {"lower": pred_gap - 1000, "upper": pred_gap + 1000})
    lower_95 = pi_95["lower"]
    upper_95 = pi_95["upper"]

    # Forecast payload constructed directly from real webapp forecast outputs
    forecast_payload = {
        "month": month_name,
        "predicted_demand": float(pred_demand),
        "predicted_supply": float(pred_supply),
        "gap": float(pred_gap),
        "risk_level": risk_level
    }

    try:
        retriever = get_rag_retriever()
        
        # Step 5: FAISS Vector Retrieval (Top-5 Chunks)
        retrieval_res = retriever.retrieve(forecast_payload, top_k=5)
        generated_query = retrieval_res["generated_query"]
        retrieved_chunks = retrieval_res["retrieved_chunks"]
        
        # Step 6: Gemini LLM / Grounded Evidence Recommendation Engine
        rec_res = run_gemini_recommendation(forecast_payload, retrieved_chunks)
        recommendation_text = rec_res["recommendation"]
        execution_mode = rec_res.get("execution_mode", "RAG Knowledge Engine")
        
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
            "framework_disclaimer": "RAG recommendation engine synthesizes evidence from official Tamil Nadu & National energy policies and CEA/TNERC guidelines. It does NOT alter ML point forecasts."
        }
        
    except Exception as e:
        logger.error(f"Error executing RAG retrieval & recommendation: {e}", exc_info=True)
        return {
            "status": "Error",
            "condition": condition,
            "risk_level": risk_level,
            "predicted_gap": pred_gap,
            "ci_95_range": (lower_95, upper_95),
            "generated_query": f"Tamil Nadu grid planning for {month_name} gap {pred_gap:.2f} MU ({risk_level} Risk)",
            "retrieved_chunks": [],
            "recommendation_text": f"RAG Execution Error: {type(e).__name__}: {str(e)}",
            "execution_mode": "Error Fallback Mode",
            "framework_disclaimer": "RAG recommendation engine encountered a runtime error."
        }
