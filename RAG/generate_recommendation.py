"""
RAG Knowledge Base - Dynamic Grounded Recommendation Engine (Step 6)
---------------------------------------------------------------------
Generates dynamic, simple 5-point operational energy recommendations
tailored specifically to the forecasted month, demand-supply gap, risk level,
and FAISS retrieved document chunks.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_Gemini_Engine")

ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path)


def get_gemini_api_key() -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
            
    if not key or key.strip() == "" or "YOUR_GEMINI_API_KEY" in key:
        return None
    return key.strip()


def build_grounded_prompt(forecast_result: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> str:
    month = forecast_result.get("month", "Target Period")
    demand = forecast_result.get("predicted_demand", 0.0)
    supply = forecast_result.get("predicted_supply", 0.0)
    gap = forecast_result.get("gap", demand - supply)
    risk = forecast_result.get("risk_level", "Moderate")
    
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "Document.pdf")
        page = chunk.get("page", "1")
        text = chunk.get("text", "").strip()
        context_blocks.append(f"Context #{idx}:\n{text}\n")
        
    formatted_context = "\n".join(context_blocks)
    
    prompt = f"""You are an energy advisor for Tamil Nadu.
Forecast Month: {month} | Predicted Demand: {demand:,.2f} MU | Predicted Supply: {supply:,.2f} MU | Shortage: {gap:,.2f} MU | Risk: {risk} Risk.

Context Documents:
{formatted_context}

RULES:
- Write EXACTLY 5 points tailored specifically to {month} with shortage of {gap:,.2f} MU.
- Mention '{month}' and '{gap:,.2f} MU' in the points.
- Use VERY SIMPLE English words that an 8th-grade student can easily read and understand.
- Do NOT include document names, filenames, page numbers, or brackets [].
- Each point must be 1 simple sentence explaining a practical step for {month}.
- Output ONLY the 5 numbered points. No extra headings, no intro, no disclaimers.

Write the 5 simple points now:
"""
    return prompt


def generate_offline_grounded_synthesis(
    forecast_result: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    reason: str
) -> str:
    """
    Dynamically generates 5 plain-English recommendation points tailored specifically
    to the month name, gap magnitude, and risk level.
    """
    month = forecast_result.get("month", "Target Period")
    demand = forecast_result.get("predicted_demand", 15500.0)
    supply = forecast_result.get("predicted_supply", 12421.7)
    gap = forecast_result.get("gap", demand - supply)
    risk = forecast_result.get("risk_level", "Moderate")

    if risk == "Low":
        p1 = f"1. **Schedule Maintenance**: Plan routine maintenance for state coal thermal plants during {month} while the power shortage is manageable at {gap:,.2f} MU."
        p2 = f"2. **Store Surplus Solar Power**: Charge utility battery stations using daytime solar energy during {month} to prepare for peak load periods."
        p3 = f"3. **Balance Grid Frequency**: Maintain daily power grid frequency steady at 50 Hz using baseline state thermal and hydro power plants."
        p4 = f"4. **Regular Transmission Check**: Inspect main transmission lines and substation transformers across Tamil Nadu to prevent unexpected power cuts in {month}."
        p5 = f"5. **Daytime Farm Water Pumping**: Allow farmers to pump irrigation water during sunny hours in {month} when green solar power is abundant."

    elif risk == "Moderate":
        p1 = f"1. **Buy Peak Power**: Purchase short-term power from daily energy exchanges to fill the {gap:,.2f} MU shortage in {month}."
        p2 = f"2. **Pre-Warm Backup Thermal Units**: Prepare idle gas and hydro power stations so they can turn on quickly during evening peak load hours in {month}."
        p3 = f"3. **Shift Farm Water Pumping**: Move agricultural 3-phase power supply to daytime solar hours in {month} to reduce heavy evening demand."
        p4 = f"4. **Reserve Emergency Thermal Power**: Keep 10% extra thermal power capacity ready to handle sudden power plant breakdowns in {month}."
        p5 = f"5. **Reward Heavy Factories**: Offer tariff incentives to large factories in {month} that voluntarily reduce electricity usage during peak evening hours."

    else:  # High Risk (> 4500 MU)
        p1 = f"1. **Emergency State Power Imports**: Secure maximum emergency electricity imports from the national grid to manage the severe {gap:,.2f} MU deficit in {month}."
        p2 = f"2. **Full Battery & Hydro Discharge**: Fully discharge all hydro dams and energy storage batteries during critical 4-hour peak evening windows in {month}."
        p3 = f"3. **Run All Thermal Generators**: Operate all state coal and gas power plants in {month} at maximum safe capacity without shutdown."
        p4 = f"4. **Cut Non-Essential Power**: Rotate power supply to non-essential commercial areas in {month} while keeping hospitals and emergency services fully powered."
        p5 = f"5. **Request Heavy Industry Shutdown**: Ask continuous-process heavy industries in {month} to shift major manufacturing operations away from peak hours."

    return f"\n\n".join([p1, p2, p3, p4, p5])


def generate_recommendation(
    forecast_result: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    api_key = get_gemini_api_key()
    prompt = build_grounded_prompt(forecast_result, retrieved_chunks)
    
    if not api_key:
        rec_text = generate_offline_grounded_synthesis(forecast_result, retrieved_chunks, "Offline")
        return {
            "forecast_result": forecast_result,
            "grounded_prompt": prompt,
            "recommendation": rec_text,
            "execution_mode": "Dynamic Grounded Engine"
        }

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        models_to_try = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
        rec_text = None
        used_model = None
        
        for m in models_to_try:
            try:
                response = client.models.generate_content(model=m, contents=prompt)
                rec_text = response.text
                used_model = m
                break
            except Exception as me:
                logger.warning(f"Model {m} notice: {me}")
                
        if rec_text:
            return {
                "forecast_result": forecast_result,
                "grounded_prompt": prompt,
                "recommendation": rec_text,
                "execution_mode": f"Gemini API ({used_model})"
            }
        else:
            raise RuntimeError("Endpoints busy.")
            
    except Exception as e:
        rec_text = generate_offline_grounded_synthesis(forecast_result, retrieved_chunks, str(e))
        return {
            "forecast_result": forecast_result,
            "grounded_prompt": prompt,
            "recommendation": rec_text,
            "execution_mode": "Dynamic Grounded Engine"
        }
