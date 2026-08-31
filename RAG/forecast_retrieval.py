"""
RAG Knowledge Base - Forecast-Based Dynamic Query & Context Retrieval (Step 5)
-------------------------------------------------------------------------------
This module connects electricity demand/supply forecasting outputs and deterministic
risk classifications (Low <3000 MU, Moderate 3000-4500 MU, High >4500 MU) to the FAISS
vector database. It dynamically builds domain-specific RAG queries based on forecast conditions,
normalizes query embeddings, performs cosine similarity search, and retrieves top-K context chunks.
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_Forecast_Retrieval")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def safe_ascii(text: str) -> str:
    """Cleans text for safe Windows console output without charmap errors."""
    return text.encode("ascii", "replace").decode("ascii")


def calculate_risk_level(gap_mu: float) -> str:
    """
    Deterministic risk threshold evaluation:
      - Gap < 3000 MU   -> Low Risk
      - 3000 to 4500 MU -> Moderate Risk
      - Gap > 4500 MU   -> High Risk
    """
    if gap_mu < 3000.0:
        return "Low"
    elif 3000.0 <= gap_mu <= 4500.0:
        return "Moderate"
    else:
        return "High"


def build_forecast_query(forecast_result: Dict[str, Any]) -> str:
    """
    Programmatically constructs a dynamic domain-tailored search query based on
    forecast metrics (month, predicted_demand, predicted_supply, gap, risk_level).
    """
    month = forecast_result.get("month", "Target Month")
    demand = forecast_result.get("predicted_demand", 0.0)
    supply = forecast_result.get("predicted_supply", 0.0)
    gap = forecast_result.get("gap", demand - supply)
    risk = forecast_result.get("risk_level", calculate_risk_level(gap))
    
    if risk == "Low":
        query = (
            f"Tamil Nadu electricity demand supply planning for {month}. "
            f"Forecasted demand {demand:,.2f} MU, supply {supply:,.2f} MU, gap {gap:,.2f} MU (Low Risk < 3000 MU). "
            f"Measures for baseline grid stability, seasonal thermal maintenance scheduling, renewable energy integration, "
            f"resource adequacy planning, and demand-side management in Tamil Nadu."
        )
    elif risk == "Moderate":
        query = (
            f"Tamil Nadu electricity grid adequacy and power procurement for {month}. "
            f"Forecasted demand {demand:,.2f} MU, supply {supply:,.2f} MU, gap {gap:,.2f} MU (Moderate Risk 3000-4500 MU). "
            f"Strategies for short-term capacity expansion, peak load management, thermal hydro generation scheduling, "
            f"power purchase agreements (PPA), and demand-response mechanisms in Tamil Nadu."
        )
    else:  # High Risk (> 4500 MU)
        query = (
            f"Tamil Nadu power deficit crisis mitigation and emergency energy planning for {month}. "
            f"Forecasted demand {demand:,.2f} MU, supply {supply:,.2f} MU, gap {gap:,.2f} MU (High Risk > 4500 MU). "
            f"Emergency load management protocols, inter-state grid power imports, fast-ramping generation deployment, "
            f"industrial demand control, and critical supply adequacy interventions in Tamil Nadu."
        )
        
    return query


class RAGForecastRetriever:
    """
    RAG Retrieval Engine that loads FAISS index, chunk metadata, and embedding model
    to perform dynamic forecast-driven semantic search.
    """
    def __init__(self, vector_db_dir: Path):
        self.vector_db_dir = vector_db_dir
        self.faiss_index_path = vector_db_dir / "energy_knowledge_base.faiss"
        self.metadata_path = vector_db_dir / "chunk_metadata.json"
        
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index file missing at: {self.faiss_index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Chunk metadata file missing at: {self.metadata_path}")
            
        logger.info(f"Loading FAISS index from {self.faiss_index_path.name}...")
        self.index = faiss.read_index(str(self.faiss_index_path))
        
        logger.info(f"Loading chunk metadata from {self.metadata_path.name}...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        logger.info(f"Loading embedding model: {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME)
        
        assert self.index.ntotal == len(self.metadata), "FAISS index size mismatch with metadata records!"
        logger.info(f"RAG Retriever ready with {self.index.ntotal} indexed chunks.")

    def retrieve(self, forecast_result: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """
        Dynamically constructs query from forecast result, embeds query, and retrieves Top-K chunks.
        """
        # 1. Build Query
        query_str = build_forecast_query(forecast_result)
        
        # 2. Embed & Normalize Query Vector
        query_vec = self.model.encode([query_str], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        faiss.normalize_L2(query_vec)
        
        # 3. FAISS Inner Product Similarity Search
        scores, indices = self.index.search(query_vec, top_k)
        
        # 4. Extract Top-K Results & Metadata
        results = []
        for rank in range(top_k):
            idx = indices[0][rank]
            score = float(scores[0][rank])
            chunk = self.metadata[idx]
            
            results.append({
                "rank": rank + 1,
                "similarity_score": round(score, 4),
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "category": chunk["category"],
                "page": chunk["page"],
                "pages": chunk["pages"],
                "text": chunk["text"]
            })
            
        return {
            "forecast_input": forecast_result,
            "generated_query": query_str,
            "top_k": top_k,
            "retrieved_chunks": results
        }


def run_forecast_retrieval_tests(vector_db_dir: Path) -> Dict[str, Any]:
    """
    Executes retrieval tests across Low, Moderate, and High risk forecasting scenarios.
    """
    retriever = RAGForecastRetriever(vector_db_dir)
    
    scenarios = [
        {
            "name": "Scenario 1: Low Risk (< 3000 MU)",
            "forecast": {
                "month": "April 2026",
                "predicted_demand": 14200.00,
                "predicted_supply": 12500.00,
                "gap": 1700.00,
                "risk_level": "Low"
            },
            "is_test_data": True
        },
        {
            "name": "Scenario 2: Moderate Risk (3000-4500 MU) [Primary Test]",
            "forecast": {
                "month": "February 2026",
                "predicted_demand": 15500.00,
                "predicted_supply": 12421.70,
                "gap": 3078.30,
                "risk_level": "Moderate"
            },
            "is_test_data": False
        },
        {
            "name": "Scenario 3: High Risk (> 4500 MU)",
            "forecast": {
                "month": "May 2026",
                "predicted_demand": 17800.00,
                "predicted_supply": 12600.00,
                "gap": 5200.00,
                "risk_level": "High"
            },
            "is_test_data": True
        }
    ]
    
    test_outputs = []
    
    for sc in scenarios:
        res = retriever.retrieve(sc["forecast"], top_k=5)
        test_outputs.append({
            "name": sc["name"],
            "is_test_data": sc["is_test_data"],
            "forecast": sc["forecast"],
            "query": res["generated_query"],
            "results": res["retrieved_chunks"]
        })
        
    return {
        "test_outputs": test_outputs
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    vdb_directory = base_dir / "RAG" / "vector_db"
    
    test_run = run_forecast_retrieval_tests(vdb_directory)
    
    for test in test_run["test_outputs"]:
        lbl = " [TEST DATA]" if test["is_test_data"] else ""
        print("\n" + "=" * 90)
        print(f"{test['name'].upper()}{lbl}")
        print("=" * 90)
        print(f"Month: {test['forecast']['month']} | Demand: {test['forecast']['predicted_demand']:,.2f} MU | Supply: {test['forecast']['predicted_supply']:,.2f} MU | Gap: {test['forecast']['gap']:,.2f} MU | Risk: {test['forecast']['risk_level']}")
        print("-" * 90)
        print(f"GENERATED QUERY:\n{test['query']}")
        print("-" * 90)
        print("TOP 5 RETRIEVED DOCUMENT CHUNKS:")
        print("-" * 90)
        for r in test["results"]:
            clean_snip = safe_ascii(r['text'][:220].replace('\n', ' '))
            print(f"Rank: #{r['rank']} | Similarity: {r['similarity_score']:.4f} | Source: {r['source']} | Category: [{r['category']}] | Page: {r['page']}")
            print(f"  Chunk ID : {r['chunk_id']}")
            print(f"  Text     : {clean_snip}...")
            print("-" * 90)
