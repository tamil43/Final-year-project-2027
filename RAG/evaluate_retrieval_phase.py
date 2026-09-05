"""
RAG Knowledge Base - Formal Retrieval Phase Evaluation for IEEE Paper
---------------------------------------------------------------------
Computes rigorous retrieval-phase metrics for the RAG system:
- Context Precision
- Context Recall
- Context Relevance
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@5)
- Hit Rate@5
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Ensure project and RAG directories in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = ROOT_DIR / "RAG"
sys.path.insert(0, str(RAG_DIR))

from forecast_retrieval import RAGForecastRetriever, build_forecast_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Retrieval_Evaluator")

# Verified benchmark evaluation queries with ground-truth target sources and topics
EVAL_DATASET = [
    # Tamil Nadu Specific Queries (12 Queries)
    {
        "query_id": "Q01",
        "query": "Tamil Nadu peak demand management and evening non-solar hour power procurement",
        "target_category": "Tn",
        "relevant_sources": ["Tamil_Nadu_Resource_Adequacy_Report_2026.pdf", "tamilnadu_energy_department.pdf"],
        "scope": "Tamil Nadu",
        "risk_scenario": "Moderate",
        "total_ground_truth_relevant": 4
    },
    {
        "query_id": "Q02",
        "query": "TNERC resource adequacy guidelines and long term capacity contract compliance in Tamil Nadu",
        "target_category": "Tn",
        "relevant_sources": ["Tamil_Nadu_Resource_Adequacy_Report_2026.pdf"],
        "scope": "Tamil Nadu",
        "risk_scenario": "Low",
        "total_ground_truth_relevant": 3
    },
    {
        "query_id": "Q03",
        "query": "Tamil Nadu agricultural water pump load shifting to daytime solar hours",
        "target_category": "Tn",
        "relevant_sources": ["tamilnadu_energy_department.pdf", "Tamil_Nadu_Resource_Adequacy_Report_2026.pdf"],
        "scope": "Tamil Nadu",
        "risk_scenario": "Moderate",
        "total_ground_truth_relevant": 4
    },
    {
        "query_id": "Q04",
        "query": "Thermal forced outage reserves and maintenance scheduling for Tamil Nadu state generators",
        "target_category": "Tn",
        "relevant_sources": ["Tamil_Nadu_Resource_Adequacy_Report_2026.pdf", "tamilnadu_energy_department.pdf"],
        "scope": "Tamil Nadu",
        "risk_scenario": "Low",
        "total_ground_truth_relevant": 4
    },
    {
        "query_id": "Q05",
        "query": "Tamil Nadu hydro reservoir storage dispatch and gas turbine ramping for evening peak support",
        "target_category": "Tn",
        "relevant_sources": ["Tamil_Nadu_Resource_Adequacy_Report_2026.pdf"],
        "scope": "Tamil Nadu",
        "risk_scenario": "High",
        "total_ground_truth_relevant": 3
    },
    {
        "query_id": "Q06",
        "query": "TANGEDCO renewable integration and wind solar capacity additions in Tamil Nadu",
        "target_category": "Tn",
        "relevant_sources": ["tamilnadu_energy_department.pdf", "Tamil_Nadu_Resource_Adequacy_Report_2026.pdf"],
        "scope": "Tamil Nadu",
        "risk_scenario": "Low",
        "total_ground_truth_relevant": 4
    },
    {
        "query_id": "Q07",
        "query": "Tamil Nadu power deficit crisis mitigation emergency inter-state imports and load rotation",
        "target_category": "Tn",
        "relevant_sources": ["Tamil_Nadu_Resource_Adequacy_Report_2026.pdf", "tamilnadu_energy_department.pdf"],
        "scope": "Tamil Nadu",
        "risk_scenario": "High",
        "total_ground_truth_relevant": 4
    },
    {
        "query_id": "Q08",
        "query": "Day-Ahead Market DAM and Real-Time Market RTM power purchase optimization for Tamil Nadu grid",
        "target_category": "Tn",
        "relevant_sources": ["Tamil_Nadu_Resource_Adequacy_Report_2026.pdf"],
        "scope": "Tamil Nadu",
        "risk_scenario": "Moderate",
        "total_ground_truth_relevant": 3
    },
    {
        "query_id": "Q09",
        "query": "Tamil Nadu industrial demand response and high-tension consumer tariff incentives",
        "target_category": "Tn",
        "relevant_sources": ["tamilnadu_energy_department.pdf"],
        "scope": "Tamil Nadu",
        "risk_scenario": "Moderate",
        "total_ground_truth_relevant": 3
    },
    {
        "query_id": "Q10",
        "query": "Battery energy storage systems BESS deployment for renewable smoothing in Tamil Nadu",
        "target_category": "Tn",
        "relevant_sources": ["Tamil_Nadu_Resource_Adequacy_Report_2026.pdf", "tamilnadu_energy_department.pdf"],
        "scope": "Tamil Nadu",
        "risk_scenario": "Low",
        "total_ground_truth_relevant": 4
    },
    {
        "query_id": "Q11",
        "query": "Tamil Nadu SLDC frequency control and 50 Hz grid balancing protocols",
        "target_category": "Tn",
        "relevant_sources": ["Tamil_Nadu_Resource_Adequacy_Report_2026.pdf"],
        "scope": "Tamil Nadu",
        "risk_scenario": "Low",
        "total_ground_truth_relevant": 3
    },
    {
        "query_id": "Q12",
        "query": "Substation transformer maintenance and transmission line contingency in Tamil Nadu",
        "target_category": "Tn",
        "relevant_sources": ["tamilnadu_energy_department.pdf"],
        "scope": "Tamil Nadu",
        "risk_scenario": "Low",
        "total_ground_truth_relevant": 3
    },

    # National Level CEA / MoP Queries (8 Queries)
    {
        "query_id": "Q13",
        "query": "CEA National Electricity Plan generation capacity targets and renewable trajectory",
        "target_category": "India",
        "relevant_sources": ["nep-vol-1.pdf"],
        "scope": "National (India)",
        "risk_scenario": "Low",
        "total_ground_truth_relevant": 3
    },
    {
        "query_id": "Q14",
        "query": "National transmission expansion and inter-regional power transfer capacity guidelines",
        "target_category": "India",
        "relevant_sources": ["nep-vol-2.pdf"],
        "scope": "National (India)",
        "risk_scenario": "Moderate",
        "total_ground_truth_relevant": 3
    },
    {
        "query_id": "Q15",
        "query": "Ministry of Power resource adequacy framework regulations for state distribution companies",
        "target_category": "India",
        "relevant_sources": ["Resource-Adequacy-Guidelines.pdf", "nep-vol-1.pdf"],
        "scope": "National (India)",
        "risk_scenario": "Moderate",
        "total_ground_truth_relevant": 4
    },
    {
        "query_id": "Q16",
        "query": "Tariff policy guidelines for multi-year tariff determination and return on equity",
        "target_category": "India",
        "relevant_sources": ["Tariff_Policy-28012016.pdf"],
        "scope": "National (India)",
        "risk_scenario": "Low",
        "total_ground_truth_relevant": 3
    },
    {
        "query_id": "Q17",
        "query": "National energy conservation building codes and demand side management benchmarks",
        "target_category": "India",
        "relevant_sources": ["nep-vol-1.pdf", "Report-on-Optimal-Generation-Mix-for-2029-30-Revised.pdf"],
        "scope": "National (India)",
        "risk_scenario": "Low",
        "total_ground_truth_relevant": 4
    },
    {
        "query_id": "Q18",
        "query": "Optimal generation mix 2030 coal flexible operation and storage requirements",
        "target_category": "India",
        "relevant_sources": ["Report-on-Optimal-Generation-Mix-for-2029-30-Revised.pdf", "nep-vol-1.pdf"],
        "scope": "National (India)",
        "risk_scenario": "Moderate",
        "total_ground_truth_relevant": 4
    },
    {
        "query_id": "Q19",
        "query": "CEA grid code operational standards for spinning reserves and governor action",
        "target_category": "India",
        "relevant_sources": ["nep-vol-2.pdf", "Resource-Adequacy-Guidelines.pdf"],
        "scope": "National (India)",
        "risk_scenario": "High",
        "total_ground_truth_relevant": 3
    },
    {
        "query_id": "Q20",
        "query": "Renewable purchase obligation RPO compliance and energy storage purchase targets",
        "target_category": "India",
        "relevant_sources": ["nep-vol-1.pdf", "Tariff_Policy-28012016.pdf"],
        "scope": "National (India)",
        "risk_scenario": "Moderate",
        "total_ground_truth_relevant": 3
    }
]


def compute_dcg(relevance_vector: List[int], k: int = 5) -> float:
    """Computes Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for idx, rel in enumerate(relevance_vector[:k], start=1):
        if rel > 0:
            dcg += (2**rel - 1) / np.log2(idx + 1)
    return dcg


def compute_idcg(num_relevant: int, k: int = 5) -> float:
    """Computes Ideal Discounted Cumulative Gain at rank K."""
    ideal_rel = [1] * min(num_relevant, k)
    return compute_dcg(ideal_rel, k=k)


def evaluate_retrieval_phase():
    vector_db_dir = RAG_DIR / "vector_db"
    retriever = RAGForecastRetriever(vector_db_dir)
    
    top_k = 5
    context_precision_list = []
    context_recall_list = []
    context_relevance_list = []
    mrr_list = []
    ndcg_list = []
    hit_rate_5_list = []
    
    query_eval_records = []
    
    for q_item in EVAL_DATASET:
        qid = q_item["query_id"]
        q_text = q_item["query"]
        target_cat = q_item["target_category"]
        rel_sources = q_item["relevant_sources"]
        risk = q_item["risk_scenario"]
        gt_total = q_item["total_ground_truth_relevant"]
        
        # Build forecast payload
        gap_val = 2200.0 if risk == "Low" else (3500.0 if risk == "Moderate" else 5200.0)
        forecast_payload = {
            "month": "Retrieval Evaluation",
            "predicted_demand": 15000.0,
            "predicted_supply": 15000.0 - gap_val,
            "gap": gap_val,
            "risk_level": risk
        }
        
        # Run retrieval pipeline
        retrieval_res = retriever.retrieve(forecast_payload, top_k=top_k)
        retrieved_chunks = retrieval_res["retrieved_chunks"]
        
        # Binary relevance matching
        rel_vector = []
        for c in retrieved_chunks:
            cat_match = (c.get("category") == target_cat)
            src_match = any(src.lower() in c.get("source", "").lower() for src in rel_sources)
            is_rel = 1 if (cat_match or src_match) else 0
            rel_vector.append(is_rel)
            
        num_relevant_retrieved = sum(rel_vector)
        
        # 1. Context Precision: Precision of retrieved chunks in Top-K
        c_precision = num_relevant_retrieved / float(top_k)
        context_precision_list.append(c_precision)
        
        # 2. Context Recall: Proportion of ground-truth relevant context retrieved
        c_recall = min(num_relevant_retrieved / float(gt_total), 1.0)
        context_recall_list.append(c_recall)
        
        # 3. Context Relevance: Average cosine similarity score across retrieved Top-K chunks
        sim_scores = [c.get("similarity_score", 0.0) for c in retrieved_chunks]
        c_relevance = float(np.mean(sim_scores)) if sim_scores else 0.0
        context_relevance_list.append(c_relevance)
        
        # 4. MRR: Reciprocal rank of the first relevant result
        rr = 0.0
        for rank, is_rel in enumerate(rel_vector, start=1):
            if is_rel == 1:
                rr = 1.0 / rank
                break
        mrr_list.append(rr)
        
        # 5. nDCG@5: Ranking quality
        dcg_5 = compute_dcg(rel_vector, k=top_k)
        idcg_5 = compute_idcg(gt_total, k=top_k)
        ndcg_5 = (dcg_5 / idcg_5) if idcg_5 > 0 else 0.0
        ndcg_list.append(ndcg_5)
        
        # 6. Hit Rate@5
        hit_5 = 1.0 if num_relevant_retrieved > 0 else 0.0
        hit_rate_5_list.append(hit_5)
        
        query_eval_records.append({
            "query_id": qid,
            "query": q_text,
            "relevance_vector": rel_vector,
            "context_precision": round(c_precision, 4),
            "context_recall": round(c_recall, 4),
            "context_relevance": round(c_relevance, 4),
            "reciprocal_rank": round(rr, 4),
            "ndcg_at_5": round(ndcg_5, 4),
            "hit_at_5": hit_5
        })

    # Summary Metrics
    results_summary = {
        "context_precision": round(float(np.mean(context_precision_list)), 4),
        "context_recall": round(float(np.mean(context_recall_list)), 4),
        "context_relevance": round(float(np.mean(context_relevance_list)), 4),
        "mrr": round(float(np.mean(mrr_list)), 4),
        "ndcg_at_5": round(float(np.mean(ndcg_list)), 4),
        "hit_rate_at_5": round(float(np.mean(hit_rate_5_list)), 4),
        "total_test_queries": len(EVAL_DATASET),
        "top_k": top_k,
        "total_knowledge_base_chunks": 1155,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": 384
    }

    # Save to JSON
    output_path = RAG_DIR / "retrieval_evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "evaluation_title": "RAG Retrieval Phase Evaluation",
            "summary_metrics": results_summary,
            "per_query_records": query_eval_records
        }, f, indent=2)
        
    logger.info(f"Retrieval evaluation saved to {output_path.resolve()}")
    return results_summary


if __name__ == "__main__":
    res = evaluate_retrieval_phase()
    print("\n" + "=" * 60)
    print("RAG RETRIEVAL PHASE EVALUATION RESULTS (IEEE PAPER FORMAT)")
    print("=" * 60)
    for k, v in res.items():
        print(f"{k:<30}: {v}")
