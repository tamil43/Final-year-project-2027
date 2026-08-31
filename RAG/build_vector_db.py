"""
RAG Knowledge Base - Vector Embeddings & FAISS Database Creation (Step 4)
-------------------------------------------------------------------------
This module loads RAG/processed/chunks.json, generates 384-dimensional dense embeddings
using sentence-transformers/all-MiniLM-L6-v2, L2-normalizes the vectors for cosine similarity,
builds an IndexFlatIP FAISS vector index, serializes the database and metadata mappings to
RAG/vector_db/, and performs a basic sanity retrieval test.
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_VectorDB")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def safe_print_text(text: str) -> str:
    """Safely cleans text for Windows console printing without charmap encoding crashes."""
    return text.encode("ascii", "replace").decode("ascii")


def build_faiss_vector_database(
    chunks_file: Path,
    vector_db_dir: Path
) -> Dict[str, Any]:
    """
    Encodes text chunks, builds FAISS IndexFlatIP index, and saves index + metadata.
    """
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
    logger.info(f"Loading chunks dataset from {chunks_file.resolve()}")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
        
    num_chunks = len(chunks_data)
    logger.info(f"Loaded {num_chunks} chunk records.")
    
    # 1. Load Embedding Model
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    start_time = time.time()
    model = SentenceTransformer(MODEL_NAME)
    
    # 2. Extract Text Array
    texts = [c["text"] for c in chunks_data]
    
    # 3. Batch Encode & Normalize Embeddings
    logger.info(f"Generating embeddings for {num_chunks} chunks in batch mode...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    
    encode_time = time.time() - start_time
    logger.info(f"Embeddings generated in {encode_time:.2f} seconds. Shape: {embeddings.shape}")
    
    assert embeddings.shape[0] == num_chunks, f"Mismatch: {embeddings.shape[0]} vs {num_chunks}"
    assert embeddings.shape[1] == EMBEDDING_DIM, f"Unexpected dim: {embeddings.shape[1]}"
    
    # Ensure L2 normalization for Cosine Similarity via Inner Product
    faiss.normalize_L2(embeddings)
    
    # 4. Build FAISS IndexFlatIP
    logger.info("Initializing FAISS IndexFlatIP (Cosine Similarity)...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    
    assert index.ntotal == num_chunks, f"FAISS index size mismatch: {index.ntotal} vs {num_chunks}"
    logger.info(f"FAISS index successfully built. Total indexed vectors: {index.ntotal}")
    
    # 5. Save Vector DB Files
    vector_db_dir.mkdir(parents=True, exist_ok=True)
    
    faiss_index_path = vector_db_dir / "energy_knowledge_base.faiss"
    metadata_path = vector_db_dir / "chunk_metadata.json"
    config_path = vector_db_dir / "embedding_config.json"
    
    faiss.write_index(index, str(faiss_index_path))
    logger.info(f"Saved FAISS index to {faiss_index_path.resolve()}")
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved chunk metadata mapping to {metadata_path.resolve()}")
    
    config_payload = {
        "model_name": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIM,
        "similarity_metric": "Cosine Similarity (IndexFlatIP with L2 Normalization)",
        "normalization_method": "L2 Unit Length Normalization",
        "total_chunks_indexed": index.ntotal,
        "faiss_index_file": faiss_index_path.name,
        "metadata_file": metadata_path.name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved embedding configuration to {config_path.resolve()}")
    
    # 6. Basic Sanity Retrieval Test
    test_query = "What are the measures and planning considerations for maintaining electricity supply adequacy?"
    logger.info(f"Running sanity retrieval test with query: '{test_query}'")
    
    query_vector = model.encode([test_query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    faiss.normalize_L2(query_vector)
    
    k = 5
    scores, indices = index.search(query_vector, k)
    
    sanity_results = []
    for rank in range(k):
        idx = indices[0][rank]
        score = float(scores[0][rank])
        matched_chunk = chunks_data[idx]
        sanity_results.append({
            "rank": rank + 1,
            "similarity_score": round(score, 4),
            "chunk_id": matched_chunk["chunk_id"],
            "source": matched_chunk["source"],
            "category": matched_chunk["category"],
            "page": matched_chunk["page"],
            "pages": matched_chunk["pages"],
            "preview_text": matched_chunk["text"][:250].replace("\n", " ") + "..."
        })
        
    return {
        "model_name": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIM,
        "total_chunks_indexed": index.ntotal,
        "faiss_index_size_bytes": faiss_index_path.stat().st_size,
        "faiss_index_path": str(faiss_index_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "config_path": str(config_path.resolve()),
        "sanity_query": test_query,
        "sanity_results": sanity_results
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    chunks_json = base_dir / "RAG" / "processed" / "chunks.json"
    vdb_dir = base_dir / "RAG" / "vector_db"
    
    report = build_faiss_vector_database(chunks_json, vdb_dir)
    
    print("\n" + "=" * 90)
    print("STEP 4: VECTOR EMBEDDINGS & FAISS VECTOR DATABASE REPORT")
    print("=" * 90)
    print(f"Embedding Model Used     : {report['model_name']}")
    print(f"Embedding Dimension      : {report['embedding_dimension']}")
    print(f"Total Chunks Indexed     : {report['total_chunks_indexed']}")
    print(f"FAISS Index File Size    : {report['faiss_index_size_bytes'] / 1024:.2f} KB")
    print(f"FAISS Index Path         : {report['faiss_index_path']}")
    print(f"Metadata Mapping Path    : {report['metadata_path']}")
    print("=" * 90)
    print(f"\nSANITY TEST RETRIEVAL RESULTS (Query: '{report['sanity_query']}')")
    print("-" * 90)
    for res in report["sanity_results"]:
        safe_snippet = safe_print_text(res['preview_text'])
        print(f"Rank #{res['rank']} | Score: {res['similarity_score']:.4f} | Category: [{res['category']}] | Source: {res['source']} (Page {res['page']})")
        print(f"  Chunk ID : {res['chunk_id']}")
        print(f"  Snippet  : {safe_snippet}")
        print("-" * 90)
