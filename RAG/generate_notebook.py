"""
Generates the complete 6-Stage RAG pipeline notebook: RAG/rag_pipeline.ipynb
Covering PDF Extraction, Cleaning, Chunking, FAISS Vector Indexing, Forecast Retrieval, and Gemini LLM Grounded Recommendations.
"""

import nbformat as nbf
from pathlib import Path

def create_rag_notebook():
    nb = nbf.v4.new_notebook()
    
    # Notebook Title & Overview
    cell1 = nbf.v4.new_markdown_cell("""# Tamil Nadu Electricity Demand-Supply Forecasting — RAG Pipeline
## Complete 6-Stage Modular Implementation

This notebook implements the complete Retrieval-Augmented Generation (RAG) system for electricity policy and resource adequacy planning in Tamil Nadu.

### End-to-End RAG Architecture:
1. **Stage 1**: Document Ingestion & Page-Level PDF Text Extraction
2. **Stage 2**: Page-Level Deterministic Text Cleaning & Quality Diagnostics
3. **Stage 3**: Token-Aware Semantic Text Chunking & Lineage Tracking
4. **Stage 4**: Vector Embedding Generation & FAISS Vector Database Creation
5. **Stage 5**: Dynamic Forecast-Based Query Construction & Similarity Retrieval
6. **Stage 6**: Gemini LLM Integration & Grounded Energy Recommendation Generation

---""")

    # Stage 1 Section
    cell_s1_hdr = nbf.v4.new_markdown_cell("""---
## Stage 1: Document Ingestion & Page-Level PDF Text Extraction""")

    cell2 = nbf.v4.new_code_cell("""import os
import json
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from pypdf import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_Pipeline")

# Define repository & data paths
BASE_DIR = Path.cwd()
KB_DIR = BASE_DIR / "knowledge_base" if (BASE_DIR / "knowledge_base").exists() else BASE_DIR / "RAG" / "knowledge_base"
PROCESSED_DIR = BASE_DIR / "processed" if (BASE_DIR / "processed").exists() else BASE_DIR / "RAG" / "processed"
VECTOR_DB_DIR = BASE_DIR / "vector_db" if (BASE_DIR / "vector_db").exists() else BASE_DIR / "RAG" / "vector_db"

PAGES_OUTPUT_FILE = PROCESSED_DIR / "extracted_pages.json"
CLEANED_PAGES_FILE = PROCESSED_DIR / "cleaned_pages.json"
CHUNKS_OUTPUT_FILE = PROCESSED_DIR / "chunks.json"
PREVIEW_TXT_FILE = PROCESSED_DIR / "chunks_preview.txt"

FAISS_INDEX_FILE = VECTOR_DB_DIR / "energy_knowledge_base.faiss"
METADATA_MAPPING_FILE = VECTOR_DB_DIR / "chunk_metadata.json"
CONFIG_FILE = VECTOR_DB_DIR / "embedding_config.json"

# Load environment variables from .env
load_dotenv(dotenv_path=BASE_DIR / ".env")

print(f"Knowledge Base Path : {KB_DIR.resolve()}")
print(f"Processed Dir Path  : {PROCESSED_DIR.resolve()}")
print(f"Vector DB Path      : {VECTOR_DB_DIR.resolve()}")""")

    cell3 = nbf.v4.new_markdown_cell("""### 1.1 Knowledge Base Discovery""")
    cell4 = nbf.v4.new_code_cell("""def discover_pdf_documents(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {base_dir}")
    return sorted(
        list(base_dir.rglob("*.pdf")),
        key=lambda p: (p.parent.name.lower(), p.name.lower())
    )

discovered_pdfs = discover_pdf_documents(KB_DIR)
print(f"Total PDF Documents Discovered: {len(discovered_pdfs)}\\n")
for idx, pdf in enumerate(discovered_pdfs, start=1):
    category = pdf.parent.name
    size_mb = pdf.stat().st_size / (1024 * 1024)
    print(f"  {idx}. [{category:<5}] {pdf.name:<45} ({size_mb:6.2f} MB)")""")

    cell5 = nbf.v4.new_markdown_cell("""### 1.2 Page-by-Page PDF Extraction""")
    cell6 = nbf.v4.new_code_cell("""def extract_pages_from_pdf(pdf_path: Path, min_char_threshold: int = 50) -> Dict[str, Any]:
    category = pdf_path.parent.name
    filename = pdf_path.name
    doc_records, empty_pages, short_pages, extraction_errors = [], [], [], []
    
    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
    except Exception as e:
        logger.error(f"Failed to open/parse PDF {filename}: {e}")
        return {
            "source": filename, "category": category, "total_pages": 0, "extracted_count": 0,
            "empty_pages": [], "short_pages": [], "extraction_errors": [{"page": 0, "error": str(e)}],
            "records": [], "total_chars": 0
        }
    
    total_chars = 0
    extracted_count = 0
    
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
            cleaned_text = raw_text.strip()
            char_len = len(cleaned_text)
            
            if char_len == 0:
                empty_pages.append(page_num)
            elif char_len < min_char_threshold:
                short_pages.append({"page": page_num, "char_count": char_len, "preview": cleaned_text[:40]})
                extracted_count += 1
            else:
                extracted_count += 1
                
            total_chars += char_len
            doc_records.append({"source": filename, "category": category, "page": page_num, "text": cleaned_text})
        except Exception as pe:
            logger.warning(f"Error extracting text from {filename} page {page_num}: {pe}")
            extraction_errors.append({"page": page_num, "error": str(pe)})
            doc_records.append({"source": filename, "category": category, "page": page_num, "text": ""})
            
    return {
        "source": filename, "category": category, "total_pages": total_pages,
        "extracted_count": extracted_count, "empty_pages": empty_pages,
        "short_pages": short_pages, "extraction_errors": extraction_errors,
        "records": doc_records, "total_chars": total_chars
    }""")

    cell7 = nbf.v4.new_markdown_cell("""### 1.3 Stage 1 Execution & Output Generation""")
    cell8 = nbf.v4.new_code_cell("""all_page_records = []
doc_summaries = []

for pdf_path in discovered_pdfs:
    res = extract_pages_from_pdf(pdf_path, min_char_threshold=50)
    all_page_records.extend(res["records"])
    doc_summaries.append({
        "source": res["source"], "category": res["category"], "total_pages": res["total_pages"],
        "extracted_pages": res["extracted_count"], "empty_pages_count": len(res["empty_pages"]),
        "short_pages_count": len(res["short_pages"]), "errors_count": len(res["extraction_errors"]),
        "total_chars": res["total_chars"],
        "avg_chars_per_page": round(res["total_chars"] / max(res["total_pages"], 1), 1)
    })

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
with open(PAGES_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_page_records, f, indent=2, ensure_ascii=False)
print(f"Saved {len(all_page_records)} page records to {PAGES_OUTPUT_FILE.name}")""")

    # Stage 2 Section
    cell_s2_hdr = nbf.v4.new_markdown_cell("""---
## Stage 2: Page-Level Text Cleaning & Quality Verification""")

    cell_s2_code1 = nbf.v4.new_code_cell("""def clean_page_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text
    text = re.sub(r'[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f]', ' ', text)
    text = re.sub(r'([a-zA-Z]{2,})-\\s*\\n\\s*([a-zA-Z]{2,})', r'\\1\\2', text)
    text = text.replace('\\ufffd', ' ')
    
    raw_lines = [line.strip() for line in text.split('\\n')]
    cleaned_paragraphs = []
    current_para = []
    
    for line in raw_lines:
        if not line:
            if current_para:
                cleaned_paragraphs.append(" ".join(current_para))
                current_para = []
            continue
            
        is_heading_or_list = bool(
            re.match(r'^(?:\\d+(?:\\.\\d+)*|\\[[a-zA-Z0-9]+\\]|[•\\-\\*\u2022])\\s+', line) or
            re.match(r'^(?:SECTION|CHAPTER|TABLE|ANNEXURE|DISCLAIMER|CONTENTS|EXECUTIVE SUMMARY|INTRODUCTION)\\b', line, re.I)
        )
        
        if is_heading_or_list and current_para:
            cleaned_paragraphs.append(" ".join(current_para))
            current_para = [line]
        else:
            current_para.append(line)
            
    if current_para:
        cleaned_paragraphs.append(" ".join(current_para))
        
    cleaned_text = "\\n\\n".join(cleaned_paragraphs)
    lines = [re.sub(r'[ \\t]+', ' ', l).strip() for l in cleaned_text.split('\\n')]
    cleaned_text = "\\n".join(lines)
    return re.sub(r'\\n{3,}', '\\n\\n', cleaned_text).strip()""")

    cell_s2_exec = nbf.v4.new_code_cell("""cleaned_records = []
for item in all_page_records:
    raw_txt = item.get("text", "")
    clean_txt = clean_page_text(raw_txt)
    record = {
        "source": item["source"],
        "category": item["category"],
        "page": item["page"],
        "text": raw_txt,
        "cleaned_text": clean_txt
    }
    cleaned_records.append(record)

with open(CLEANED_PAGES_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_records, f, indent=2, ensure_ascii=False)
print(f"Saved {len(cleaned_records)} cleaned page records to {CLEANED_PAGES_FILE.name}")""")

    # Stage 3 Section
    cell_s3_hdr = nbf.v4.new_markdown_cell("""---
## Stage 3: Token-Aware Semantic Text Chunking""")

    cell_s3_code1 = nbf.v4.new_code_cell("""try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)


def split_text_into_semantic_units(text: str) -> List[str]:
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split('\\n\\n') if p.strip()]
    units = []
    sentence_endings = re.compile(r'(?<=[.!?])\\s+')
    for para in paragraphs:
        if count_tokens(para) <= 400:
            units.append(para)
        else:
            sentences = sentence_endings.split(para)
            for s in sentences:
                s_clean = s.strip()
                if s_clean:
                    units.append(s_clean)
    return units


def create_semantic_chunks_for_document(
    doc_pages: List[Dict[str, Any]],
    target_tokens: int = 900,
    max_tokens: int = 1100,
    overlap_tokens: int = 120
) -> List[Dict[str, Any]]:
    if not doc_pages:
        return []
    
    source = doc_pages[0]["source"]
    category = doc_pages[0]["category"]
    clean_stem = re.sub(r'[^a-zA-Z0-9]', '_', Path(source).stem)
    
    atomic_units: List[Tuple[str, int]] = []
    for p_rec in doc_pages:
        pg_num = p_rec["page"]
        pg_text = p_rec.get("cleaned_text", "").strip()
        if not pg_text:
            continue
        u_list = split_text_into_semantic_units(pg_text)
        for u in u_list:
            atomic_units.append((u, pg_num))
            
    if not atomic_units:
        return []
    
    chunks = []
    current_units: List[Tuple[str, int]] = []
    current_token_count = 0
    chunk_idx = 1
    
    for u_text, u_page in atomic_units:
        u_tokens = count_tokens(u_text)
        
        if u_tokens > max_tokens:
            sub_clauses = re.split(r'(?<=[;,])\\s+', u_text)
            for sc in sub_clauses:
                sc_tokens = count_tokens(sc)
                if current_token_count + sc_tokens > target_tokens and current_units:
                    chunk_rec = build_chunk_record(current_units, source, category, clean_stem, chunk_idx, overlap_tokens)
                    chunks.append(chunk_rec["chunk"])
                    chunk_idx += 1
                    current_units = chunk_rec["overlap_units"]
                    current_token_count = sum(count_tokens(txt) for txt, _ in current_units)
                current_units.append((sc, u_page))
                current_token_count += sc_tokens
            continue
            
        if current_token_count + u_tokens > target_tokens and current_units:
            chunk_rec = build_chunk_record(current_units, source, category, clean_stem, chunk_idx, overlap_tokens)
            chunks.append(chunk_rec["chunk"])
            chunk_idx += 1
            current_units = chunk_rec["overlap_units"]
            current_token_count = sum(count_tokens(txt) for txt, _ in current_units)
            
        current_units.append((u_text, u_page))
        current_token_count += u_tokens
        
    if current_units:
        chunk_str = "\\n\\n".join(txt for txt, _ in current_units)
        t_cnt = count_tokens(chunk_str)
        pages_covered = sorted(list(set(pg for _, pg in current_units)))
        page_start = pages_covered[0]
        chunk_id = f"{category}_{clean_stem}_p{page_start}_c{chunk_idx}"
        chunks.append({
            "chunk_id": chunk_id,
            "source": source,
            "category": category,
            "page": page_start,
            "pages": pages_covered,
            "token_count": t_cnt,
            "char_count": len(chunk_str),
            "word_count": len(chunk_str.split()),
            "text": chunk_str
        })
        
    return chunks


def build_chunk_record(units: List[Tuple[str, int]], source: str, category: str, clean_stem: str, chunk_idx: int, overlap_tokens: int):
    chunk_str = "\\n\\n".join(txt for txt, _ in units)
    t_cnt = count_tokens(chunk_str)
    pages_covered = sorted(list(set(pg for _, pg in units)))
    page_start = pages_covered[0]
    chunk_id = f"{category}_{clean_stem}_p{page_start}_c{chunk_idx}"
    
    chunk_dict = {
        "chunk_id": chunk_id,
        "source": source,
        "category": category,
        "page": page_start,
        "pages": pages_covered,
        "token_count": t_cnt,
        "char_count": len(chunk_str),
        "word_count": len(chunk_str.split()),
        "text": chunk_str
    }
    
    overlap_units = []
    accum_tokens = 0
    for txt, pg in reversed(units):
        tk = count_tokens(txt)
        if accum_tokens + tk <= overlap_tokens:
            overlap_units.insert(0, (txt, pg))
            accum_tokens += tk
        else:
            break
            
    return {"chunk": chunk_dict, "overlap_units": overlap_units}""")

    cell_s3_exec = nbf.v4.new_code_cell("""docs_map = {}
for p in cleaned_records:
    docs_map.setdefault(p["source"], []).append(p)

all_chunks = []
for src, p_list in docs_map.items():
    p_list_sorted = sorted(p_list, key=lambda x: x["page"])
    doc_chunks = create_semantic_chunks_for_document(p_list_sorted, target_tokens=900, overlap_tokens=120)
    all_chunks.extend(doc_chunks)

with open(CHUNKS_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)
print(f"Generated and saved {len(all_chunks)} chunks to {CHUNKS_OUTPUT_FILE.name}")""")

    # Stage 4 Section
    cell_s4_hdr = nbf.v4.new_markdown_cell("""---
## Stage 4: Vector Embedding Generation & FAISS Vector Database Creation""")

    cell_s4_code = nbf.v4.new_code_cell("""MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

model = SentenceTransformer(MODEL_NAME)
texts = [c["text"] for c in all_chunks]

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=False,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype(np.float32)

faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(embeddings)

VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(FAISS_INDEX_FILE))

with open(METADATA_MAPPING_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"FAISS Index created with {index.ntotal} vectors saved to {FAISS_INDEX_FILE.name}")""")

    # Stage 5 Section
    cell_s5_hdr = nbf.v4.new_markdown_cell("""---
## Stage 5: Dynamic Query Construction & Forecast-Based Retrieval""")

    cell_s5_code1 = nbf.v4.new_code_cell("""def calculate_risk_level(gap_mu: float) -> str:
    if gap_mu < 3000.0:
        return "Low"
    elif 3000.0 <= gap_mu <= 4500.0:
        return "Moderate"
    else:
        return "High"


def build_forecast_query(forecast_result: Dict[str, Any]) -> str:
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
    else:
        query = (
            f"Tamil Nadu power deficit crisis mitigation and emergency energy planning for {month}. "
            f"Forecasted demand {demand:,.2f} MU, supply {supply:,.2f} MU, gap {gap:,.2f} MU (High Risk > 4500 MU). "
            f"Emergency load management protocols, inter-state grid power imports, fast-ramping generation deployment, "
            f"industrial demand control, and critical supply adequacy interventions in Tamil Nadu."
        )
    return query


def retrieve_relevant_chunks(forecast_result: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    query_str = build_forecast_query(forecast_result)
    
    query_vec = model.encode([query_str], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    faiss.normalize_L2(query_vec)
    
    scores, indices = index.search(query_vec, top_k)
    
    results = []
    for rank in range(top_k):
        idx = indices[0][rank]
        score = float(scores[0][rank])
        chunk = all_chunks[idx]
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
        "generated_query": query_str,
        "retrieved_chunks": results
    }""")

    # Stage 6 Section
    cell_s6_hdr = nbf.v4.new_markdown_cell("""---
## Stage 6: Gemini LLM Integration & Grounded Energy Recommendation

In Stage 6, we integrate the **Google Gemini LLM** to synthesize grounded energy planning recommendations based on the Top-K retrieved chunks from Stage 5.

### Strict Groundedness Rules:
1. **Zero Hallucination**: Every recommended measure MUST be anchored in retrieved Evidence Blocks.
2. **Exact Citations**: Inline citations format `[Document Name, Page N]`.
3. **Tamil Nadu Prioritization**: Tamil Nadu (`Tn`) evidence takes precedence for state operational decisions.
4. **Insufficient Evidence Clause**: Explicitly flags any policy area lacking retrieved context.""")

    cell_s6_code1 = nbf.v4.new_code_cell("""from generate_recommendation import generate_recommendation

# February 2026 Primary Test Case (Demand: 15,500.00 MU, Supply: 12,421.70 MU, Gap: 3,078.30 MU, Risk: Moderate)
feb_forecast = {
    "month": "February 2026",
    "predicted_demand": 15500.00,
    "predicted_supply": 12421.70,
    "gap": 3078.30,
    "risk_level": "Moderate"
}

retrieved_output = retrieve_relevant_chunks(feb_forecast, top_k=5)
rec_result = generate_recommendation(feb_forecast, retrieved_output["retrieved_chunks"])

print("=" * 90)
print("STAGE 6: GROUNDED ENERGY RECOMMENDATION RESULTS")
print("=" * 90)
print(f"Forecast Target     : {feb_forecast['month']}")
print(f"Predicted Gap       : {feb_forecast['gap']:,.2f} MU")
print(f"Risk Classification : {feb_forecast['risk_level']} Risk")
print(f"Execution Mode      : {rec_result['execution_mode']}")
print("=" * 90)
print("\\n" + rec_result["recommendation"])
print("=" * 90)""")

    nb.cells = [
        cell1, cell_s1_hdr, cell2, cell3, cell4, cell5, cell6, cell7, cell8,
        cell_s2_hdr, cell_s2_code1, cell_s2_exec,
        cell_s3_hdr, cell_s3_code1, cell_s3_exec,
        cell_s4_hdr, cell_s4_code,
        cell_s5_hdr, cell_s5_code1,
        cell_s6_hdr, cell_s6_code1
    ]
    
    target_path = Path("d:/Final Year Project 2027/RAG/rag_pipeline.ipynb")
    with open(target_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Updated notebook at {target_path}")

if __name__ == "__main__":
    create_rag_notebook()
