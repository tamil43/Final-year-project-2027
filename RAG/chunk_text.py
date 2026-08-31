"""
RAG Knowledge Base - Semantic Text Chunking Pipeline (Step 3)
--------------------------------------------------------------
This script loads page-level cleaned records from RAG/processed/cleaned_pages.json,
applies token-aware sentence & paragraph hierarchical chunking (800-1000 tokens target,
100-150 tokens overlap), preserves source metadata and multi-page spanning tracking,
generates RAG/processed/chunks.json and RAG/processed/chunks_preview.txt, and reports metrics.
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Try importing tiktoken for exact token counting; fallback to word-ratio estimation if unavailable
try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_Chunker")


def split_text_into_semantic_units(text: str) -> List[str]:
    """
    Splits page text into paragraphs and sentences while keeping section titles intact.
    """
    if not text:
        return []
    
    # 1. Split into paragraphs by double newlines
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    units = []
    
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        if para_tokens <= 400:
            units.append(para)
        else:
            # Paragraph is long; split into sentences
            sentences = sentence_endings.split(para)
            for s in sentences:
                s_clean = s.strip()
                if s_clean:
                    units.append(s_clean)
                    
    return units


def create_semantic_chunks_for_document(
    doc_pages: List[Dict[str, Any]],
    target_tokens: int = 900,
    min_tokens: int = 150,
    max_tokens: int = 1100,
    overlap_tokens: int = 120
) -> List[Dict[str, Any]]:
    """
    Chunks a document's sequential pages into overlapping semantic windows.
    Tracks page boundaries and multi-page spans cleanly.
    """
    if not doc_pages:
        return []
    
    source = doc_pages[0]["source"]
    category = doc_pages[0]["category"]
    clean_stem = re.sub(r'[^a-zA-Z0-9]', '_', Path(source).stem)
    
    # Build a sequence of atomic units annotated with their page number
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
        
        # If single unit is excessively large (> max_tokens), handle by sentence/clause boundary
        if u_tokens > max_tokens:
            # Sub-split long unit
            sub_clauses = re.split(r'(?<=[;,])\s+', u_text)
            for sc in sub_clauses:
                sc_tokens = count_tokens(sc)
                if current_token_count + sc_tokens > target_tokens and current_units:
                    # Flush chunk
                    chunk_rec = build_chunk_record(
                        current_units, source, category, clean_stem, chunk_idx, overlap_tokens
                    )
                    chunks.append(chunk_rec["chunk"])
                    chunk_idx += 1
                    current_units = chunk_rec["overlap_units"]
                    current_token_count = sum(count_tokens(txt) for txt, _ in current_units)
                    
                current_units.append((sc, u_page))
                current_token_count += sc_tokens
            continue
            
        if current_token_count + u_tokens > target_tokens and current_units:
            # Flush chunk
            chunk_rec = build_chunk_record(
                current_units, source, category, clean_stem, chunk_idx, overlap_tokens
            )
            chunks.append(chunk_rec["chunk"])
            chunk_idx += 1
            current_units = chunk_rec["overlap_units"]
            current_token_count = sum(count_tokens(txt) for txt, _ in current_units)
            
        current_units.append((u_text, u_page))
        current_token_count += u_tokens
        
    # Flush trailing units
    if current_units:
        chunk_str = "\n\n".join(txt for txt, _ in current_units)
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


def build_chunk_record(
    units: List[Tuple[str, int]],
    source: str,
    category: str,
    clean_stem: str,
    chunk_idx: int,
    overlap_tokens: int
) -> Dict[str, Any]:
    """
    Constructs a chunk dictionary and identifies overlap units for next window.
    """
    chunk_str = "\n\n".join(txt for txt, _ in units)
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
    
    # Calculate overlap units from tail
    overlap_units = []
    accum_tokens = 0
    for txt, pg in reversed(units):
        tk = count_tokens(txt)
        if accum_tokens + tk <= overlap_tokens:
            overlap_units.insert(0, (txt, pg))
            accum_tokens += tk
        else:
            break
            
    return {
        "chunk": chunk_dict,
        "overlap_units": overlap_units
    }


def run_semantic_chunking_pipeline(
    input_file: Path,
    output_chunks_file: Path,
    preview_file: Path,
    target_tokens: int = 900,
    overlap_tokens: int = 120
) -> Dict[str, Any]:
    """
    Main pipeline to chunk cleaned_pages.json into chunks.json and chunks_preview.txt.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Cleaned pages file not found: {input_file}")
        
    with open(input_file, "r", encoding="utf-8") as f:
        cleaned_pages = json.load(f)
        
    logger.info(f"Loaded {len(cleaned_pages)} pages from {input_file.name}")
    
    # Group pages by source document
    docs_map: Dict[str, List[Dict[str, Any]]] = {}
    for p in cleaned_pages:
        src = p["source"]
        docs_map.setdefault(src, []).append(p)
        
    all_chunks: List[Dict[str, Any]] = []
    doc_chunk_counts: Dict[str, int] = {}
    cat_chunk_counts: Dict[str, int] = {"Tn": 0, "India": 0}
    
    for src, p_list in docs_map.items():
        # Sort pages chronologically
        p_list_sorted = sorted(p_list, key=lambda x: x["page"])
        doc_chunks = create_semantic_chunks_for_document(
            doc_pages=p_list_sorted,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens
        )
        all_chunks.extend(doc_chunks)
        doc_chunk_counts[src] = len(doc_chunks)
        cat = p_list_sorted[0]["category"]
        cat_chunk_counts[cat] = cat_chunk_counts.get(cat, 0) + len(doc_chunks)
        
    # Save output chunks.json
    output_chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    # Save preview sample text file
    with open(preview_file, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("RAG KNOWLEDGE BASE — SEMANTIC CHUNKS READABLE PREVIEW SAMPLE\n")
        f.write("=" * 90 + "\n\n")
        for chunk in all_chunks[:15]:  # First 15 preview chunks
            f.write(f"CHUNK ID   : {chunk['chunk_id']}\n")
            f.write(f"SOURCE     : {chunk['source']}\n")
            f.write(f"CATEGORY   : {chunk['category']}\n")
            f.write(f"PAGES      : {chunk['pages']} (Primary: {chunk['page']})\n")
            f.write(f"METRICS    : {chunk['token_count']} tokens | {chunk['char_count']} chars | {chunk['word_count']} words\n")
            f.write("-" * 90 + "\n")
            f.write(chunk["text"] + "\n")
            f.write("=" * 90 + "\n\n")
            
    # Metrics
    token_lengths = [c["token_count"] for c in all_chunks]
    char_lengths = [c["char_count"] for c in all_chunks]
    short_chunks = [c for c in all_chunks if c["token_count"] < 100]
    
    return {
        "total_input_pages": len(cleaned_pages),
        "total_chunks": len(all_chunks),
        "chunks_by_category": cat_chunk_counts,
        "chunks_by_document": doc_chunk_counts,
        "min_chunk_tokens": min(token_lengths) if token_lengths else 0,
        "max_chunk_tokens": max(token_lengths) if token_lengths else 0,
        "avg_chunk_tokens": round(sum(token_lengths) / max(len(token_lengths), 1), 1),
        "avg_chunk_chars": round(sum(char_lengths) / max(len(char_lengths), 1), 1),
        "extremely_short_chunks": len(short_chunks),
        "output_json": str(output_chunks_file.resolve()),
        "preview_txt": str(preview_file.resolve()),
        "chunks": all_chunks
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    in_json = base_dir / "RAG" / "processed" / "cleaned_pages.json"
    out_json = base_dir / "RAG" / "processed" / "chunks.json"
    out_prev = base_dir / "RAG" / "processed" / "chunks_preview.txt"
    
    res = run_semantic_chunking_pipeline(in_json, out_json, out_prev)
    
    print("\n" + "=" * 80)
    print("STEP 3: SEMANTIC CHUNKING QUALITY REPORT")
    print("=" * 80)
    print(f"Total Input Pages         : {res['total_input_pages']}")
    print(f"Total Chunks Generated    : {res['total_chunks']}")
    print(f"  - Tamil Nadu (Tn)       : {res['chunks_by_category'].get('Tn', 0)}")
    print(f"  - National (India)      : {res['chunks_by_category'].get('India', 0)}")
    print(f"Token Length Range        : {res['min_chunk_tokens']} min | {res['max_chunk_tokens']} max")
    print(f"Average Chunk Size        : {res['avg_chunk_tokens']} tokens ({res['avg_chunk_chars']} characters)")
    print(f"Extremely Short Chunks    : {res['extremely_short_chunks']} (< 100 tokens)")
    print(f"Output Chunks JSON        : {res['output_json']}")
    print(f"Preview Text File         : {res['preview_txt']}")
    print("=" * 80)
