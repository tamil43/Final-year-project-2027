"""
RAG Knowledge Base - Text Cleaning & Chunking Pipeline (Step 2)
----------------------------------------------------------------
This script loads page-level JSON records from RAG/processed/extracted_pages.json,
applies robust text cleaning rules (fixing hyphenation, normalizing whitespace,
stripping unprintable control artifacts), and partitions text into overlapping,
sentence-aware chunks while preserving full metadata for downstream vector search.
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_Chunker")


def clean_text(text: str) -> str:
    """
    Cleans raw PDF extracted text.
    
    1. Removes form feeds and control characters.
    2. Replaces replacement symbol artifacts (e.g. unicode replacement chars).
    3. Fixes line-break hyphenation (e.g., 'genera-\ntion' -> 'generation').
    4. Normalizes internal line breaks and consecutive spaces.
    """
    if not text:
        return ""
    
    # 1. Remove form-feed and unprintable control chars (preserve standard whitespace)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    
    # 2. Fix hyphenation across line breaks (e.g., "power-\nsupply" -> "powersupply", "techno-\nlogy" -> "technology")
    text = re.sub(r'([a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,})', r'\1\2', text)
    
    # 3. Replace Unicode replacement character  with space or clean mark
    text = text.replace('\ufffd', ' ')
    
    # 4. Standardize line breaks: convert single newlines within sentences to spaces, keep paragraph breaks
    # Replace multiple spaces / tabs with single space
    lines = text.split('\n')
    cleaned_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    cleaned_lines = [line for line in cleaned_lines if line]  # remove empty lines
    
    cleaned_text = " ".join(cleaned_lines)
    
    # Final space normalization
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def chunk_text_by_sentence(
    text: str,
    source: str,
    category: str,
    page: int,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Splits text into sentence-aware overlapping chunks.
    
    Args:
        text: Cleaned text string for a page.
        source: Source document filename.
        category: 'Tn' or 'India'.
        page: Page number.
        chunk_size: Target maximum character length per chunk.
        chunk_overlap: Target character overlap between consecutive chunks.
        
    Returns:
        List of structured chunk records.
    """
    if not text or len(text) < 20:
        return []
    
    # Split text into sentences using regex boundary lookbehind
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_sentences: List[str] = []
    current_length = 0
    chunk_idx = 1
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # If adding this sentence exceeds chunk_size and we already have content
        if current_length + sentence_len > chunk_size and current_sentences:
            chunk_str = " ".join(current_sentences)
            clean_filename = re.sub(r'[^a-zA-Z0-9]', '_', Path(source).stem)
            chunk_id = f"{clean_filename}_p{page}_c{chunk_idx}"
            
            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "category": category,
                "page": page,
                "text": chunk_str,
                "char_count": len(chunk_str),
                "word_count": len(chunk_str.split())
            })
            chunk_idx += 1
            
            # Form overlap window by keeping sentences from the end of current_sentences
            overlap_sentences: List[str] = []
            overlap_len = 0
            for prev_s in reversed(current_sentences):
                if overlap_len + len(prev_s) <= chunk_overlap:
                    overlap_sentences.insert(0, prev_s)
                    overlap_len += len(prev_s)
                else:
                    break
                    
            current_sentences = overlap_sentences
            current_length = sum(len(s) for s in current_sentences)
            
        current_sentences.append(sentence)
        current_length += sentence_len
        
    # Flush remaining sentences
    if current_sentences:
        chunk_str = " ".join(current_sentences)
        clean_filename = re.sub(r'[^a-zA-Z0-9]', '_', Path(source).stem)
        chunk_id = f"{clean_filename}_p{page}_c{chunk_idx}"
        
        chunks.append({
            "chunk_id": chunk_id,
            "source": source,
            "category": category,
            "page": page,
            "text": chunk_str,
            "char_count": len(chunk_str),
            "word_count": len(chunk_str.split())
        })
        
    return chunks


def process_pages_to_chunks(
    pages_json_path: Path,
    output_chunks_path: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> Dict[str, Any]:
    """
    Loads extracted pages, cleans text, produces chunks, and saves chunks.json.
    """
    if not pages_json_path.exists():
        raise FileNotFoundError(f"Extracted pages file not found: {pages_json_path}")
        
    with open(pages_json_path, "r", encoding="utf-8") as f:
        extracted_pages = json.load(f)
        
    logger.info(f"Loaded {len(extracted_pages)} page records from {pages_json_path.name}")
    
    all_chunks: List[Dict[str, Any]] = []
    tn_chunk_count = 0
    india_chunk_count = 0
    
    for record in extracted_pages:
        raw_text = record.get("text", "")
        cleaned = clean_text(raw_text)
        
        page_chunks = chunk_text_by_sentence(
            text=cleaned,
            source=record["source"],
            category=record["category"],
            page=record["page"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        for c in page_chunks:
            if c["category"] == "Tn":
                tn_chunk_count += 1
            else:
                india_chunk_count += 1
            all_chunks.append(c)
            
    # Create destination dir if needed
    output_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Generated {len(all_chunks)} chunks (Tn: {tn_chunk_count}, India: {india_chunk_count})")
    logger.info(f"Saved chunks to {output_chunks_path.resolve()}")
    
    avg_char = sum(c["char_count"] for c in all_chunks) / max(len(all_chunks), 1)
    avg_word = sum(c["word_count"] for c in all_chunks) / max(len(all_chunks), 1)
    
    return {
        "total_chunks": len(all_chunks),
        "tn_chunks": tn_chunk_count,
        "india_chunks": india_chunk_count,
        "avg_char_count": round(avg_char, 1),
        "avg_word_count": round(avg_word, 1),
        "output_path": str(output_chunks_path.resolve())
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    input_file = base_dir / "RAG" / "processed" / "extracted_pages.json"
    output_file = base_dir / "RAG" / "processed" / "chunks.json"
    
    res = process_pages_to_chunks(input_file, output_file)
    print("\n" + "=" * 80)
    print("STAGE 2: TEXT CLEANING & CHUNKING SUMMARY")
    print("=" * 80)
    print(f"Total Chunks Generated : {res['total_chunks']}")
    print(f"  - Tamil Nadu (Tn)    : {res['tn_chunks']}")
    print(f"  - National (India)   : {res['india_chunks']}")
    print(f"Average Chunk Size     : {res['avg_char_count']} characters ({res['avg_word_count']} words)")
    print(f"Output File Saved      : {res['output_path']}")
    print("=" * 80)
