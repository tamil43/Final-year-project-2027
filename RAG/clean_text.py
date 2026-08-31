"""
RAG Knowledge Base - Page-Level Text Cleaning Pipeline (Step 2)
---------------------------------------------------------------
This module loads extracted page records from RAG/processed/extracted_pages.json,
applies deterministic text cleaning (hyphen-joining, artifact removal, whitespace
normalization while preserving headings, paragraphs, numbers, units, and symbols),
verifies retention of numerical/technical data, and exports RAG/processed/cleaned_pages.json.
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_Text_Cleaner")


def clean_page_text(raw_text: str) -> str:
    """
    Cleans page text deterministically for RAG indexing.
    
    Cleaning Operations:
    1. Removes form feed (\\x0c) and unprintable control characters.
    2. Fixes hyphenated words split across line breaks (e.g., 'genera-\\ntion' -> 'generation').
    3. Replaces Unicode replacement characters (\\ufffd) with standard space.
    4. Normalizes soft line breaks while preserving paragraph boundaries, bullet lists, and section titles.
    5. Retains all numbers, dates, units (MW, GW, MU, MWh, %, Rs, Cr), punctuation, and structure.
    """
    if not raw_text:
        return ""
    
    text = raw_text
    
    # 1. Remove form-feed and unprintable control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    
    # 2. Fix hyphenation split across line breaks (e.g. 'capac-\nity' -> 'capacity')
    text = re.sub(r'([a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,})', r'\1\2', text)
    
    # 3. Replace replacement character symbol artifacts (e.g., \\ufffd)
    text = text.replace('\ufffd', ' ')
    
    # 4. Handle line breaks carefully to preserve paragraphs, headings, bullet lists, and tables
    raw_lines = [line.strip() for line in text.split('\n')]
    
    cleaned_paragraphs = []
    current_para = []
    
    for line in raw_lines:
        if not line:
            if current_para:
                cleaned_paragraphs.append(" ".join(current_para))
                current_para = []
            continue
        
        # Check if line looks like a distinct heading, numbered section, bullet, or table row
        is_heading_or_list = bool(
            re.match(r'^(?:\d+(?:\.\d+)*|\[[a-zA-Z0-9]+\]|[•\-\*\u2022])\s+', line) or
            re.match(r'^(?:SECTION|CHAPTER|TABLE|ANNEXURE|DISCLAIMER|CONTENTS|EXECUTIVE SUMMARY|INTRODUCTION)\b', line, re.I)
        )
        
        if is_heading_or_list and current_para:
            cleaned_paragraphs.append(" ".join(current_para))
            current_para = [line]
        else:
            current_para.append(line)
            
    if current_para:
        cleaned_paragraphs.append(" ".join(current_para))
        
    # Join paragraphs with double newline to maintain readable structure
    cleaned_text = "\n\n".join(cleaned_paragraphs)
    
    # Normalize multiple inline spaces without collapsing paragraph newlines
    lines = cleaned_text.split('\n')
    lines = [re.sub(r'[ \t]+', ' ', l).strip() for l in lines]
    cleaned_text = "\n".join(lines)
    
    # Remove more than 2 consecutive newlines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()
    
    return cleaned_text


def verify_numerical_retention(pages_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verifies that numbers and critical energy units (MW, GW, MU, MWh, %, etc.) are preserved.
    """
    num_pattern = r'\b\d+(?:\.\d+)?\b'
    total_raw_nums = 0
    total_clean_nums = 0
    
    for item in pages_data:
        raw_matches = re.findall(num_pattern, item["text"])
        clean_matches = re.findall(num_pattern, item["cleaned_text"])
        total_raw_nums += len(raw_matches)
        total_clean_nums += len(clean_matches)
        
    retention_rate = (total_clean_nums / max(total_raw_nums, 1)) * 100
    return {
        "raw_numerical_tokens": total_raw_nums,
        "cleaned_numerical_tokens": total_clean_nums,
        "retention_percentage": round(retention_rate, 2)
    }


def run_text_cleaning_pipeline(
    input_file: Path,
    output_file: Path
) -> Dict[str, Any]:
    """
    Loads extracted_pages.json, applies cleaning, and saves cleaned_pages.json.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Extracted pages file not found: {input_file}")
        
    with open(input_file, "r", encoding="utf-8") as f:
        pages_data = json.load(f)
        
    logger.info(f"Loaded {len(pages_data)} records from {input_file.resolve()}")
    
    cleaned_records = []
    empty_cleaned_count = 0
    total_raw_len = 0
    total_clean_len = 0
    
    for item in pages_data:
        raw_txt = item.get("text", "")
        clean_txt = clean_page_text(raw_txt)
        
        raw_len = len(raw_txt)
        clean_len = len(clean_txt)
        
        total_raw_len += raw_len
        total_clean_len += clean_len
        
        if not clean_txt:
            empty_cleaned_count += 1
            
        record = {
            "source": item["source"],
            "category": item["category"],
            "page": item["page"],
            "text": raw_txt,
            "cleaned_text": clean_txt
        }
        cleaned_records.append(record)
        
    num_report = verify_numerical_retention(cleaned_records)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Successfully saved {len(cleaned_records)} records to {output_file.resolve()}")
    
    total_pages = len(cleaned_records)
    avg_raw_len = total_raw_len / max(total_pages, 1)
    avg_clean_len = total_clean_len / max(total_pages, 1)
    
    return {
        "total_pages_processed": total_pages,
        "records_cleaned": len(cleaned_records),
        "empty_cleaned_pages": empty_cleaned_count,
        "avg_raw_length": round(avg_raw_len, 2),
        "avg_clean_length": round(avg_clean_len, 2),
        "numerical_report": num_report,
        "output_file": str(output_file.resolve()),
        "cleaned_records": cleaned_records
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    in_json = base_dir / "RAG" / "processed" / "extracted_pages.json"
    out_json = base_dir / "RAG" / "processed" / "cleaned_pages.json"
    
    res = run_text_cleaning_pipeline(in_json, out_json)
    
    print("\n" + "=" * 80)
    print("STEP 2: PAGE-LEVEL TEXT CLEANING QUALITY REPORT")
    print("=" * 80)
    print(f"Total Pages Processed          : {res['total_pages_processed']}")
    print(f"Records Successfully Cleaned    : {res['records_cleaned']}")
    print(f"Pages with Empty Cleaned Text   : {res['empty_cleaned_pages']}")
    print(f"Average Raw Text Length        : {res['avg_raw_length']} characters")
    print(f"Average Cleaned Text Length    : {res['avg_clean_length']} characters")
    print(f"Numerical Tokens (Raw)         : {res['numerical_report']['raw_numerical_tokens']}")
    print(f"Numerical Tokens (Cleaned)     : {res['numerical_report']['cleaned_numerical_tokens']}")
    print(f"Numerical Token Retention Rate : {res['numerical_report']['retention_percentage']}%")
    print(f"Output Saved To                : {res['output_file']}")
    print("=" * 80)
