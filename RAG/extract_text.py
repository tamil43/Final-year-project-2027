"""
RAG Knowledge Base - PDF Text Extraction Pipeline (Step 1)
----------------------------------------------------------
This script recursively discovers energy domain policy and planning PDFs from
the knowledge base (categorized into Tamil Nadu 'Tn' and National 'India' documents),
extracts text page-by-page, conducts quality diagnostics, preserves metadata, and
serializes structured page records into JSON for downstream chunking and retrieval.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("PDF_Extractor")


def discover_pdf_documents(base_dir: Path) -> List[Path]:
    """
    Recursively scans base_dir for PDF files.
    
    Args:
        base_dir: Root directory containing knowledge base subfolders.
        
    Returns:
        List of Path objects for discovered PDF files.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {base_dir}")
    
    pdf_paths = sorted(
        list(base_dir.rglob("*.pdf")),
        key=lambda p: (p.parent.name.lower(), p.name.lower())
    )
    return pdf_paths


def extract_pages_from_pdf(
    pdf_path: Path,
    min_char_threshold: int = 50
) -> Dict[str, Any]:
    """
    Extracts text page-by-page from a single PDF document.
    
    Args:
        pdf_path: Path to the PDF file.
        min_char_threshold: Character threshold below which a page is flagged as short.
        
    Returns:
        Dictionary containing extracted page records, document statistics, and anomalies.
    """
    category = pdf_path.parent.name
    filename = pdf_path.name
    
    doc_records: List[Dict[str, Any]] = []
    empty_pages: List[int] = []
    short_pages: List[Dict[str, Any]] = []
    extraction_errors: List[Dict[str, Any]] = []
    
    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
    except Exception as e:
        logger.error(f"Failed to open/parse PDF {filename}: {e}")
        return {
            "source": filename,
            "category": category,
            "total_pages": 0,
            "extracted_count": 0,
            "empty_pages": [],
            "short_pages": [],
            "extraction_errors": [{"page": 0, "error": str(e)}],
            "records": [],
            "total_chars": 0
        }
    
    total_chars = 0
    extracted_count = 0
    
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
            cleaned_text = raw_text.strip()
            char_len = len(cleaned_text)
            
            # Identify problematic or anomalous pages
            if char_len == 0:
                empty_pages.append(page_num)
            elif char_len < min_char_threshold:
                short_pages.append({
                    "page": page_num,
                    "char_count": char_len,
                    "preview": cleaned_text[:40]
                })
                extracted_count += 1
            else:
                extracted_count += 1
                
            total_chars += char_len
            
            # Store structured page record (1-indexed page number)
            # We record all pages, preserving text content (or empty string for scanned/blank pages)
            doc_records.append({
                "source": filename,
                "category": category,
                "page": page_num,
                "text": cleaned_text
            })
            
        except Exception as pe:
            logger.warning(f"Error extracting text from {filename} page {page_num}: {pe}")
            extraction_errors.append({"page": page_num, "error": str(pe)})
            doc_records.append({
                "source": filename,
                "category": category,
                "page": page_num,
                "text": ""
            })
            
    return {
        "source": filename,
        "category": category,
        "total_pages": total_pages,
        "extracted_count": extracted_count,
        "empty_pages": empty_pages,
        "short_pages": short_pages,
        "extraction_errors": extraction_errors,
        "records": doc_records,
        "total_chars": total_chars
    }


def run_extraction_pipeline(
    kb_dir: Path,
    output_file: Path,
    min_char_threshold: int = 50
) -> Dict[str, Any]:
    """
    Executes end-to-end PDF extraction across all documents in knowledge base.
    
    Args:
        kb_dir: Path to knowledge_base folder.
        output_file: Target JSON file path for extracted pages.
        min_char_threshold: Character threshold for flagging short pages.
        
    Returns:
        Pipeline execution summary.
    """
    logger.info(f"Scanning knowledge base at: {kb_dir.resolve()}")
    pdf_files = discover_pdf_documents(kb_dir)
    logger.info(f"Discovered {len(pdf_files)} PDF files.")
    
    all_page_records: List[Dict[str, Any]] = []
    doc_summaries: List[Dict[str, Any]] = []
    
    for pdf_path in pdf_files:
        logger.info(f"Extracting [{pdf_path.parent.name}] {pdf_path.name}...")
        doc_result = extract_pages_from_pdf(pdf_path, min_char_threshold=min_char_threshold)
        all_page_records.extend(doc_result["records"])
        
        doc_summaries.append({
            "source": doc_result["source"],
            "category": doc_result["category"],
            "total_pages": doc_result["total_pages"],
            "extracted_pages": doc_result["extracted_count"],
            "empty_pages_count": len(doc_result["empty_pages"]),
            "empty_pages_list": doc_result["empty_pages"],
            "short_pages_count": len(doc_result["short_pages"]),
            "short_pages_list": doc_result["short_pages"],
            "errors_count": len(doc_result["extraction_errors"]),
            "total_characters": doc_result["total_chars"],
            "avg_chars_per_page": round(
                doc_result["total_chars"] / max(doc_result["total_pages"], 1), 2
            )
        })
        
    # Ensure destination directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save structured records to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_page_records, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Successfully saved {len(all_page_records)} page records to: {output_file.resolve()}")
    
    return {
        "pdf_count": len(pdf_files),
        "total_records": len(all_page_records),
        "doc_summaries": doc_summaries
    }


if __name__ == "__main__":
    # Resolve project root paths
    project_root = Path(__file__).resolve().parent.parent
    kb_directory = project_root / "RAG" / "knowledge_base"
    output_json_path = project_root / "RAG" / "processed" / "extracted_pages.json"
    
    results = run_extraction_pipeline(
        kb_dir=kb_directory,
        output_file=output_json_path
    )
    
    # Print formatted summary table
    print("\n" + "=" * 95)
    print(f"{'DOCUMENT NAME':<45} | {'CATEGORY':<8} | {'PAGES':<6} | {'EXTRACTED':<10} | {'EMPTY':<6} | {'SHORT':<6}")
    print("=" * 95)
    for doc in results["doc_summaries"]:
        print(
            f"{doc['source']:<45} | "
            f"{doc['category']:<8} | "
            f"{doc['total_pages']:<6} | "
            f"{doc['extracted_pages']:<10} | "
            f"{doc['empty_pages_count']:<6} | "
            f"{doc['short_pages_count']:<6}"
        )
    print("=" * 95)
