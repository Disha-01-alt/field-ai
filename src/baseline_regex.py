"""
Baseline 1: OCR + Regular Expression Field Extractor for SROIE Receipts.

This module provides a deterministic, zero-learning heuristic baseline:
- Date: Regular expressions covering Malaysian & international receipt formats
- Total: Currency regex with keyword proximity (avoiding subtotal / cash / change)
- Company: Header line positional heuristics
"""

import os
import re
import json
import glob
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.metrics import compute_field_metrics


class RegexReceiptExtractor:
    """Deterministic rule-based information extractor for scanned receipts."""

    # Date regex patterns ordered by specificity
    DATE_PATTERNS = [
        # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
        re.compile(r"\b([0-3]?[0-9][/\-\.][0-1]?[0-9][/\-\.](?:20|19)?\d{2})\b"),
        # YYYY/MM/DD or YYYY-MM-DD
        re.compile(r"\b((?:20|19)\d{2}[/\-\.][0-1]?[0-9][/\-\.][0-3]?[0-9])\b"),
        # 15 JAN 2019 or 15-JAN-19
        re.compile(
            r"\b([0-3]?[0-9][\s\-\.](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s\-\.](?:20|19)?\d{2})\b",
            re.IGNORECASE,
        ),
        # JAN 15, 2019
        re.compile(
            r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s\-\.][0-3]?[0-9],?[\s\-\.](?:20|19)?\d{2})\b",
            re.IGNORECASE,
        ),
    ]

    # Keyword indicators for grand total
    TOTAL_KEYWORDS = [
        "grand total",
        "total amount",
        "amount due",
        "total rm",
        "net total",
        "total:",
        "total",
        "balance due",
        "jumlah",
    ]

    # Negative total keywords that frequently cause false positives
    EXCLUDE_TOTAL_KEYWORDS = [
        "subtotal",
        "sub-total",
        "sub total",
        "tax total",
        "cash",
        "change",
        "rounding",
        "discount",
        "qty",
        "item",
    ]

    # Currency value pattern: e.g. 12.50 or 1,234.50
    PRICE_PATTERN = re.compile(r"(?:rm|myr|\$)?\s*([0-9]{1,4}(?:,[0-9]{3})*\.[0-9]{2})\b", re.IGNORECASE)

    # Header noise lines to skip when looking for company name
    COMPANY_NOISE = [
        "tax invoice",
        "simplified tax invoice",
        "invoice",
        "receipt",
        "official receipt",
        "welcome",
        "thank you",
        "gst reg",
        "reg no",
        "cash sale",
        "sales slip",
    ]

    @classmethod
    def extract_date(cls, lines: List[str]) -> str:
        """Scan receipt lines for the transaction date."""
        for line in lines:
            # Skip lines that look like serial numbers or tax registration
            if any(k in line.lower() for k in ["gst", "reg", "tel", "fax", "roc"]):
                continue
            for pattern in cls.DATE_PATTERNS:
                match = pattern.search(line)
                if match:
                    date_str = match.group(1).strip()
                    # Sanity check: must contain at least 4 digits
                    if sum(c.isdigit() for c in date_str) >= 4:
                        return date_str
        return ""

    @classmethod
    def extract_total(cls, lines_with_boxes: List[Tuple[str, List[int]]]) -> str:
        """
        Extract the grand total by identifying lines with total keywords,
        avoiding subtotals and picking the decimal price nearby.
        """
        candidate_totals = []

        for idx, (line, box) in enumerate(lines_with_boxes):
            lower_line = line.lower()

            # Skip lines containing negative indicators
            if any(neg in lower_line for neg in cls.EXCLUDE_TOTAL_KEYWORDS):
                continue

            # Check if this line contains any total keyword
            has_total_keyword = any(kw in lower_line for kw in cls.TOTAL_KEYWORDS)

            if has_total_keyword:
                # Look for price on the same line first
                matches = cls.PRICE_PATTERN.findall(line)
                if matches:
                    val = matches[-1].replace(",", "")
                    candidate_totals.append((val, idx, 10))  # High priority
                else:
                    # Look at adjacent line (e.g. Total on line i, Amount on line i+1)
                    if idx + 1 < len(lines_with_boxes):
                        next_line, _ = lines_with_boxes[idx + 1]
                        next_matches = cls.PRICE_PATTERN.findall(next_line)
                        if next_matches:
                            candidate_totals.append((next_matches[-1].replace(",", ""), idx, 8))

        if candidate_totals:
            # Pick highest priority, or if tie, the one appearing lower down on the receipt
            candidate_totals.sort(key=lambda x: (x[2], x[1]), reverse=True)
            return candidate_totals[0][0]

        # Fallback: scan from the bottom upwards for the last valid currency value
        for line, _ in reversed(lines_with_boxes):
            if any(neg in line.lower() for neg in cls.EXCLUDE_TOTAL_KEYWORDS):
                continue
            matches = cls.PRICE_PATTERN.findall(line)
            if matches:
                return matches[-1].replace(",", "")

        return ""

    @classmethod
    def extract_company(cls, lines_with_boxes: List[Tuple[str, List[int]]]) -> str:
        """
        Extract merchant name by inspecting prominent top-most text lines
        before address or transaction details.
        """
        candidate_lines = []
        for line, box in lines_with_boxes[:8]:  # inspect top 8 lines
            cleaned = line.strip()
            lower = cleaned.lower()

            # Skip empty lines, pure numbers, or short junk
            if len(cleaned) < 3 or cleaned.isdigit():
                continue

            # Skip noise titles
            if any(noise in lower for noise in cls.COMPANY_NOISE):
                continue

            # Skip lines that look like addresses, phone numbers, or dates
            if any(k in lower for k in ["jalan", "street", "no.", "tel:", "phone", "fax", "highway", "avenue", "sdn", "bhd"]):
                # Allow SDN BHD for company name
                if not any(co in lower for co in ["sdn", "bhd", "enterprise", "store", "restaurant", "cafe", "coffee", "mart"]):
                    continue
            if cls.extract_date([cleaned]):
                continue

            # Prefer lines with alphabetical content
            alpha_count = sum(c.isalpha() for c in cleaned)
            if alpha_count >= 3:
                candidate_lines.append(cleaned)

        if candidate_lines:
            # Take the topmost valid line
            return candidate_lines[0]

        return ""

    @classmethod
    def extract_fields(cls, lines_with_boxes: List[Tuple[str, List[int]]]) -> Dict[str, str]:
        """Extract all target fields from receipt lines."""
        lines = [line for line, _ in lines_with_boxes]
        return {
            "company": cls.extract_company(lines_with_boxes),
            "date": cls.extract_date(lines),
            "total": cls.extract_total(lines_with_boxes),
        }


def load_lines_from_box_file(file_path: str) -> List[Tuple[str, List[int]]]:
    """Load text lines and bounding boxes from a CSV or TXT box file."""
    lines_with_boxes = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 9:
                try:
                    coords = [int(p.strip()) for p in parts[:8]]
                    text = ",".join(parts[8:]).strip()
                    if text:
                        lines_with_boxes.append((text, coords))
                except ValueError:
                    continue
    return lines_with_boxes


def run_baseline_evaluation(
    data_dir: str = "data/sroie",
    processed_test_file: str = "data/processed/test.json",
    output_path: str = "results/regex_baseline.json",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate the regex baseline on the SROIE test split.
    Automatically uses data/processed/test.json if available,
    or falls back to raw data/sroie directory.
    """
    predictions = {}
    ground_truths = {}

    # Prefer preprocessed test.json if present
    if os.path.exists(processed_test_file):
        print(f"Loading test split from {processed_test_file}...")
        with open(processed_test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        print(f"Evaluating Baseline 1 on {len(test_data)} test receipts...")
        for item in test_data:
            doc_id = item["id"]
            # Group words sharing identical or near bounding boxes into coherent lines
            lines = []
            curr_box = None
            curr_words = []
            for w, b in zip(item["words"], item["bboxes"]):
                if curr_box is None:
                    curr_box = b
                    curr_words = [w]
                elif b == curr_box:
                    curr_words.append(w)
                else:
                    lines.append((" ".join(curr_words), curr_box))
                    curr_box = b
                    curr_words = [w]
            if curr_words:
                lines.append((" ".join(curr_words), curr_box))

            pred = RegexReceiptExtractor.extract_fields(lines)
            predictions[doc_id] = pred
            ground_truths[doc_id] = item["entities"]

    else:
        import random
        random.seed(seed)

        data_path = Path(data_dir)
        box_files = sorted(data_path.glob("**/box/*.csv")) or sorted(data_path.glob("**/box/*.txt"))
        
        # Pair with ground truth keys
        paired_docs = []
        for bf in box_files:
            stem = bf.stem
            key_candidates = [
                bf.parent.parent / "key" / f"{stem}.json",
                bf.parent / f"{stem}.json",
            ]
            key_file = next((k for k in key_candidates if k.exists()), None)
            if key_file:
                with open(key_file, "r", encoding="utf-8", errors="ignore") as f:
                    try:
                        gt = json.load(f)
                        paired_docs.append((bf, gt))
                    except Exception:
                        continue

        print(f"Total valid paired documents: {len(paired_docs)}")
        if not paired_docs:
            print("Error: No paired documents found.")
            return {}

        random.shuffle(paired_docs)
        split_idx = int(len(paired_docs) * (1 - test_ratio))
        test_docs = paired_docs[split_idx:]
        print(f"Evaluating Baseline 1 on test set of {len(test_docs)} documents...")

        for bf, gt in test_docs:
            doc_id = bf.stem
            lines_with_boxes = load_lines_from_box_file(str(bf))
            pred = RegexReceiptExtractor.extract_fields(lines_with_boxes)
            predictions[doc_id] = pred
            ground_truths[doc_id] = gt

    metrics = compute_field_metrics(predictions, ground_truths)
    
    output_data = {
        "model": "OCR + Regex Baseline",
        "modality": "Plain Text (Rules)",
        "test_size": len(predictions),
        "metrics": metrics,
        "sample_predictions": {k: predictions[k] for k in list(predictions.keys())[:5]},
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 50)
    print("Baseline 1 (OCR + Regex) Evaluation Results:")
    print("=" * 50)
    for field in ["company", "date", "total", "macro_avg"]:
        m = metrics.get(field, {})
        print(f"  {field.upper():<12} | Precision: {m.get('precision', 0):.4f} | Recall: {m.get('recall', 0):.4f} | F1: {m.get('f1', 0):.4f}")
    print("=" * 50)
    print(f"Results saved to {output_path}")

    return output_data


if __name__ == "__main__":
    run_baseline_evaluation()
