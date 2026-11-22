"""
Unified Evaluation Metrics for Receipt Information Extraction.

This module provides the single source of truth for evaluating:
1. Exact Match Field-Level Extraction (Precision, Recall, F1 per field and Macro F1)
2. Normalized String Matching (field-aware: date/total/company handled separately)
3. Empirical Delta Analysis: ΔF1 = F1(LayoutLM) - F1(BERT)
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple


def normalize_text(text: Optional[str]) -> str:
    """Normalize text for fair exact-match comparison."""
    if text is None:
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Normalize unicode whitespace and collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing punctuation and whitespace
    text = text.strip(" \t\n\r.,:;'\"`")
    return text


def normalize_total(text: Optional[str]) -> str:
    """Normalize currency/total amounts (strip currency symbols and trailing spaces)."""
    if text is None:
        return ""
    text = str(text).lower()
    # Remove currency prefixes: rm, $, usd, myr
    text = re.sub(r"\b(rm|usd|myr|sgd|\$)\b", "", text)
    text = text.replace("$", "").replace(",", "")
    # Find decimal number pattern
    match = re.search(r"(\d+\.\d{2})", text)
    if match:
        return match.group(1)
    match_int = re.search(r"(\d+)", text)
    if match_int:
        return match_int.group(1) + ".00"
    return text.strip()


def normalize_date(text: Optional[str]) -> str:
    """Normalize date strings for robust matching."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Remove separators to unify e.g. 15/01/2019 vs 15-01-2019 vs 15.01.2019
    text = re.sub(r"[.\-/\s]", "/", text)
    return text


def field_match(pred: str, gt: str, field_name: str) -> bool:
    """Check if prediction matches ground truth based on field type."""
    norm_pred = normalize_text(pred)
    norm_gt = normalize_text(gt)

    if not norm_gt:
        return not norm_pred

    if norm_pred == norm_gt:
        return True

    # Field-specific soft-matching
    if field_name.lower() == "total":
        return normalize_total(pred) == normalize_total(gt)

    if field_name.lower() == "date":
        return normalize_date(pred) == normalize_date(gt)

    # For company and address, check bidirectional substring if length >= 4
    if field_name.lower() in ("company", "address"):
        if norm_pred and norm_gt:
            if norm_pred in norm_gt or norm_gt in norm_pred:
                # Require significant length overlap (at least 75%)
                shorter = min(len(norm_pred), len(norm_gt))
                longer = max(len(norm_pred), len(norm_gt))
                if shorter / longer >= 0.75:
                    return True

    return False


def compute_field_metrics(
    predictions: Dict[str, Dict[str, str]],
    ground_truths: Dict[str, Dict[str, str]],
    target_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute strict Precision, Recall, and F1 for each field across all documents.
    
    Args:
        predictions: {doc_id: {"company": ..., "date": ..., "total": ...}}
        ground_truths: {doc_id: {"company": ..., "date": ..., "total": ...}}
        target_fields: list of field keys to evaluate (default: ['company', 'date', 'total'])
        
    Returns:
        Dictionary with per-field and overall macro/micro P, R, F1.
    """
    if target_fields is None:
        target_fields = ["company", "date", "total"]

    results = {}
    macro_f1_sum = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for field in target_fields:
        tp = 0
        fp = 0
        fn = 0

        for doc_id, gt_dict in ground_truths.items():
            gt_val = gt_dict.get(field, "")
            pred_dict = predictions.get(doc_id, {})
            pred_val = pred_dict.get(field, "")

            has_gt = bool(normalize_text(gt_val))
            has_pred = bool(normalize_text(pred_val))

            if has_gt and has_pred:
                if field_match(pred_val, gt_val, field):
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif has_pred and not has_gt:
                fp += 1
            elif has_gt and not has_pred:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results[field] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,
        }
        macro_f1_sum += f1
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_macro_f1 = macro_f1_sum / len(target_fields) if target_fields else 0.0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    results["macro_avg"] = {
        "precision": round(sum(results[f]["precision"] for f in target_fields) / len(target_fields), 4),
        "recall": round(sum(results[f]["recall"] for f in target_fields) / len(target_fields), 4),
        "f1": round(overall_macro_f1, 4),
    }
    results["micro_avg"] = {
        "precision": round(micro_precision, 4),
        "recall": round(micro_recall, 4),
        "f1": round(micro_f1, 4),
    }

    return results


def compute_delta_f1(
    bert_metrics: Dict[str, Any],
    layoutlm_metrics: Dict[str, Any],
    target_fields: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute ΔF1 = F1(LayoutLM) - F1(BERT) per field and overall."""
    if target_fields is None:
        target_fields = ["company", "date", "total"]

    deltas = {}
    for field in target_fields:
        bert_f1 = bert_metrics.get(field, {}).get("f1", 0.0)
        layoutlm_f1 = layoutlm_metrics.get(field, {}).get("f1", 0.0)
        deltas[field] = round(layoutlm_f1 - bert_f1, 4)

    bert_macro = bert_metrics.get("macro_avg", {}).get("f1", 0.0)
    layoutlm_macro = layoutlm_metrics.get("macro_avg", {}).get("f1", 0.0)
    deltas["macro_avg"] = round(layoutlm_macro - bert_macro, 4)

    return deltas


def format_results_markdown_table(
    all_metrics: Dict[str, Dict[str, Any]],
    target_fields: Optional[List[str]] = None,
) -> str:
    """Format multiple model results into a GitHub Flavored Markdown table."""
    if target_fields is None:
        target_fields = ["company", "date", "total"]

    header = "| Model Paradigm | Input Modality | " + " | ".join(f.capitalize() + " F1" for f in target_fields) + " | Overall Macro F1 |\n"
    separator = "|:---|:---|" + "|".join(":---:" for _ in target_fields) + "|:---:|\n"
    
    rows = []
    for model_name, info in all_metrics.items():
        modality = info.get("modality", "N/A")
        metrics = info.get("metrics", {})
        f1_vals = [f"{metrics.get(f, {}).get('f1', 0.0) * 100:.2f}%" for f in target_fields]
        macro_f1 = f"{metrics.get('macro_avg', {}).get('f1', 0.0) * 100:.2f}%"
        rows.append(f"| **{model_name}** | {modality} | " + " | ".join(f1_vals) + f" | **{macro_f1}** |")

    return header + separator + "\n".join(rows)


if __name__ == "__main__":
    # Self-test with sample synthetic ground truths and predictions
    sample_gt = {
        "doc1": {"company": "Starbucks Coffee", "date": "15/01/2019", "total": "24.50"},
        "doc2": {"company": "Target Store 123", "date": "2019-02-20", "total": "110.00"},
    }
    sample_pred = {
        "doc1": {"company": "Starbucks Coffee", "date": "15/01/2019", "total": "24.50"},
        "doc2": {"company": "Target Store", "date": "2019/02/20", "total": "110.00"},
    }
    res = compute_field_metrics(sample_pred, sample_gt)
    print("Self-test metrics:\n", json.dumps(res, indent=2))
