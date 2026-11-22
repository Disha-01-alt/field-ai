"""
Real-data evaluation of the extraction pipeline.

    real receipt photo -> preprocessing -> PaddleOCR -> text + boxes -> advanced regex
                       -> {company, date, total}  -> compared with the dataset's answers

Nothing here uses the pre-made words/boxes in data/processed/*.json. The receipt
IMAGES of the SROIE dataset are downloaded, read by PaddleOCR, and the extracted
fields are scored against the dataset's ground-truth labels with src/metrics.py.

OCR is the slow part, so the OCR result of every image is cached on disk and the run
can be stopped and resumed. Changing only the regex rules re-scores from the cache
in seconds.

Usage:
    python -m src.real_eval --limit 50          # 50 test receipts
    python -m src.real_eval --limit 0           # all test receipts
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from src.advanced_regex import AdvancedRegexExtractor, OCRToken
from src.metrics import compute_field_metrics

SROIE_REPO = "podbilabs/sroie-donut"   # public copy of the SROIE receipts (images + labels)
FIELDS = ("company", "date", "total")


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

def load_ground_truth(metadata_path: str) -> Dict[str, Dict[str, str]]:
    """Read metadata.jsonl -> {image_file_name: {company, date, total}}."""
    truth: Dict[str, Dict[str, str]] = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gt = row["ground_truth"]
            if isinstance(gt, str):
                gt = json.loads(gt)
            parsed = gt.get("gt_parse", gt)
            truth[row["file_name"]] = {k: str(parsed.get(k, "") or "") for k in FIELDS}
    return truth


def download_sroie(
    split: str = "test",
    data_dir: str = "data/sroie",
    limit: Optional[int] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Download receipt images + labels of one SROIE split from Hugging Face.
    Files that already exist are not downloaded again.
    Returns the ground truth for the (first `limit`) images, sorted by file name.
    """
    from huggingface_hub import hf_hub_download

    split_dir = os.path.join(data_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    meta_path = hf_hub_download(SROIE_REPO, f"{split}/metadata.jsonl", repo_type="dataset", local_dir=data_dir)
    truth = load_ground_truth(meta_path)

    names = sorted(truth)
    if limit:
        names = names[:limit]

    for name in names:
        if not os.path.exists(os.path.join(split_dir, name)):
            hf_hub_download(SROIE_REPO, f"{split}/{name}", repo_type="dataset", local_dir=data_dir)

    return {name: truth[name] for name in names}


# ---------------------------------------------------------------------------
# 2. OCR (slow, cached)
# ---------------------------------------------------------------------------

def run_ocr(
    names: List[str],
    image_dir: str,
    cache_dir: str,
    pipeline: Any,
    show_progress: bool = True,
) -> Dict[str, int]:
    """
    Run the pipeline (preprocessing + PaddleOCR) on every image that is not cached yet and
    save its tokens to `cache_dir/<image>.json`. Safe to stop and run again.
    Returns {"done": n_new, "cached": n_already, "errors": n_failed}.
    """
    os.makedirs(cache_dir, exist_ok=True)
    todo = [n for n in names if not os.path.exists(os.path.join(cache_dir, n + ".json"))]
    stats = {"done": 0, "cached": len(names) - len(todo), "errors": 0}

    iterator = todo
    if show_progress and todo:
        from tqdm import tqdm
        iterator = tqdm(todo, desc="OCR", unit="img")

    for name in iterator:
        started = time.time()
        try:
            result = pipeline.process(os.path.join(image_dir, name))
            record = {
                "tokens": result["tokens"],
                "preprocessing": {k: v for k, v in result["preprocessing"].items()},
                "seconds": round(time.time() - started, 2),
            }
        except Exception as exc:  # one bad image must not stop a long run
            record = {"error": f"{type(exc).__name__}: {exc}"}
            stats["errors"] += 1
        with open(os.path.join(cache_dir, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(record, f, default=str)
        stats["done"] += 1

    return stats


# ---------------------------------------------------------------------------
# 3. Extraction + scoring (fast, uses the cache)
# ---------------------------------------------------------------------------

def extract_from_cache(names: List[str], cache_dir: str) -> Dict[str, Dict[str, str]]:
    """Run the advanced regex extractor on the cached OCR tokens. Failed/missing OCR -> empty fields."""
    predictions: Dict[str, Dict[str, str]] = {}
    for name in names:
        path = os.path.join(cache_dir, name + ".json")
        record: Dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        if "tokens" not in record:
            predictions[name] = {k: "" for k in FIELDS}
            continue
        tokens = [OCRToken(t["text"], t["box"], t.get("confidence", 1.0)) for t in record["tokens"]]
        predictions[name] = AdvancedRegexExtractor.extract_fields(tokens)
    return predictions


def evaluate_real_data(
    split: str = "test",
    limit: Optional[int] = 50,
    fast: bool = True,
    data_dir: str = "data/sroie",
    cache_root: str = "outputs/ocr_cache",
    output_path: str = "results/regex_baseline.json",
    pipeline: Any = None,
) -> Dict[str, Any]:
    """
    Full real-data benchmark of "PaddleOCR + Advanced Regex":
      download images -> OCR (cached) -> extract fields -> score -> write `output_path`.

    limit: number of receipts to use (None or 0 = the whole split).
    fast:  use PaddleOCR's small "mobile" models (about 3-4x faster, similar quality on receipts).
    """
    truth = download_sroie(split=split, data_dir=data_dir, limit=limit)
    names = list(truth)
    image_dir = os.path.join(data_dir, split)
    cache_dir = os.path.join(cache_root, "mobile" if fast else "server")

    if pipeline is None:
        from src.pipeline import ReceiptExtractionPipeline
        pipeline = ReceiptExtractionPipeline(fast=fast)

    print(f"Running the real pipeline on {len(names)} SROIE '{split}' receipt images "
          f"({'mobile' if fast else 'server'} OCR models)...")
    stats = run_ocr(names, image_dir, cache_dir, pipeline)

    predictions = extract_from_cache(names, cache_dir)
    metrics = compute_field_metrics(predictions, truth)

    ocr_errors = sum(1 for n in names if not _has_tokens(cache_dir, n))
    results = {
        "model": "PaddleOCR + Advanced Regex",
        "modality": "Real receipt images -> PaddleOCR -> text + boxes (rules)",
        "test_size": len(names),
        "ocr_errors": ocr_errors,
        "metrics": metrics,
        "sample_predictions": {k: predictions[k] for k in names[:5]},
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_report(results, stats)
    return results


def _has_tokens(cache_dir: str, name: str) -> bool:
    path = os.path.join(cache_dir, name + ".json")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        return "tokens" in json.load(f)


def print_report(results: Dict[str, Any], stats: Optional[Dict[str, int]] = None) -> None:
    metrics = results["metrics"]
    print("\n" + "=" * 60)
    print(f"Real-data results: PaddleOCR + Advanced Regex on {results['test_size']} receipt images")
    print("=" * 60)
    for field in (*FIELDS, "macro_avg"):
        r = metrics.get(field, {})
        print(f"{field.upper():<10} | Precision {r.get('precision', 0):.4f} | "
              f"Recall {r.get('recall', 0):.4f} | F1 {r.get('f1', 0):.4f}")
    print("=" * 60)
    if results.get("ocr_errors"):
        print(f"WARNING: OCR failed on {results['ocr_errors']} image(s); they count as empty predictions.")
    if stats:
        print(f"OCR: {stats['done']} new, {stats['cached']} from cache.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the receipt pipeline on real SROIE images")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=50, help="number of receipts (0 = all)")
    parser.add_argument("--server-models", action="store_true", help="use the larger, slower OCR models")
    parser.add_argument("--output", default="results/regex_baseline.json")
    args = parser.parse_args()
    evaluate_real_data(split=args.split, limit=args.limit or None, fast=not args.server_models,
                       output_path=args.output)


if __name__ == "__main__":
    main()
