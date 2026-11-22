"""
Build the training / test data from REAL receipt photos read by PaddleOCR.

    photo -> preprocessing -> PaddleOCR -> words + boxes     (cached, see src/real_eval.py)
                                              |
    dataset answers (company, date, total, address) -----> word labels (BIO)  -> data/ocr/{train,test}.json

The output files have exactly the same format as data/processed/*.json, so the existing
train_bert() and train() (LayoutLM) can train on them unchanged. The difference is that the
words and boxes now come from OCR of the photos, with all of OCR's mistakes, which is what
the models will see on a new receipt.

Splits (all photos are used):
    train.json = SROIE train + val photos   (the trainer holds out part of it for early stopping)
    test.json  = SROIE test photos          (never used for training)
"""

import json
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from src.dataset import LABEL2ID, align_tokens_with_entities
from src.real_eval import (FIELDS, LazyPipeline, _load_tokens, download_sroie, run_ocr,
                           tokens_to_words)

ALL_FIELDS = (*FIELDS, "address")
# Tolerance for OCR misreads when matching an answer to OCR words (1.0 = identical).
MIN_RATIO = {"company": 0.85, "date": 0.85, "address": 0.80}


def _squash(text: str) -> str:
    """Lowercase letters and digits only, so '15/01/2019' == '15-01-2019' and 'S/B.' == 'sb'."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def best_fuzzy_span(words: List[str], answer: str, min_ratio: float, taken: List[bool]) -> Optional[Tuple[int, int]]:
    """
    Find the run of OCR words that looks most like `answer` (tolerating misread characters and
    words that OCR split or merged). Returns (start, end_exclusive), or None if nothing is close
    enough or the only candidates overlap words that already carry a label.
    """
    target = _squash(answer)
    n_answer = len(str(answer).split())
    if len(target) < 4 or not words:
        return None

    squashed = [_squash(w) for w in words]
    best, best_ratio = None, min_ratio
    for start in range(len(words)):
        if taken[start]:
            continue
        joined = ""
        for end in range(start + 1, min(len(words), start + n_answer + 3) + 1):
            if taken[end - 1]:
                break
            joined += squashed[end - 1]
            if end - start < max(1, n_answer - 2) or not joined:
                continue
            matcher = SequenceMatcher(None, target, joined)
            if matcher.real_quick_ratio() < best_ratio or matcher.quick_ratio() < best_ratio:
                continue
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best, best_ratio = (start, end), ratio
    return best


def align_ocr_words(words: List[str], entities: Dict[str, str]) -> List[str]:
    """
    BIO tags for OCR words. Exact matching first (src/dataset.py); for a field that was not found
    exactly (OCR misread it), fall back to the closest-looking run of words.
    Totals are only matched exactly: a "close" amount would label the wrong number.
    """
    tags = align_tokens_with_entities(words, entities)
    taken = [t != "O" for t in tags]
    for field in ("date", "company", "address"):
        answer = entities.get(field, "")
        if not answer or any(t.endswith(field.upper()) for t in tags):
            continue
        span = best_fuzzy_span(words, answer, MIN_RATIO[field], taken)
        if span is None:
            continue
        start, end = span
        tags[start] = f"B-{field.upper()}"
        for i in range(start + 1, end):
            tags[i] = f"I-{field.upper()}"
        for i in range(start, end):
            taken[i] = True
    return tags


def build_sample(name: str, tokens: List[Dict[str, Any]], entities: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """One training sample (same format as data/processed/*.json) from one receipt's OCR tokens."""
    words, boxes = tokens_to_words(tokens)
    if not words:
        return None
    tags = align_ocr_words(words, entities)
    return {
        "id": os.path.splitext(name)[0],
        "words": words,
        "bboxes": boxes,
        "ner_tags": [LABEL2ID[t] for t in tags],
        "bio_tags": tags,
        "entities": entities,
    }


def label_coverage(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Share of receipts in which each field could be located in the OCR text (the rest are labelled 'O')."""
    out = {}
    for field in ALL_FIELDS:
        tag = f"B-{field.upper()}"
        hit = sum(1 for s in samples if tag in s["bio_tags"] and s["entities"].get(field))
        have = sum(1 for s in samples if s["entities"].get(field))
        out[field] = round(hit / have, 3) if have else 0.0
    return out


def prepare_ocr_dataset(
    data_dir: str = "data/sroie",
    cache_root: str = "outputs/ocr_cache",
    out_dir: str = "data/ocr",
    fast: bool = True,
    limit: Optional[int] = None,
    pipeline: Any = None,
) -> Dict[str, Any]:
    """
    Download the SROIE photos of every split, read them with the real pipeline (cached; safe to
    stop and rerun), label the OCR words from the dataset's answers and write
    `out_dir/train.json` (train + val photos) and `out_dir/test.json` (test photos).

    limit: photos per split (None or 0 = all of them). Use a small number for a quick trial.
    """
    cache_dir = os.path.join(cache_root, "mobile" if fast else "server")
    pipeline = pipeline or LazyPipeline(fast=fast)

    samples: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        truth = download_sroie(split=split, data_dir=data_dir, limit=limit or None, fields=ALL_FIELDS)
        names = list(truth)
        print(f"\n[{split}] {len(names)} photos")
        stats = run_ocr(names, os.path.join(data_dir, split), cache_dir, pipeline)
        print(f"[{split}] OCR: {stats['done']} new, {stats['cached']} from cache, {stats['errors']} failed")

        built = []
        for name in names:
            tokens = _load_tokens(cache_dir, name)
            sample = build_sample(name, tokens, truth[name]) if tokens else None
            if sample is not None:
                built.append(sample)
        samples[split] = built

    train = samples["train"] + samples["val"]
    test = samples["test"]
    if {s["id"] for s in train} & {s["id"] for s in test}:
        raise ValueError("The same receipt appears in both train and test.")

    os.makedirs(out_dir, exist_ok=True)
    for fname, data in (("train.json", train), ("test.json", test)):
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            json.dump(data, f)

    summary = {
        "train_receipts": len(train),
        "test_receipts": len(test),
        "train_label_coverage": label_coverage(train),
        "test_label_coverage": label_coverage(test),
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"OCR dataset written to {out_dir}/  ({len(train)} train, {len(test)} test receipts)")
    print("Share of receipts where the answer was found in the OCR text and labelled:")
    for split_name, cov in (("train", summary["train_label_coverage"]), ("test", summary["test_label_coverage"])):
        print(f"  {split_name:5}: " + "  ".join(f"{k} {v:.0%}" for k, v in cov.items()))
    print("=" * 60)
    return summary
