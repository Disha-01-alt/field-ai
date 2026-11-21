
"""
Dataset Ingestion, Coordinate Normalization, and BIO Alignment for SROIE Receipts.

This module processes raw SROIE Task 3 files:
1. Parses polygon bounding boxes [x1, y1, x2, y2, x3, y3, x4, y4, text] into axis-aligned [x0, y0, x1, y1]
2. Normalizes coordinates to the integer domain [0, 1000] required by LayoutLM
3. Aligns extracted text tokens with ground truth entities to assign BIO tags
4. Exports standardized JSON / Hugging Face compatible dataset splits
"""

import os
import re
import glob
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


# Canonical label schema
LABELS = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-DATE",
    "I-DATE",
    "B-TOTAL",
    "I-TOTAL",
    "B-ADDRESS",
    "I-ADDRESS",
]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def normalize_bbox(bbox: List[int], width: int = 1000, height: int = 1000) -> List[int]:
    """
    Normalize bounding box [x0, y0, x1, y1] to [0, 1000] scale.
    Guarantees x0 <= x1 and y0 <= y1 and values within [0, 1000].
    """
    x0, y0, x1, y1 = bbox
    # Ensure proper orientation
    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)

    # Scale to 0-1000
    norm_x0 = int(min(max(0, (x_min / width) * 1000), 1000))
    norm_y0 = int(min(max(0, (y_min / height) * 1000), 1000))
    norm_x1 = int(min(max(0, (x_max / width) * 1000), 1000))
    norm_y1 = int(min(max(0, (y_max / height) * 1000), 1000))

    # Avoid zero-area degradation by adding 1 unit if equal
    if norm_x1 <= norm_x0:
        norm_x1 = min(1000, norm_x0 + 1)
    if norm_y1 <= norm_y0:
        norm_y1 = min(1000, norm_y0 + 1)

    return [norm_x0, norm_y0, norm_x1, norm_y1]


def parse_sroie_box_line(line: str) -> Optional[Tuple[List[int], str]]:
    """
    Parse a single SROIE box line: x1,y1,x2,y2,x3,y3,x4,y4,text_string
    Handles commas embedded inside the text string.
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split(",")
    if len(parts) < 9:
        return None

    try:
        coords = [int(p.strip()) for p in parts[:8]]
    except ValueError:
        return None

    # Text may contain commas
    text = ",".join(parts[8:]).strip()
    if not text:
        return None

    # Compute bounding box
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return [x_min, y_min, x_max, y_max], text


def clean_text(s: str) -> str:
    """Normalize whitespace and strip punctuation for alignment."""
    return re.sub(r"\s+", " ", s).strip().lower()


def align_tokens_with_entities(
    words: List[str],
    entities: Dict[str, str],
) -> List[str]:
    """
    Assign BIO tags to words by matching against ground truth entities.
    Prioritizes specific entities: DATE -> TOTAL -> COMPANY -> ADDRESS.
    """
    tags = ["O"] * len(words)
    joined_words = " ".join(words).lower()

    # Priority of entity matching
    field_order = ["date", "total", "company", "address"]

    for field in field_order:
        gt_value = entities.get(field, "")
        if not gt_value:
            continue

        gt_tokens = clean_text(gt_value).split()
        if not gt_tokens:
            continue

        gt_len = len(gt_tokens)
        cleaned_words = [clean_text(w) for w in words]

        # Scan for matching sub-sequences of words
        for i in range(len(words) - gt_len + 1):
            window = cleaned_words[i : i + gt_len]
            # Check exact match of sequence
            if window == gt_tokens:
                # Check if not already tagged
                if all(tags[i + k] == "O" for k in range(gt_len)):
                    tags[i] = f"B-{field.upper()}"
                    for k in range(1, gt_len):
                        tags[i + k] = f"I-{field.upper()}"

        # Soft matching fallback for single-token total or date
        if field in ("total", "date"):
            norm_gt = clean_text(gt_value).replace("$", "").replace("rm", "").strip()
            for i, w in enumerate(cleaned_words):
                if tags[i] == "O" and norm_gt and norm_gt == w.replace("$", "").replace("rm", "").strip():
                    tags[i] = f"B-{field.upper()}"

    return tags


def load_sroie_document(box_path: str, key_path: str) -> Optional[Dict[str, Any]]:
    """Parse a single receipt document (box file + key JSON file)."""
    if not os.path.exists(box_path) or not os.path.exists(key_path):
        return None

    try:
        with open(key_path, "r", encoding="utf-8", errors="ignore") as f:
            entities = json.load(f)
    except Exception:
        return None

    words = []
    raw_boxes = []

    with open(box_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_sroie_box_line(line)
            if parsed:
                box, text = parsed
                # Split line into words for word-level tokens
                subwords = text.split()
                if not subwords:
                    continue
                # For simplicity, assign the line bbox to each word in the line
                # (or linearly interpolate if desired)
                for w in subwords:
                    words.append(w)
                    raw_boxes.append(box)

    if not words:
        return None

    # Estimate page width and height from coordinates
    page_w = max([b[2] for b in raw_boxes] + [1000])
    page_h = max([b[3] for b in raw_boxes] + [1000])

    normalized_boxes = [normalize_bbox(b, page_w, page_h) for b in raw_boxes]
    bio_tags = align_tokens_with_entities(words, entities)
    tag_ids = [LABEL2ID.get(t, 0) for t in bio_tags]

    doc_id = Path(box_path).stem
    return {
        "id": doc_id,
        "words": words,
        "bboxes": normalized_boxes,
        "ner_tags": tag_ids,
        "bio_tags": bio_tags,
        "entities": entities,
    }


def load_sroie_dataset(data_dir: str) -> List[Dict[str, Any]]:
    """
    Scan a directory containing box/ and key/ subdirectories,
    supporting both .csv and .txt box files with .json keys.
    """
    data_path = Path(data_dir)
    box_files = (
        sorted(data_path.glob("**/box/*.csv"))
        + sorted(data_path.glob("**/box/*.txt"))
        + sorted(data_path.glob("*.csv"))
        + sorted(data_path.glob("*.txt"))
    )
    # Deduplicate in case patterns overlap
    seen_stems = set()
    unique_box_files = []
    for bf in box_files:
        if bf.stem not in seen_stems:
            seen_stems.add(bf.stem)
            unique_box_files.append(bf)
    
    samples = []
    for bf in unique_box_files:
        stem = bf.stem
        # Look in key/ subdirectory or same dir
        key_candidates = [
            bf.parent.parent / "key" / f"{stem}.json",
            bf.parent / f"{stem}.json",
            bf.parent.parent / "entities" / f"{stem}.json",
        ]
        key_path = next((p for p in key_candidates if p.exists()), None)
        if key_path:
            doc = load_sroie_document(str(bf), str(key_path))
            if doc:
                samples.append(doc)

    return samples


def export_splits(
    data_dir: str = "data/sroie",
    output_dir: str = "data/processed",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load SROIE dataset, split into frozen train/test sets, and save as JSON.
    """
    import random
    random.seed(seed)

    samples = load_sroie_dataset(data_dir)
    print(f"Total documents loaded for splitting: {len(samples)}")
    if not samples:
        return [], []

    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - test_ratio))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train.json")
    test_file = os.path.join(output_dir, "test.json")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_samples, f)
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_samples, f)

    print(f"Exported {len(train_samples)} train samples to {train_file}")
    print(f"Exported {len(test_samples)} test samples to {test_file}")

    return train_samples, test_samples


if __name__ == "__main__":
    print(f"Loaded {len(LABELS)} BIO classes: {LABELS}")
    # Demonstration normalization
    test_box = [50, 100, 350, 400]
    norm = normalize_bbox(test_box, width=1000, height=1000)
    print(f"Test normalization: {test_box} -> {norm}")
    
    # Process and export splits if data exists
    if os.path.exists("data/sroie"):
        export_splits()
