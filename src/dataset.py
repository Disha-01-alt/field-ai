
"""
Dataset Ingestion, Coordinate Normalization, and BIO Alignment
for SROIE Receipts.

Features:
1. Parses SROIE polygon bounding boxes.
2. Converts polygons into axis-aligned bounding boxes.
3. Normalizes coordinates to [0, 1000] for LayoutLM.
4. Uses actual image dimensions whenever available.
5. Performs normalized BIO entity alignment.
6. Supports company, date, total, and address entities.
7. Exports reproducible train/test JSON splits.

Important:
SROIE box files provide line-level coordinates.
Each word from a line receives that original line-level box.
This implementation does not fabricate word-level coordinates.
"""

import os
import re
import json
import random
import warnings
import unicodedata

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


# ============================================================
# 1. LABEL SCHEMA
# ============================================================

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

ID2LABEL = {
    i: label for i, label in enumerate(LABELS)
}

LABEL2ID = {
    label: i for i, label in enumerate(LABELS)
}

ENTITY_FIELDS = [
    "date",
    "total",
    "company",
    "address",
]

CURRENCY_TOKENS = {
    "rm", "myr", "sgd", "usd", "inr",
    "eur", "gbp", "$", "€", "£", "₹",
}


# ============================================================
# 2. TEXT NORMALIZATION
# ============================================================

def clean_text(s: str) -> str:
    """
    Basic text normalization.

    Preserves punctuation because punctuation can be meaningful
    in dates, monetary values, and company names.
    """
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def normalize_entity_token(
    token: str,
    field: str,
) -> Optional[str]:
    """
    Normalize one OCR or ground-truth token for entity alignment.

    Company:
        Ignore punctuation and case.

    Date:
        Ignore punctuation and case.

    Total:
        Normalize currency and decimal representation.
    """

    token = unicodedata.normalize(
        "NFKC",
        str(token)
    ).lower().strip()

    if not token:
        return None

    if field == "total":

        # Remove currency symbols and codes.
        token = re.sub(
            r"\b(rm|myr|sgd|usd|inr|eur|gbp)\b",
            "",
            token,
        )

        token = (
            token.replace("$", "")
            .replace("€", "")
            .replace("£", "")
            .replace("₹", "")
            .replace(",", "")
            .strip()
        )

        if not token:
            return None

        match = re.search(
            r"-?\d+(?:\.\d+)?",
            token
        )

        if not match:
            return None

        try:
            value = Decimal(match.group(0))

            if not value.is_finite():
                return None

            return format(
                value.quantize(Decimal("0.01")),
                "f"
            )

        except InvalidOperation:
            return None

    if field == "date":
        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            token,
        )

        return normalized or None

    if field in ("company", "address"):
        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            token,
        )

        return normalized or None

    normalized = re.sub(
        r"\s+",
        "",
        token,
    )

    return normalized or None


# ============================================================
# 3. BOUNDING BOX NORMALIZATION
# ============================================================

def normalize_bbox(
    bbox: List[int],
    width: int = 1000,
    height: int = 1000,
) -> List[int]:
    """
    Normalize [x0, y0, x1, y1] into LayoutLM's [0, 1000] domain.

    Handles:
    - Negative coordinates
    - Reversed coordinates
    - Coordinates outside page boundaries
    - Zero-area boxes
    """

    if len(bbox) != 4:
        raise ValueError(
            f"Expected 4 bounding box coordinates, got {len(bbox)}"
        )

    width = max(int(width), 1)
    height = max(int(height), 1)

    x0, y0, x1, y1 = map(float, bbox)

    x_min = min(x0, x1)
    x_max = max(x0, x1)

    y_min = min(y0, y1)
    y_max = max(y0, y1)

    # Clamp coordinates to actual page dimensions.
    x_min = min(max(x_min, 0), width)
    x_max = min(max(x_max, 0), width)

    y_min = min(max(y_min, 0), height)
    y_max = min(max(y_max, 0), height)

    # Normalize to [0, 1000].
    norm_x0 = round(x_min / width * 1000)
    norm_y0 = round(y_min / height * 1000)

    norm_x1 = round(x_max / width * 1000)
    norm_y1 = round(y_max / height * 1000)

    norm_x0 = min(max(norm_x0, 0), 1000)
    norm_y0 = min(max(norm_y0, 0), 1000)

    norm_x1 = min(max(norm_x1, 0), 1000)
    norm_y1 = min(max(norm_y1, 0), 1000)

    # Ensure positive area even at the page boundary.
    if norm_x1 <= norm_x0:
        if norm_x0 >= 1000:
            norm_x0 = 999
        norm_x1 = norm_x0 + 1

    if norm_y1 <= norm_y0:
        if norm_y0 >= 1000:
            norm_y0 = 999
        norm_y1 = norm_y0 + 1

    return [
        norm_x0,
        norm_y0,
        norm_x1,
        norm_y1,
    ]


# ============================================================
# 4. SROIE BOX PARSING
# ============================================================

def parse_sroie_box_line(
    line: str,
) -> Optional[Tuple[List[int], str]]:
    """
    Parse a SROIE line:

    x1,y1,x2,y2,x3,y3,x4,y4,text

    Returns:
        ([xmin, ymin, xmax, ymax], text)
    """

    line = line.strip()

    if not line:
        return None

    parts = line.split(",")

    if len(parts) < 9:
        return None

    try:
        coords = [
            int(part.strip())
            for part in parts[:8]
        ]

    except ValueError:
        return None

    text = ",".join(parts[8:]).strip()

    if not text:
        return None

    xs = coords[0::2]
    ys = coords[1::2]

    bbox = [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]

    return bbox, text


# ============================================================
# 5. IMAGE DIMENSION DETECTION
# ============================================================

def find_image_dimensions(
    box_path: str,
) -> Optional[Tuple[int, int]]:
    """
    Find the corresponding receipt image and obtain its dimensions.

    Expected common SROIE structure:

        train/
            box/
            img/
            key/

    Returns:
        (width, height), or None if unavailable.
    """

    box_path = Path(box_path)

    stem = box_path.stem

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
    ]

    image_directories = [
        box_path.parent.parent / "img",
        box_path.parent.parent / "image",
        box_path.parent.parent / "images",
        box_path.parent,
    ]

    image_path = None

    for directory in image_directories:

        for extension in image_extensions:

            candidate = directory / (
                stem + extension
            )

            if candidate.exists():
                image_path = candidate
                break

        if image_path is not None:
            break

    if image_path is None:
        return None

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size

        if width > 0 and height > 0:
            return width, height

    except ImportError:
        warnings.warn(
            "Pillow is unavailable. "
            "Install it to read actual receipt image dimensions."
        )

    except Exception as exc:
        warnings.warn(
            f"Could not read image dimensions for {image_path}: {exc}"
        )

    return None


# ============================================================
# 6. ENTITY ALIGNMENT HELPERS
# ============================================================

def _get_entity_tokens(
    value: str,
    field: str,
) -> List[str]:
    """
    Convert a ground-truth entity into normalized tokens.

    Currency-only tokens are ignored for total alignment.
    """

    if not value:
        return []

    tokens = []

    for raw_token in str(value).split():

        normalized = normalize_entity_token(
            raw_token,
            field,
        )

        if normalized:
            tokens.append(normalized)

    return tokens


def _find_entity_spans(
    words: List[str],
    target_tokens: List[str],
    field: str,
) -> List[Tuple[int, int]]:
    """
    Find exact normalized contiguous token sequences.

    Returns:
        List of (start_index, end_index) pairs.
        End index is exclusive.
    """

    if not target_tokens:
        return []

    normalized_words = [
        normalize_entity_token(word, field)
        for word in words
    ]

    spans = []
    target_length = len(target_tokens)

    for start in range(len(words) - target_length + 1):

        window = normalized_words[
            start:start + target_length
        ]

        if window == target_tokens:
            spans.append(
                (start, start + target_length)
            )

    return spans


# ============================================================
# 7. IMPROVED BIO ALIGNMENT
# ============================================================

def align_tokens_with_entities(
    words: List[str],
    entities: Dict[str, str],
) -> List[str]:
    """
    Assign BIO labels to OCR words.

    Improvements:
    - Normalized punctuation handling.
    - Company-name multi-token alignment.
    - Currency-aware total alignment.
    - Decimal normalization.
    - Prevents overlapping entity assignments.
    - Avoids labeling every duplicate occurrence.

    Matching is exact after normalization. It deliberately
    avoids aggressive fuzzy matching that could create
    incorrect ground-truth labels.
    """

    tags = ["O"] * len(words)

    if not words or not entities:
        return tags

    normalized_entities = {
        str(key).lower(): value
        for key, value in entities.items()
    }

    # Date first, followed by total, company, address.
    field_order = [
        "date",
        "total",
        "company",
        "address",
    ]

    for field in field_order:

        gt_value = normalized_entities.get(
            field,
            ""
        )

        if not gt_value:
            continue

        target_tokens = _get_entity_tokens(
            gt_value,
            field,
        )

        if not target_tokens:
            continue

        spans = _find_entity_spans(
            words,
            target_tokens,
            field,
        )

        if not spans:
            continue

        # A duplicate monetary amount may appear elsewhere
        # in a receipt. Prefer the last occurrence for total.
        # For company and date, prefer the first occurrence.
        if field == "total":
            spans.sort(
                key=lambda span: span[0],
                reverse=True,
            )
        else:
            spans.sort(
                key=lambda span: span[0]
            )

        selected_span = None

        for start, end in spans:

            if all(
                tags[index] == "O"
                for index in range(start, end)
            ):
                selected_span = (start, end)
                break

        if selected_span is None:
            continue

        start, end = selected_span

        tags[start] = f"B-{field.upper()}"

        for index in range(start + 1, end):
            tags[index] = f"I-{field.upper()}"

    return tags


# ============================================================
# 8. LOAD SINGLE SROIE DOCUMENT
# ============================================================

def load_sroie_document(
    box_path: str,
    key_path: str,
) -> Optional[Dict[str, Any]]:
    """
    Load a receipt document.

    Output:
        id
        words
        bboxes
        ner_tags
        bio_tags
        entities
    """

    if not os.path.exists(box_path):
        return None

    if not os.path.exists(key_path):
        return None

    try:
        with open(
            key_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:

            entities = json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(entities, dict):
        return None

    words = []
    raw_boxes = []

    try:
        with open(
            box_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:

            for line in file:

                parsed = parse_sroie_box_line(
                    line
                )

                if parsed is None:
                    continue

                box, text = parsed

                # Preserve OCR word order.
                tokens = text.split()

                if not tokens:
                    continue

                for token in tokens:

                    words.append(token)

                    # SROIE provides line-level coordinates.
                    # Preserve the original line-level bbox.
                    raw_boxes.append(box)

    except OSError:
        return None

    if not words:
        return None

    # --------------------------------------------------------
    # Determine actual receipt dimensions.
    # --------------------------------------------------------

    dimensions = find_image_dimensions(
        box_path
    )

    if dimensions is not None:

        page_width, page_height = dimensions

    else:

        # Fallback when image files are unavailable.
        # This is an approximation, not true page dimensions.
        page_width = max(
            max(box[2] for box in raw_boxes),
            1,
        )

        page_height = max(
            max(box[3] for box in raw_boxes),
            1,
        )

        warnings.warn(
            f"Image dimensions unavailable for "
            f"{Path(box_path).stem}. "
            f"Using maximum OCR coordinates as an approximation."
        )

    normalized_boxes = [
        normalize_bbox(
            box,
            width=page_width,
            height=page_height,
        )
        for box in raw_boxes
    ]

    # --------------------------------------------------------
    # Generate BIO tags.
    # --------------------------------------------------------

    bio_tags = align_tokens_with_entities(
        words,
        entities,
    )

    tag_ids = [
        LABEL2ID.get(tag, LABEL2ID["O"])
        for tag in bio_tags
    ]

    doc_id = Path(box_path).stem

    return {
        "id": doc_id,
        "words": words,
        "bboxes": normalized_boxes,
        "ner_tags": tag_ids,
        "bio_tags": bio_tags,
        "entities": entities,
    }


# ============================================================
# 9. LOAD COMPLETE SROIE DATASET
# ============================================================

def load_sroie_dataset(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """
    Scan a directory containing SROIE box and key files.

    Supports:
        .csv
        .txt
        .json

    Common directory structure:

        data/sroie/
            train/
                box/
                img/
                key/
            test/
                box/
                img/
                key/
    """

    data_path = Path(data_dir)

    box_files = (
        sorted(data_path.glob("**/box/*.csv"))
        + sorted(data_path.glob("**/box/*.txt"))
        + sorted(data_path.glob("*.csv"))
        + sorted(data_path.glob("*.txt"))
    )

    # Deduplicate by actual file path, not filename alone.
    unique_box_files = []
    seen_paths = set()

    for box_file in box_files:

        resolved_path = str(
            box_file.resolve()
        )

        if resolved_path in seen_paths:
            continue

        seen_paths.add(resolved_path)

        unique_box_files.append(box_file)

    samples = []

    for box_file in unique_box_files:

        stem = box_file.stem

        key_candidates = [
            box_file.parent.parent
            / "key"
            / f"{stem}.json",

            box_file.parent
            / f"{stem}.json",

            box_file.parent.parent
            / "entities"
            / f"{stem}.json",
        ]

        key_path = next(
            (
                path
                for path in key_candidates
                if path.exists()
            ),
            None,
        )

        if key_path is None:
            continue

        document = load_sroie_document(
            str(box_file),
            str(key_path),
        )

        if document is not None:
            samples.append(document)

    return samples


# ============================================================
# 10. EXPORT TRAIN / TEST SPLITS
# ============================================================

def export_splits(
    data_dir: str = "data/sroie",
    output_dir: str = "data/processed",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]]
]:
    """
    Load SROIE and export reproducible train/test splits.

    IMPORTANT:
    Use the same seed and unchanged dataset ordering when
    reproducing an existing benchmark split.

    Do not regenerate the split after model training if the
    existing test set is intended to remain frozen.
    """

    if not 0.0 < test_ratio < 1.0:
        raise ValueError(
            "test_ratio must be between 0 and 1."
        )

    samples = load_sroie_dataset(
        data_dir
    )

    print(
        f"Total documents loaded: {len(samples)}"
    )

    if len(samples) < 2:
        print(
            "Not enough documents to create train/test splits."
        )
        return [], []

    # Stable ordering before deterministic shuffle.
    samples.sort(
        key=lambda sample: sample["id"]
    )

    rng = random.Random(seed)

    rng.shuffle(samples)

    split_idx = int(
        len(samples) * (1 - test_ratio)
    )

    split_idx = min(
        max(split_idx, 1),
        len(samples) - 1,
    )

    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    train_file = os.path.join(
        output_dir,
        "train.json",
    )

    test_file = os.path.join(
        output_dir,
        "test.json",
    )

    with open(
        train_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            train_samples,
            file,
            ensure_ascii=False,
        )

    with open(
        test_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            test_samples,
            file,
            ensure_ascii=False,
        )

    print(
        f"Exported {len(train_samples)} training documents "
        f"to {train_file}"
    )

    print(
        f"Exported {len(test_samples)} testing documents "
        f"to {test_file}"
    )

    return train_samples, test_samples


# ============================================================
# 11. DATASET DIAGNOSTICS
# ============================================================

def print_dataset_diagnostics(
    samples: List[Dict[str, Any]],
) -> None:
    """Print entity and token statistics."""

    print("\n" + "=" * 55)
    print("DATASET DIAGNOSTICS")
    print("=" * 55)

    print(
        f"Documents: {len(samples)}"
    )

    total_tokens = 0

    entity_counts = {
        field: 0
        for field in ENTITY_FIELDS
    }

    tagged_counts = {
        field: 0
        for field in ENTITY_FIELDS
    }

    for sample in samples:

        total_tokens += len(
            sample["words"]
        )

        for field in ENTITY_FIELDS:

            if sample["entities"].get(field):
                entity_counts[field] += 1

            prefix = field.upper()

            tagged_counts[field] += sum(
                1
                for tag in sample["bio_tags"]
                if tag in (
                    f"B-{prefix}",
                    f"I-{prefix}",
                )
            )

    print(
        f"Total OCR tokens: {total_tokens}"
    )

    print("\nEntity coverage:")

    for field in ENTITY_FIELDS:

        print(
            f"{field.upper():<12} "
            f"Ground truth: {entity_counts[field]:>5} | "
            f"Tagged tokens: {tagged_counts[field]:>5}"
        )

    print("=" * 55)


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        f"Loaded {len(LABELS)} BIO classes:"
    )

    print(LABELS)

    # Demonstration normalization.
    test_box = [
        50, 100,
        350, 400,
    ]

    normalized = normalize_bbox(
        test_box,
        width=1000,
        height=1000,
    )

    print(
        f"Test normalization: "
        f"{test_box} -> {normalized}"
    )

    if os.path.exists("data/sroie"):

        train_samples, test_samples = export_splits()

        print_dataset_diagnostics(
            train_samples
        )

        print_dataset_diagnostics(
            test_samples
        )
