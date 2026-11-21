
"""
Baseline 1: OCR + Regular Expression Field Extractor for SROIE Receipts.

Improved version:
- Date: Malaysian and international date formats
- Company: Multi-line, candidate-based company extraction
- Total: Context-aware monetary candidate ranking
- Evaluation: Original field-wise evaluation pipeline preserved
"""

import os
import re
import json
import random
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from src.metrics import compute_field_metrics


class RegexReceiptExtractor:
    """Deterministic rule-based receipt information extractor."""

    # ---------------------------------------------------------
    # 1. COMPANY EXTRACTION CONFIGURATION
    # ---------------------------------------------------------

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
        "customer copy",
        "merchant copy",
    ]

    COMPANY_LEGAL_TERMS = (
        "sdn", "bhd", "berhad", "enterprise",
        "pte", "ltd", "limited", "inc", "llc",
        "plc", "corp", "corporation",
    )

    COMPANY_BUSINESS_TERMS = (
        "restaurant", "bakery", "cafe", "coffee",
        "market", "mart", "hotel", "store",
        "trading", "services", "company",
        "supermarket", "pharmacy",
    )

    ADDRESS_TERMS = (
        "jalan", "street", "road", "avenue",
        "highway", "postcode", "postal",
        "telephone", "phone", "fax",
        "www.", "http", "@",
        "registration no", "company no",
        "roc", "gst reg",
    )

    # ---------------------------------------------------------
    # 2. TOTAL EXTRACTION CONFIGURATION
    # ---------------------------------------------------------

    AMOUNT_PATTERN = re.compile(
        r"(?<![\w])"
        r"(?:(?:rm|myr|sgd|usd|inr)\s*|[$€£₹]\s*)?"
        r"((?:\d{1,3}(?:,\d{3})+|\d+)"
        r"(?:\.\d{1,2})?)"
        r"(?!\d)",
        re.IGNORECASE,
    )

    POSITIVE_TOTAL_LABELS = {
        "grand total": 15,
        "total payable": 15,
        "amount payable": 15,
        "amount due": 14,
        "balance due": 14,
        "net payable": 14,
        "net total": 13,
        "invoice total": 13,
        "total amount": 12,
        "total": 9,
        "jumlah": 10,
    }

    NEGATIVE_AMOUNT_LABELS = {
        "subtotal": 13,
        "sub total": 13,
        "sub-total": 13,
        "tax total": 12,
        "tax": 7,
        "gst": 7,
        "sst": 7,
        "discount": 9,
        "cash": 9,
        "change": 11,
        "rounding": 7,
        "item": 6,
        "qty": 6,
        "unit price": 8,
        "price each": 8,
        "amount paid": 8,
        "paid": 6,
        "tender": 9,
    }

    # Retain old constants for compatibility with existing code.
    TOTAL_KEYWORDS = list(POSITIVE_TOTAL_LABELS.keys())

    EXCLUDE_TOTAL_KEYWORDS = list(
        NEGATIVE_AMOUNT_LABELS.keys()
    )

    PRICE_PATTERN = re.compile(
        r"(?:rm|myr|\$)?\s*"
        r"([0-9]{1,4}(?:,[0-9]{3})*\.[0-9]{2})\b",
        re.IGNORECASE,
    )

    # ---------------------------------------------------------
    # 3. DATE EXTRACTION CONFIGURATION
    # ---------------------------------------------------------

    DATE_PATTERNS = [
        re.compile(
            r"\b([0-3]?[0-9][/.\-][0-1]?[0-9][/.\-](?:20|19)?\d{2})\b"
        ),
        re.compile(
            r"\b((?:20|19)\d{2}[/.\-][0-1]?[0-9][/.\-][0-3]?[0-9])\b"
        ),
        re.compile(
            r"\b([0-3]?[0-9][\s\-\.]"
            r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
            r"[a-z]*[\s\-\.](?:20|19)?\d{2})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
            r"[a-z]*[\s\-\.][0-3]?[0-9],?[\s\-\.](?:20|19)?\d{2})\b",
            re.IGNORECASE,
        ),
    ]

    # ---------------------------------------------------------
    # 4. GENERAL HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace without destroying meaningful text."""
        return re.sub(r"\s+", " ", str(text)).strip()

    @staticmethod
    def _get_box_geometry(
        box: Optional[List[int]]
    ) -> Tuple[float, float, float, float]:
        """
        Return x_min, y_min, x_max, y_max.

        Supports 4-coordinate and 8-coordinate OCR boxes.
        """
        if not box:
            return 0.0, 0.0, 0.0, 0.0

        try:
            values = [float(v) for v in box]

            if len(values) >= 8:
                xs = values[0::2]
                ys = values[1::2]

                return (
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys),
                )

            if len(values) == 4:
                return (
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                )

        except (ValueError, TypeError):
            pass

        return 0.0, 0.0, 0.0, 0.0

    @classmethod
    def _is_noise_company_line(cls, text: str) -> bool:
        lower = text.lower()

        for noise in cls.COMPANY_NOISE:
            if lower == noise or lower.startswith(noise + " "):
                return True

        return False

    @classmethod
    def _is_address_line(cls, text: str) -> bool:
        lower = text.lower()

        return any(
            term in lower
            for term in cls.ADDRESS_TERMS
        )

    @staticmethod
    def _has_date(text: str) -> bool:
        for pattern in RegexReceiptExtractor.DATE_PATTERNS:
            if pattern.search(text):
                return True

        return False

    # ---------------------------------------------------------
    # 5. DATE EXTRACTION
    # ---------------------------------------------------------

    @classmethod
    def extract_date(cls, lines: List[str]) -> str:
        """Extract transaction date from receipt lines."""

        for line in lines:
            lower = line.lower()

            if any(
                keyword in lower
                for keyword in ["gst", "reg", "tel", "fax", "roc"]
            ):
                continue

            for pattern in cls.DATE_PATTERNS:
                match = pattern.search(line)

                if match:
                    date_str = match.group(1).strip()

                    if sum(c.isdigit() for c in date_str) >= 4:
                        return date_str

        return ""

    # ---------------------------------------------------------
    # 6. COMPANY EXTRACTION
    # ---------------------------------------------------------

    @classmethod
    def _score_company_candidate(
        cls,
        text: str,
        index: int,
        total_lines: int,
    ) -> float:
        """Score a possible merchant/company name."""

        text = cls._normalize_text(text)
        lower = text.lower()

        if len(text) < 3:
            return -100.0

        if not any(c.isalpha() for c in text):
            return -100.0

        if cls._is_noise_company_line(text):
            return -100.0

        if cls._has_date(text):
            return -100.0

        if cls._is_address_line(text):
            # A legal company suffix can appear on the same line
            # as a company name, so do not reject it automatically.
            has_legal_term = any(
                re.search(
                    r"\b" + re.escape(term) + r"\b",
                    lower
                )
                for term in cls.COMPANY_LEGAL_TERMS
            )

            if not has_legal_term:
                return -100.0

        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)

        if letters < 3:
            return -100.0

        if digits > letters:
            return -100.0

        words = re.findall(
            r"[A-Za-z][A-Za-z&'.()\-]*",
            text
        )

        score = 2.0

        # More meaningful alphabetic content.
        score += min(letters / 12.0, 3.0)

        # Reward plausible multi-word company names.
        score += min(len(words), 6) * 0.35

        # Uppercase text is common in merchant headers.
        if text.isupper():
            score += 1.5

        # Legal company suffixes are strong indicators.
        if any(
            re.search(
                r"\b" + re.escape(term) + r"\b",
                lower
            )
            for term in cls.COMPANY_LEGAL_TERMS
        ):
            score += 4.0

        # Common business-related terms.
        if any(term in lower for term in cls.COMPANY_BUSINESS_TERMS):
            score += 1.0

        # Header position is useful, but not decisive.
        position_score = 1.0 - (
            index / max(total_lines, 1)
        )

        score += 3.0 * position_score

        # Penalize likely registration and invoice identifiers.
        if re.search(r"\b\d{4,}\b", text):
            score -= 2.0

        if re.search(
            r"\b(invoice|receipt|tax|cashier|date|time|table)\b",
            lower
        ):
            score -= 2.5

        # Very long lines are more likely to contain addresses.
        if len(text) > 90:
            score -= 2.0

        return score

    @classmethod
    def _can_merge_company_lines(
        cls,
        first: Tuple[str, List[int]],
        second: Tuple[str, List[int]],
    ) -> bool:
        """Determine whether two neighboring lines can form one name."""

        text_a, box_a = first
        text_b, box_b = second

        lower_a = text_a.lower()
        lower_b = text_b.lower()

        if cls._is_noise_company_line(text_b):
            return False

        if cls._has_date(text_b):
            return False

        if cls._is_address_line(text_b):
            return False

        if len(text_a) + len(text_b) > 100:
            return False

        # A second line containing a legal suffix is a strong
        # candidate for a company-name continuation.
        has_legal_suffix = any(
            re.search(
                r"\b" + re.escape(term) + r"\b",
                lower_b
            )
            for term in cls.COMPANY_LEGAL_TERMS
        )

        if has_legal_suffix:
            return True

        # Compare vertical spacing and horizontal alignment.
        x1, y1, x2, y2 = cls._get_box_geometry(box_a)
        bx1, by1, bx2, by2 = cls._get_box_geometry(box_b)

        if x2 > x1 and bx2 > bx1:
            height_a = max(y2 - y1, 1.0)
            height_b = max(by2 - by1, 1.0)

            vertical_gap = by1 - y2

            horizontally_aligned = (
                abs(bx1 - x1)
                <= max(height_a, height_b) * 2.5
            )

            close_vertically = (
                -height_a <= vertical_gap
                <= max(height_a, height_b) * 2.0
            )

            if not (horizontally_aligned and close_vertically):
                return False

        # Short uppercase continuation lines are plausible.
        if (
            len(text_b.split()) <= 3
            and text_b.isupper()
            and len(text_b) <= 35
        ):
            return True

        return False

    @classmethod
    def extract_company(
        cls,
        lines_with_boxes: List[Tuple[str, List[int]]],
    ) -> str:
        """
        Extract merchant name using candidate scoring.

        Evaluates individual lines and adjacent multi-line candidates.
        """

        if not lines_with_boxes:
            return ""

        # The merchant name is commonly in the header, but we
        # inspect more than the original eight lines.
        limit = min(len(lines_with_boxes), 15)

        candidates = []

        for idx in range(limit):
            text, box = lines_with_boxes[idx]
            text = cls._normalize_text(text)

            score = cls._score_company_candidate(
                text,
                idx,
                limit,
            )

            if score > -50:
                candidates.append((score, idx, text))

        # Generate multi-line candidates.
        # Do not invent word-level boxes: use the original line boxes.
        for start in range(limit):
            for length in (2, 3):
                end = start + length

                if end > limit:
                    continue

                group = lines_with_boxes[start:end]

                if not all(
                    cls._can_merge_company_lines(
                        group[i],
                        group[i + 1]
                    )
                    for i in range(len(group) - 1)
                ):
                    continue

                merged_text = cls._normalize_text(
                    " ".join(item[0] for item in group)
                )

                score = cls._score_company_candidate(
                    merged_text,
                    start,
                    limit,
                )

                # A multi-line candidate must have evidence
                # that it is a continuation, not just extra text.
                if score > -50:
                    score += 1.0
                    candidates.append(
                        (score, start, merged_text)
                    )

        if not candidates:
            return ""

        candidates.sort(
            key=lambda item: (
                item[0],
                -item[1],
                min(len(item[2]), 80),
            ),
            reverse=True,
        )

        return candidates[0][2]

    # ---------------------------------------------------------
    # 7. TOTAL EXTRACTION
    # ---------------------------------------------------------

    @classmethod
    def _find_label_score(
        cls,
        text: str,
        labels: Dict[str, int],
    ) -> int:
        """Find the strongest whole-word label in a text line."""

        lower = text.lower()
        best_score = 0

        for label, weight in labels.items():
            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(label)
                + r"(?![a-z0-9])"
            )

            if re.search(pattern, lower):
                best_score = max(best_score, weight)

        return best_score

    @staticmethod
    def _normalize_amount(raw: str) -> Optional[str]:
        """Normalize a monetary candidate to two decimal places."""

        try:
            value = float(raw.replace(",", ""))

            if value < 0 or value > 1_000_000_000:
                return None

            return f"{value:.2f}"

        except (ValueError, TypeError):
            return None

    @classmethod
    def extract_total(
        cls,
        lines_with_boxes: List[Tuple[str, List[int]]],
    ) -> str:
        """
        Extract the most likely final payable amount.

        Uses:
        - Same-line positive labels
        - Nearby positive labels
        - Negative labels
        - Currency and decimal formatting
        - Vertical receipt position

        Does not automatically choose the largest or last amount.
        """

        if not lines_with_boxes:
            return ""

        candidates = []
        total_lines = len(lines_with_boxes)

        for idx, (line, box) in enumerate(lines_with_boxes):
            lower = line.lower()

            # A date should not be treated as a monetary amount.
            if cls._has_date(line):
                continue

            positive_score = cls._find_label_score(
                line,
                cls.POSITIVE_TOTAL_LABELS,
            )

            negative_score = cls._find_label_score(
                line,
                cls.NEGATIVE_AMOUNT_LABELS,
            )

            # Check neighboring lines for split labels.
            previous_positive = 0
            next_positive = 0
            previous_negative = 0
            next_negative = 0

            if idx > 0:
                previous_line = lines_with_boxes[idx - 1][0]

                previous_positive = cls._find_label_score(
                    previous_line,
                    cls.POSITIVE_TOTAL_LABELS,
                )

                previous_negative = cls._find_label_score(
                    previous_line,
                    cls.NEGATIVE_AMOUNT_LABELS,
                )

            if idx + 1 < total_lines:
                next_line = lines_with_boxes[idx + 1][0]

                next_positive = cls._find_label_score(
                    next_line,
                    cls.POSITIVE_TOTAL_LABELS,
                )

                next_negative = cls._find_label_score(
                    next_line,
                    cls.NEGATIVE_AMOUNT_LABELS,
                )

            matches = list(cls.AMOUNT_PATTERN.finditer(line))

            for match in matches:
                raw_amount = match.group(1)

                normalized = cls._normalize_amount(raw_amount)

                if normalized is None:
                    continue

                matched_text = match.group(0).lower()

                has_currency = bool(
                    re.search(
                        r"\b(rm|myr|sgd|usd|inr)\b|[$€£₹]",
                        matched_text,
                    )
                )

                has_decimal = "." in raw_amount

                # Integer-only values without currency or a
                # total label are likely IDs or quantities.
                if (
                    not has_currency
                    and not has_decimal
                    and positive_score == 0
                    and previous_positive == 0
                    and next_positive == 0
                ):
                    continue

                score = 0.0

                # Strongest evidence: label on the same line.
                score += positive_score * 1.5

                # Nearby labels are useful when OCR separates
                # "TOTAL" and the actual monetary value.
                score += previous_positive * 0.75
                score += next_positive * 0.55

                # Currency and decimal formatting.
                if has_currency:
                    score += 2.5

                if has_decimal:
                    score += 1.5

                # Negative label penalties.
                score -= negative_score * 1.5
                score -= previous_negative * 0.6
                score -= next_negative * 0.6

                # "Total including tax" is still a valid total.
                if re.search(
                    r"\b(incl|including|inclusive)\s+(tax|gst|sst)\b",
                    lower,
                ):
                    score += 4.0

                # Prefer lower receipt regions as supporting
                # evidence, not as the deciding factor.
                position_score = idx / max(total_lines - 1, 1)
                score += position_score * 2.0

                # Keep metadata for debugging and candidate ranking.
                candidates.append({
                    "value": normalized,
                    "score": score,
                    "line_index": idx,
                    "raw_line": line,
                    "has_currency": has_currency,
                    "has_decimal": has_decimal,
                })

        if not candidates:
            return ""

        # Prefer candidates with actual positive total-label evidence.
        labelled = [
            candidate
            for candidate in candidates
            if (
                cls._find_label_score(
                    candidate["raw_line"],
                    cls.POSITIVE_TOTAL_LABELS,
                ) > 0
                or (
                    candidate["line_index"] > 0
                    and cls._find_label_score(
                        lines_with_boxes[
                            candidate["line_index"] - 1
                        ][0],
                        cls.POSITIVE_TOTAL_LABELS,
                    ) > 0
                )
                or (
                    candidate["line_index"] + 1 < total_lines
                    and cls._find_label_score(
                        lines_with_boxes[
                            candidate["line_index"] + 1
                        ][0],
                        cls.POSITIVE_TOTAL_LABELS,
                    ) > 0
                )
            )
        ]

        if labelled:
            candidates = labelled

        else:
            # No explicit total label was found.
            # Keep only plausible currency/decimal candidates.
            candidates = [
                candidate
                for candidate in candidates
                if (
                    candidate["has_currency"]
                    or candidate["has_decimal"]
                )
            ]

            if not candidates:
                return ""

            # In the absence of a label, avoid pretending that
            # a heuristic fallback is certain.
            for candidate in candidates:
                candidate["score"] -= 4.0

        candidates.sort(
            key=lambda candidate: (
                candidate["score"],
                candidate["line_index"],
            ),
            reverse=True,
        )

        best = candidates[0]

        # If all candidates are strongly negative, abstain.
        if best["score"] < -5.0:
            return ""

        return best["value"]

    # ---------------------------------------------------------
    # 8. COMPLETE FIELD EXTRACTION
    # ---------------------------------------------------------

    @classmethod
    def extract_fields(
        cls,
        lines_with_boxes: List[Tuple[str, List[int]]],
    ) -> Dict[str, str]:
        """Extract company, date, and total."""

        lines = [
            line
            for line, _ in lines_with_boxes
        ]

        return {
            "company": cls.extract_company(lines_with_boxes),
            "date": cls.extract_date(lines),
            "total": cls.extract_total(lines_with_boxes),
        }


# -------------------------------------------------------------
# 9. OCR BOX FILE LOADER
# -------------------------------------------------------------

def load_lines_from_box_file(
    file_path: str,
) -> List[Tuple[str, List[int]]]:
    """Load text lines and quadrilateral bounding boxes."""

    lines_with_boxes = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 9:
                continue

            try:
                coords = [
                    int(part.strip())
                    for part in parts[:8]
                ]

                text = ",".join(parts[8:]).strip()

                if text:
                    lines_with_boxes.append(
                        (text, coords)
                    )

            except ValueError:
                continue

    return lines_with_boxes


# -------------------------------------------------------------
# 10. BASELINE EVALUATION
# -------------------------------------------------------------

def run_baseline_evaluation(
    data_dir: str = "data/sroie",
    processed_test_file: str = "data/processed/test.json",
    output_path: str = "results/regex_baseline.json",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate the OCR + Regex baseline.

    Uses the existing processed test split when available.
    Otherwise, loads raw SROIE box and key files.
    """

    predictions = {}
    ground_truths = {}

    # ---------------------------------------------------------
    # A. Load the existing processed test split
    # ---------------------------------------------------------

    if os.path.exists(processed_test_file):

        print(
            f"Loading test split from {processed_test_file}..."
        )

        with open(
            processed_test_file,
            "r",
            encoding="utf-8",
        ) as file:

            test_data = json.load(file)

        print(
            f"Evaluating Baseline 1 on "
            f"{len(test_data)} test receipts..."
        )

        for item in test_data:

            doc_id = item["id"]

            words = item["words"]
            boxes = item["bboxes"]

            # Reconstruct lines from identical OCR boxes.
            lines = []
            current_box = None
            current_words = []

            for word, box in zip(words, boxes):

                if current_box is None:
                    current_box = box
                    current_words = [word]

                elif box == current_box:
                    current_words.append(word)

                else:
                    lines.append((
                        " ".join(current_words),
                        current_box,
                    ))

                    current_box = box
                    current_words = [word]

            if current_words:
                lines.append((
                    " ".join(current_words),
                    current_box,
                ))

            pred = RegexReceiptExtractor.extract_fields(lines)

            predictions[doc_id] = pred
            ground_truths[doc_id] = item["entities"]

    # ---------------------------------------------------------
    # B. Fallback: load raw SROIE files
    # ---------------------------------------------------------

    else:

        random.seed(seed)

        data_path = Path(data_dir)

        box_files = (
            sorted(data_path.glob("**/box/*.csv"))
            or sorted(data_path.glob("**/box/*.txt"))
        )

        paired_docs = []

        for box_file in box_files:

            stem = box_file.stem

            key_candidates = [
                box_file.parent.parent
                / "key"
                / f"{stem}.json",

                box_file.parent
                / f"{stem}.json",
            ]

            key_file = next(
                (
                    path
                    for path in key_candidates
                    if path.exists()
                ),
                None,
            )

            if key_file is None:
                continue

            try:
                with open(
                    key_file,
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                ) as file:

                    ground_truth = json.load(file)

                paired_docs.append((
                    box_file,
                    ground_truth,
                ))

            except (OSError, json.JSONDecodeError):
                continue

        print(
            f"Total valid paired documents: "
            f"{len(paired_docs)}"
        )

        if not paired_docs:
            print("Error: No paired documents found.")
            return {}

        random.shuffle(paired_docs)

        split_idx = int(
            len(paired_docs) * (1 - test_ratio)
        )

        test_docs = paired_docs[split_idx:]

        print(
            f"Evaluating Baseline 1 on test set of "
            f"{len(test_docs)} documents..."
        )

        for box_file, ground_truth in test_docs:

            doc_id = box_file.stem

            lines = load_lines_from_box_file(
                str(box_file)
            )

            pred = RegexReceiptExtractor.extract_fields(
                lines
            )

            predictions[doc_id] = pred
            ground_truths[doc_id] = ground_truth

    # ---------------------------------------------------------
    # C. Compute metrics
    # ---------------------------------------------------------

    metrics = compute_field_metrics(
        predictions,
        ground_truths,
    )

    output_data = {
        "model": "OCR + Regex Baseline",
        "modality": "Plain Text (Rules)",
        "test_size": len(predictions),
        "metrics": metrics,
        "sample_predictions": {
            key: predictions[key]
            for key in list(predictions.keys())[:5]
        },
    }

    os.makedirs(
        os.path.dirname(
            os.path.abspath(output_path)
        ),
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # D. Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 55)
    print("Baseline 1 (OCR + Regex) Evaluation Results")
    print("=" * 55)

    for field in [
        "company",
        "date",
        "total",
        "macro_avg",
    ]:

        result = metrics.get(field, {})

        print(
            f"{field.upper():<12} | "
            f"Precision: {result.get('precision', 0):.4f} | "
            f"Recall: {result.get('recall', 0):.4f} | "
            f"F1: {result.get('f1', 0):.4f}"
        )

    print("=" * 55)

    print(
        f"Results saved to {output_path}"
    )

    return output_data


# -------------------------------------------------------------
# 11. MAIN
# -------------------------------------------------------------

if __name__ == "__main__":
    run_baseline_evaluation()
