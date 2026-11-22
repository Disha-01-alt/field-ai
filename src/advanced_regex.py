"""
Advanced Regex & Spatial Information Extraction Engine.

Features:
1. Spatial coordinate-aware layout understanding (Line clustering, multi-column pairing).
2. Advanced Date extraction with OCR error correction and multi-syntax matching.
3. Advanced Total extraction with horizontal key-value alignment (e.g. 'TOTAL' on left, '$45.20' on right).
4. Advanced Company extraction with header prominence, multi-line chaining, and noise suppression.
"""

import re
import unicodedata
from typing import Dict, List, Any, Tuple, Optional


class OCRToken:
    """Represents a recognized OCR text unit with 2D spatial bounding box."""

    def __init__(
        self,
        text: str,
        box: List[int],  # [x0, y0, x1, y1] normalized to [0, 1000]
        confidence: float = 1.0,
        raw_box: Optional[List[List[float]]] = None,
    ):
        self.text = str(text).strip()
        self.box = [int(round(c)) for c in box]
        self.confidence = float(confidence)
        self.raw_box = raw_box

        self.x0, self.y0, self.x1, self.y1 = self.box
        self.width = max(self.x1 - self.x0, 1)
        self.height = max(self.y1 - self.y0, 1)
        self.cx = (self.x0 + self.x1) / 2.0
        self.cy = (self.y0 + self.y1) / 2.0

    def __repr__(self):
        return f"OCRToken({self.text!r}, [{self.x0},{self.y0},{self.x1},{self.y1}])"


class OCRLine:
    """A line of text formed by grouping spatially contiguous tokens."""

    def __init__(self, tokens: List[OCRToken]):
        self.tokens = sorted(tokens, key=lambda t: t.x0)
        self.text = " ".join(t.text for t in self.tokens if t.text)
        if tokens:
            self.x0 = min(t.x0 for t in tokens)
            self.y0 = min(t.y0 for t in tokens)
            self.x1 = max(t.x1 for t in tokens)
            self.y1 = max(t.y1 for t in tokens)
            self.confidence = sum(t.confidence for t in tokens) / len(tokens)
        else:
            self.x0 = self.y0 = self.x1 = self.y1 = 0
            self.confidence = 0.0

        self.width = max(self.x1 - self.x0, 1)
        self.height = max(self.y1 - self.y0, 1)
        self.cx = (self.x0 + self.x1) / 2.0
        self.cy = (self.y0 + self.y1) / 2.0

    def __repr__(self):
        return f"OCRLine({self.text!r}, y={self.y0}-{self.y1})"


def cluster_tokens_into_lines(tokens: List[OCRToken], y_tolerance: float = 12.0) -> List[OCRLine]:
    """
    Cluster OCR tokens into reading lines using vertical overlap and y-center proximity.
    """
    if not tokens:
        return []

    # Sort tokens primarily by y0, secondarily by x0
    sorted_tokens = sorted(tokens, key=lambda t: (t.y0, t.x0))
    line_clusters: List[List[OCRToken]] = []

    for token in sorted_tokens:
        if not token.text.strip():
            continue

        placed = False
        for cluster in line_clusters:
            # Check vertical overlap with cluster's tokens
            cluster_y0 = min(t.y0 for t in cluster)
            cluster_y1 = max(t.y1 for t in cluster)
            cluster_cy = sum(t.cy for t in cluster) / len(cluster)
            avg_h = max(cluster_y1 - cluster_y0, token.height)

            # Two tokens belong to the same line if their vertical centers are close
            if abs(token.cy - cluster_cy) <= max(y_tolerance, avg_h * 0.45):
                cluster.append(token)
                placed = True
                break

        if not placed:
            line_clusters.append([token])

    # Convert clusters to OCRLine and sort vertically by y0
    lines = [OCRLine(cluster) for cluster in line_clusters if cluster]
    lines.sort(key=lambda line: (line.y0, line.x0))
    return lines


class AdvancedRegexExtractor:
    """
    Hybrid spatial-regex information extraction engine for receipts.
    Extracts Company, Date, and Grand Total.
    """

    COMPANY_NOISE_EXACT = {
        "tax invoice", "simplified tax invoice", "tax receipt",
        "invoice", "receipt", "official receipt", "sales slip",
        "cash sale", "cash receipt", "sales invoice", "bill",
        "welcome", "thank you", "welcome to", "thanks for coming",
        "original copy", "customer copy", "merchant copy", "store copy",
        "credit invoice", "debit note", "order slip", "kitchen slip",
    }

    # Lines that are never the merchant name, e.g. "TQ FOR SHOPPING WITH ...",
    # "(CO. NO. 123456-X)", "GST REG NO: ...", web addresses.
    COMPANY_NOISE_PATTERNS = [
        r"^\W*(tq|thank|thanks|welcome)\b",
        r"\bshopping with\b",
        r"^\W*(co|company|roc|br|reg|gst|sst)\b[\s\.]*(reg|no|id|#)",
        r"www\.|\.com\b|https?:",
    ]

    # Company registration number stuck to the end of a name, e.g.
    # "AEON CO. (M) BHD (126926-H)", "... S/B 273500-U (KSB)", "(CO.NO. 1208604-T)".
    REG_NO_TAIL = re.compile(
        r"[\s,]*[\(\[]?\s*"
        r"(?:(?:co(?:mpany)?|reg|roc|br)[\s\.]*(?:reg[\s\.]*)?(?:no|#)?[\s\.:]*)?"
        r"[a-z]{0,3}\d{4,}(?:[\s\-]*[a-z])?\s*[\)\]]?"
        r"(?:\s*\([a-z]{2,4}\))?[\s,.]*$",
        re.IGNORECASE,
    )

    # Words that start the transaction part of a receipt. A name found below
    # these is usually the customer / bill-to name, not the merchant.
    BODY_START_PATTERN = re.compile(
        r"\b(date|tarikh|inv(?:oice)?\s*(?:no|#)|receipt\s*(?:no|#)|bill\s*(?:no|#)|"
        r"slip\s*no|cashier|table|terminal|counter|doc\s*no)\b"
        r"|\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b",
        re.IGNORECASE,
    )

    COMPANY_LEGAL_SUFFIXES = [
        r"\bsdn[\s\.]*bhd\b",
        r"\bs/b\b",
        r"\bberhad\b",
        r"\bsdn\b",
        r"\bbhd\b",
        r"\benterprise\b",
        r"\benterprises\b",
        r"\btrading\b",
        r"\bltd\b",
        r"\blimited\b",
        r"\binc\b",
        r"\bcorp\b",
        r"\bcorporation\b",
        r"\bllc\b",
        r"\bplc\b",
        r"\bplt\b",
        r"\bpte\b",
        r"\bco\b",
    ]

    COMPANY_BUSINESS_TERMS = [
        "restaurant", "restoran", "cafe", "coffee", "bistro", "bakery",
        "supermarket", "hypermarket", "mini market", "market", "mart",
        "store", "shop", "station", "stationery", "pharmacy", "farmasi",
        "hardware", "electrical", "telecom", "optical", "fashion",
        "boutique", "hotel", "resort", "inn", "car wash", "auto",
        "bookstore", "books", "grocer", "florist", "salon",
    ]

    ADDRESS_INDICATORS = [
        r"\bjalan\b", r"\bjln\b", r"\bstreet\b", r"\bst\b", r"\broad\b", r"\brd\b",
        r"\bavenue\b", r"\bave\b", r"\btaman\b", r"\bblok\b", r"\bblock\b",
        r"\bno\.?\s*\d+", r"\blorong\b", r"\bpersiaran\b", r"\bbandar\b", r"\bkg\.", r"\blot\s*\d+", r"\bunit\s*\d+", r"\blevel\s*\d+",
        r"\bfloor\b", r"\bflr\b", r"\bposkod\b", r"\bpostcode\b", r"\bpostal\b",
        r"\bkuala lumpur\b", r"\bselangor\b", r"\bpenang\b", r"\bjohor\b",
        r"\bmalaysia\b", r"\bsingapore\b", r"\btel\b", r"\bphone\b", r"\bfax\b",
        r"\bemail\b", r"\bwww\.", r"\bhttp", r"@"
    ]

    TOTAL_POSITIVE_LABELS = {
        "grand total": 20,
        "total payable": 20,
        "amount payable": 20,
        "total amount payable": 20,
        "net payable": 18,
        "amount due": 18,
        "balance due": 17,
        "total amount due": 18,
        "total due": 17,
        "total amt": 16,
        "final total": 16,
        "total bill": 15,
        "invoice total": 15,
        "total sales": 14,
        "jumlah": 14,
        "jumlah besar": 18,
        "net total": 15,
        "nett total": 16,
        "net amount": 14,
        "total after": 18,
        "after adj": 18,
        "rounded total": 18,
        "total rounded": 18,
        "total nett": 15,
        "total (rm)": 16,
        "total (myr)": 16,
        "total": 10,
    }

    TOTAL_NEGATIVE_LABELS = {
        "subtotal": 16,
        "sub total": 16,
        "sub-total": 16,
        "tax total": 14,
        "gst total": 14,
        "taxable amt": 14,
        "total taxable": 14,
        "tax": 9,
        "gst": 9,
        "sst": 9,
        "vat": 9,
        "service charge": 12,
        "svc chg": 12,
        "discount": 12,
        "cash": 11,
        "change": 14,
        "kembali": 14,
        "rounding": 10,
        "round off": 10,
        "total items": 12,
        "total qty": 12,
        "qty": 8,
        "amount paid": 10,
        "paid": 9,
        "tender": 11,
        "tendered": 11,
        "unit price": 10,
        "price": 7,
        "excluding": 14,
        "exclusive": 14,
        "excl": 14,
        "saving": 14,
        "savings": 14,
        "you save": 14,
        "deposit": 10,
        "points": 10,
    }

    TAX_WORDS = {"tax", "gst", "sst", "vat"}

    # Start of the tax summary table printed under the total
    # ("GST SUMMARY", "TAX CODE  %  AMT", "CODE % NET AMT GST TOTAL").
    TAX_SUMMARY_PATTERN = re.compile(
        r"\b(gst|tax|sst|vat)\s+summary\b|\btax\s*code\b|\bcode\s+%",
        re.IGNORECASE,
    )
    # A row of that table, e.g. "SR 6 381.50 22.89", "S = GST @6%: 68.82", "T @ 6% 28.39".
    TAX_ROW_PATTERN = re.compile(
        r"^\s*(sr|zr|zrl|s|z|t|e|ze)\s*[=@:]?\s*(gst\s*)?@?\s*\d{1,2}(\.\d+)?\s*%?(\s|$)",
        re.IGNORECASE,
    )

    AMOUNT_REGEX = re.compile(
        r"(?:(?:rm|myr|sgd|usd|inr|eur|gbp)\s*|[$€£₹]\s*)?"
        r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2}))"
        r"(?!\d)",
        re.IGNORECASE,
    )

    MONTH_NAMES = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
        "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
        "january": "01", "february": "02", "march": "03", "april": "04", "june": "06",
        "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12"
    }

    # -------------------------------------------------------------
    # 1. DATE EXTRACTION
    # -------------------------------------------------------------

    @classmethod
    def _clean_date_ocr_noise(cls, text: str) -> str:
        """Fix common character recognition errors in date candidates."""
        # Fix 'O'/'o' mistaken for '0' inside numeric tokens
        t = re.sub(r"(?<=\d)[oO](?=\d)|(?<=\d)[oO]\b|\b[oO](?=\d)", "0", text)
        # Fix 'l'/'I' mistaken for '1' or '/' inside date patterns
        t = re.sub(r"(?<=\d)[lI](?=\d)", "1", t)
        return t

    @classmethod
    def extract_date(cls, lines: List[OCRLine]) -> str:
        """
        Extract transaction date using multi-pattern matching and context scoring.
        """
        date_candidates: List[Tuple[float, str]] = []

        patterns = [
            # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
            (re.compile(r"\b([0-3]?[0-9])[/.\-]([0-1]?[0-9])[/.\-]((?:20|19)\d{2})\b"), "dmy"),
            # YYYY/MM/DD or YYYY-MM-DD
            (re.compile(r"\b((?:20|19)\d{2})[/.\-]([0-1]?[0-9])[/.\-]([0-3]?[0-9])\b"), "ymd"),
            # DD/MM/YY
            (re.compile(r"\b([0-3]?[0-9])[/.\-]([0-1]?[0-9])[/.\-](\d{2})\b"), "dmy_short"),
            # DD Mon YYYY (e.g. 15 Jan 2019)
            (re.compile(
                r"\b([0-3]?[0-9])[\s\-\./]([a-zA-Z]{3,9})[\s\-\./]((?:20|19)?\d{2})\b",
                re.IGNORECASE,
            ), "d_mon_y"),
            # Mon DD YYYY (e.g. Jan 15, 2019)
            (re.compile(
                r"\b([a-zA-Z]{3,9})[\s\-\.]([0-3]?[0-9]),?[\s\-\.]((?:20|19)?\d{2})\b",
                re.IGNORECASE,
            ), "mon_d_y"),
        ]

        for line_idx, line in enumerate(lines):
            raw_text = cls._clean_date_ocr_noise(line.text)
            lower = raw_text.lower()

            # Skip lines with high false-positive indicators (tax reg, phone numbers)
            if any(term in lower for term in ["gst reg", "roc", "tel:", "fax:", "co reg"]):
                continue

            has_date_keyword = any(kw in lower for kw in ["date", "tarikh", "dt:", "time", "jam"])

            for pattern, p_type in patterns:
                for match in pattern.finditer(raw_text):
                    score = 10.0
                    if has_date_keyword:
                        score += 5.0

                    matched_str = match.group(0).strip()
                    parts = match.groups()

                    try:
                        if p_type == "dmy":
                            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                        elif p_type == "ymd":
                            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                        elif p_type == "dmy_short":
                            day, month = int(parts[0]), int(parts[1])
                            year = 2000 + int(parts[2])
                            score -= 1.0  # slight penalty for ambiguous 2-digit year
                        elif p_type == "d_mon_y":
                            day = int(parts[0])
                            mon_str = parts[1].lower()[:3]
                            if mon_str not in cls.MONTH_NAMES:
                                continue
                            month = int(cls.MONTH_NAMES[mon_str])
                            year = int(parts[2])
                            if year < 100:
                                year += 2000
                        elif p_type == "mon_d_y":
                            mon_str = parts[0].lower()[:3]
                            if mon_str not in cls.MONTH_NAMES:
                                continue
                            month = int(cls.MONTH_NAMES[mon_str])
                            day = int(parts[1])
                            year = int(parts[2])
                            if year < 100:
                                year += 2000
                        else:
                            continue

                        # Plausibility checks
                        if not (1 <= day <= 31 and 1 <= month <= 12 and 1990 <= year <= 2035):
                            continue

                        # Prefer upper receipt region (dates are typically top half)
                        rel_y = line.y0 / 1000.0
                        if rel_y < 0.5:
                            score += 2.0

                        date_candidates.append((score, matched_str))

                    except (ValueError, TypeError):
                        continue

        if not date_candidates:
            return ""

        date_candidates.sort(key=lambda x: x[0], reverse=True)
        return date_candidates[0][1]

    # -------------------------------------------------------------
    # 2. TOTAL EXTRACTION (SPATIAL & KEY-VALUE ALIGNMENT)
    # -------------------------------------------------------------

    @classmethod
    def _find_label_match(cls, text: str, label_dict: Dict[str, int], ignore: Optional[set] = None) -> int:
        """Find max weight of matched whole-word label (labels in `ignore` are skipped)."""
        lower = text.lower()
        best_weight = 0
        for label, weight in label_dict.items():
            if ignore and label in ignore:
                continue
            pattern = r"(?<![a-z0-9])" + re.escape(label) + r"(?![a-z0-9])"
            if re.search(pattern, lower):
                best_weight = max(best_weight, weight)
        return best_weight

    @classmethod
    def _line_amounts(cls, text: str) -> List[Tuple[float, str, bool]]:
        """All money amounts in a line as (value, 'x.xx', has_currency). Negative amounts are skipped."""
        amounts = []
        for match in cls.AMOUNT_REGEX.finditer(text):
            # "-2.98", "- RM0.02" are discounts / rounding adjustments, never the total
            if text[:match.start()].rstrip().endswith("-"):
                continue
            try:
                val = float(match.group(1).replace(",", ""))
            except ValueError:
                continue
            if val <= 0.0 or val > 500000.0:
                continue
            full_match = match.group(0).lower()
            has_currency = any(curr in full_match for curr in ["rm", "myr", "$", "usd", "sgd", "inr"])
            amounts.append((val, f"{val:.2f}", has_currency))
        return amounts

    @classmethod
    def extract_total(cls, lines: List[OCRLine]) -> str:
        """
        Extract grand total using spatial key-value proximity and candidate ranking.
        Handles two-column layouts where 'TOTAL' is on left and amount is on right.
        """
        if not lines:
            return ""

        # Everything from the tax summary table downwards repeats sub-amounts
        # (net, tax) next to the word TOTAL, so it must not win.
        summary_start = len(lines)
        for idx, line in enumerate(lines):
            if cls.TAX_SUMMARY_PATTERN.search(line.text):
                summary_start = idx
                break

        candidates: List[Dict[str, Any]] = []

        # Step 1: Detect all monetary candidates across all lines
        for line_idx, line in enumerate(lines):
            lower = line.text.lower()
            # "Total incl. GST" / "GST included in total"
            is_incl_tax = bool(re.search(r"\b(incl|inclusive|including)\b.*\b(tax|gst|sst|vat)\b", lower))
            line_pos_weight = cls._find_label_match(line.text, cls.TOTAL_POSITIVE_LABELS)
            ignore = set()
            if is_incl_tax:
                ignore |= cls.TAX_WORDS        # "TOTAL INCL. GST" is the total, not a tax line
            if line_pos_weight > 0:
                ignore.add("cash")             # "CASH TOTAL ..." is still a total
            line_neg_weight = cls._find_label_match(line.text, cls.TOTAL_NEGATIVE_LABELS, ignore)
            if re.search(r"\bincluded in total\b", lower):
                line_neg_weight = max(line_neg_weight, 14)   # the tax part of the total
            has_own_label = line_pos_weight > 0 or line_neg_weight > 0

            for val, formatted_val, has_currency in cls._line_amounts(line.text):
                score = 0.0
                score += line_pos_weight * 2.0
                score -= line_neg_weight * 2.5
                if is_incl_tax and line_pos_weight > 0:
                    score += 5.0

                if has_currency:
                    score += 3.0

                # Prefer lower receipt area (totals are usually at bottom 30% of receipt)
                rel_y = line.y0 / 1000.0
                if rel_y >= 0.5:
                    score += 4.0 * rel_y

                # Amount printed on its own line: use the label on the line above
                if line_idx > 0 and not has_own_label:
                    prev_line = lines[line_idx - 1]
                    prev_pos = cls._find_label_match(prev_line.text, cls.TOTAL_POSITIVE_LABELS)
                    prev_neg = cls._find_label_match(prev_line.text, cls.TOTAL_NEGATIVE_LABELS)
                    score += prev_pos * 1.2
                    score -= prev_neg * 1.5

                if line_idx >= summary_start or cls.TAX_ROW_PATTERN.search(line.text):
                    score -= 30.0

                candidates.append({
                    "val": formatted_val,
                    "num": val,
                    "score": score,
                    "line_idx": line_idx,
                    "y0": line.y0,
                    "has_label": line_pos_weight > 0,
                    "line_text": line.text,
                })

        # Step 2: Spatial Multi-Column Pairing
        # Often 'TOTAL' is its own token box on the left, and amount is a separate token on the right
        for line_idx, line in enumerate(lines):
            if line_idx >= summary_start:
                break
            line_pos_weight = cls._find_label_match(line.text, cls.TOTAL_POSITIVE_LABELS)
            if line_pos_weight >= 10:
                for cand in candidates:
                    # Same horizontal band (+/- 25 units in normalized height)
                    if abs(cand["y0"] - line.y0) <= 25:
                        cand["score"] += line_pos_weight * 1.8

        if not candidates:
            return ""

        # Filter: if any candidates have direct label association, discard unlabelled ones
        labelled_cands = [c for c in candidates if c["score"] >= 8.0]
        if labelled_cands:
            candidates = labelled_cands

        # Sort by score descending, then by vertical position (lower on receipt wins ties)
        candidates.sort(key=lambda c: (c["score"], c["y0"]), reverse=True)
        best = candidates[0]

        # Step 3: Rounding. If a "ROUNDING" line follows the chosen total, the
        # amount payable is the rounded one (e.g. 22.91 -> ROUNDING -0.01 -> 22.90).
        i = best["line_idx"]
        for j in range(i + 1, min(i + 4, summary_start, len(lines))):
            if "round" not in lines[j].text.lower():
                continue
            for k in range(j, min(j + 3, summary_start, len(lines))):
                for val, formatted_val, _ in cls._line_amounts(lines[k].text):
                    if 0.0 < abs(val - best["num"]) <= 0.05:
                        return formatted_val
            break

        return best["val"]

    # -------------------------------------------------------------
    # 3. COMPANY EXTRACTION (HEADER PROMINENCE & NOISE SUPPRESSION)
    # -------------------------------------------------------------

    @classmethod
    def clean_company_name(cls, text: str) -> str:
        """Remove registration numbers glued to the end of a name: 'AEON CO. (M) BHD (126926-H)' -> 'AEON CO. (M) BHD'."""
        cleaned = text.strip()
        for _ in range(3):
            new = cls.REG_NO_TAIL.sub("", cleaned).strip()
            if new == cleaned:
                break
            cleaned = new
        return cleaned.strip(" ,:;-")

    @classmethod
    def _is_company_noise(cls, text: str) -> bool:
        """Check if line is a generic receipt header or noise line."""
        norm = re.sub(r"[^\w\s]", "", text).lower().strip()
        if not norm or len(norm) < 3:
            return True
        if norm in cls.COMPANY_NOISE_EXACT:
            return True
        for noise in cls.COMPANY_NOISE_EXACT:
            if norm == noise or norm.startswith(noise + " ") or norm.endswith(" " + noise):
                return True
        lower = text.lower().strip()
        if any(re.search(pat, lower) for pat in cls.COMPANY_NOISE_PATTERNS):
            return True
        # Only a registration number, e.g. "(CO.NO. 1208604-T)" or "822737-X"
        if sum(c.isalpha() for c in cls.clean_company_name(text)) < 3:
            return True
        return False

    @classmethod
    def _is_address_line(cls, text: str) -> bool:
        """Check if line is part of a merchant postal address or contact info."""
        lower = text.lower()
        # A 5-digit postcode or a comma-separated list ("19, JALAN KANCIL, OFF JALAN PUDU")
        if re.search(r"(?<![\d\-])\d{5}(?![\d\-])", lower) and "," in lower:
            return True
        matches = 1 if lower.count(",") >= 2 else 0
        for pattern in cls.ADDRESS_INDICATORS:
            if re.search(pattern, lower):
                matches += 1
                if matches >= 2:
                    return True
        return False

    @classmethod
    def _has_legal_suffix(cls, text: str) -> bool:
        lower = text.lower()
        return any(re.search(pat, lower) for pat in cls.COMPANY_LEGAL_SUFFIXES)

    @classmethod
    def _is_name_continuation(cls, first: str, second: str) -> bool:
        """
        True when `second` finishes a merchant name started on `first`, e.g.
        'POPULAR BOOK' + 'CO. (M) SDN BHD', 'HOME MASTER HARDWARE &' + 'ELECTRICAL',
        'AIK HUAT HARDWARE ENTERPRISE (SETIA' + 'ALAM) SDN BHD'.
        """
        a = first.strip()
        if a.endswith(("&", "-", ",")) or a.count("(") > a.count(")"):
            return True
        b = second.strip().lower()
        # Next line starts with a legal suffix: "CO. (M) SDN BHD", "SDN BHD", "BHD."
        first_word = re.split(r"[\s\(\)]+", b)[0] if b else ""
        if first_word and cls._has_legal_suffix(first_word.rstrip(".,")):
            return True
        if first_word in ("(m)", "(m).") or b.startswith(("(m)", "& ")):
            return True
        return False

    @classmethod
    def _score_company(cls, text: str, height: float, rank: int, below_body: bool) -> float:
        lower = text.lower()
        score = 5.0
        score += min(height / 10.0, 5.0)              # font height prominence
        if text.isupper():
            score += 2.0
        if cls._has_legal_suffix(text):
            score += 6.0
        if any(term in lower for term in cls.COMPANY_BUSINESS_TERMS):
            score += 3.0
        score += max(0.0, 6.0 - 1.5 * rank)            # merchant name is printed first
        if below_body:
            score -= 8.0                               # below date / invoice no -> customer name
        if re.search(r"\b(gst|roc|reg|co\s*no|table|cashier|invoice|bill)\b", lower):
            score -= 4.0
        return score

    @classmethod
    def extract_company(cls, lines: List[OCRLine]) -> str:
        """
        Extract merchant name by inspecting the top header zone, evaluating
        font prominence, legal suffixes, business types, and multi-line merging.
        """
        if not lines:
            return ""

        # Only search within the top 35% of the receipt
        header_lines = [line for line in lines if line.y0 <= 350]
        if not header_lines:
            header_lines = lines[:8]

        body_start = len(header_lines)
        for idx, line in enumerate(header_lines):
            if cls.BODY_START_PATTERN.search(line.text):
                body_start = idx
                break

        def usable(text: str) -> bool:
            return (not cls._is_company_noise(text)
                    and not cls._is_address_line(text)
                    and sum(c.isalpha() for c in text) >= 3)

        candidates: List[Tuple[float, str]] = []
        rank = 0
        for idx, line in enumerate(header_lines):
            text = line.text.strip()
            if not usable(text):
                continue
            below_body = idx > body_start
            candidates.append((cls._score_company(text, line.height, rank, below_body), text))

            # Name printed over two or three lines: join the following lines
            merged, height = text, line.height
            for nxt in header_lines[idx + 1: idx + 3]:
                nxt_text = nxt.text.strip()
                if not usable(nxt_text) or not cls._is_name_continuation(merged, nxt_text):
                    break
                merged = f"{merged} {nxt_text}"
                height = max(height, nxt.height)
                candidates.append((cls._score_company(merged, height, rank, below_body) + 3.0, merged))
            rank += 1

        if not candidates:
            return ""

        candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        return cls.clean_company_name(candidates[0][1])

    # -------------------------------------------------------------
    # 4. UNIFIED INTERFACE
    # -------------------------------------------------------------

    @classmethod
    def extract_fields(cls, tokens: List[OCRToken]) -> Dict[str, str]:
        """
        Unified extraction entry point taking token list and returning
        {company, date, total}.
        """
        lines = cluster_tokens_into_lines(tokens)
        return {
            "company": cls.extract_company(lines),
            "date": cls.extract_date(lines),
            "total": cls.extract_total(lines),
        }

