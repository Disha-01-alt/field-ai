"""
Tests for Preprocessing and Advanced Regex Engine.
"""

import numpy as np
import pytest

from src.preprocessing import (
    to_grayscale,
    resize_bounded,
    enhance_contrast,
    denoise,
    adaptive_binarize,
    preprocess_receipt,
)
from src.advanced_regex import (
    OCRToken,
    OCRLine,
    cluster_tokens_into_lines,
    AdvancedRegexExtractor,
)


def test_preprocessing_synthetic_image():
    # Create synthetic receipt image
    synthetic_img = np.ones((1200, 800, 3), dtype=np.uint8) * 240
    # Add text-like dark bars
    synthetic_img[100:130, 200:600] = 30
    synthetic_img[200:220, 150:400] = 30
    synthetic_img[800:830, 150:300] = 30
    synthetic_img[800:830, 500:650] = 30

    processed, meta = preprocess_receipt(synthetic_img, deskew_enabled=False)
    assert processed is not None
    assert processed.shape[0] > 0 and processed.shape[1] > 0
    assert "scale_factor" in meta


def test_advanced_regex_extraction():
    # Mock tokens representing a receipt layout
    tokens = [
        # Company header (top 150 units)
        OCRToken("MAKASSAR", [150, 50, 400, 90]),
        OCRToken("FRESH", [420, 50, 600, 90]),
        OCRToken("MARKET", [620, 50, 850, 90]),
        OCRToken("SDN.", [300, 100, 450, 140]),
        OCRToken("BHD.", [470, 100, 600, 140]),
        
        # Address & Noise
        OCRToken("NO. 12, JALAN MEDAN BUKIT INDAH", [100, 160, 900, 190]),
        OCRToken("TAX INVOICE", [350, 210, 650, 240]),

        # Date & Cashier
        OCRToken("DATE:", [100, 260, 220, 290]),
        OCRToken("14/10/2017", [240, 260, 480, 290]),
        OCRToken("TIME: 14:32", [600, 260, 850, 290]),

        # Line items
        OCRToken("1 x APPLE JUICE", [100, 400, 500, 430]),
        OCRToken("8.50", [750, 400, 850, 430]),
        OCRToken("1 x FRESH SALAD", [100, 450, 500, 480]),
        OCRToken("10.04", [750, 450, 850, 480]),

        # Summary / Totals
        OCRToken("SUBTOTAL", [100, 750, 350, 780]),
        OCRToken("18.54", [750, 750, 850, 780]),
        OCRToken("GST 0%", [100, 800, 300, 830]),
        OCRToken("0.00", [750, 800, 850, 830]),
        OCRToken("TOTAL PAYABLE", [100, 860, 450, 900]),
        OCRToken("RM 18.54", [750, 860, 920, 900]),
        OCRToken("CASH", [100, 920, 250, 950]),
        OCRToken("50.00", [750, 920, 850, 950]),
        OCRToken("CHANGE", [100, 960, 280, 990]),
        OCRToken("31.46", [750, 960, 850, 990]),
    ]

    fields = AdvancedRegexExtractor.extract_fields(tokens)

    assert "MAKASSAR FRESH MARKET" in fields["company"]
    assert "SDN. BHD." in fields["company"]
    assert fields["date"] == "14/10/2017"
    assert fields["total"] == "18.54"


if __name__ == "__main__":
    test_preprocessing_synthetic_image()
    test_advanced_regex_extraction()
    print("All tests passed successfully!")

