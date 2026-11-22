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


def _lines(rows):
    """[(text, y0)] -> one full-width OCRToken per row (like PaddleOCR text lines)."""
    return [OCRToken(text, [50, y, 900, y + 25]) for text, y in rows]


def test_company_registration_number_removed():
    assert AdvancedRegexExtractor.clean_company_name("AEON CO. (M) BHD (126926-H)") == "AEON CO. (M) BHD"
    assert AdvancedRegexExtractor.clean_company_name("99 SPEED MART S/B (519537-X)") == "99 SPEED MART S/B"
    assert AdvancedRegexExtractor.clean_company_name("KING'S CONFECTIONERY S/B 273500-U (KSB)") == "KING'S CONFECTIONERY S/B"
    assert AdvancedRegexExtractor.clean_company_name("RESTORAN WAN SHENG") == "RESTORAN WAN SHENG"


def test_company_prefers_merchant_over_customer_below_date():
    tokens = _lines([
        ("GARDENIA BAKERIES (KL) SDN BHD (139386 X)", 100),
        ("LOT 3, JALAN PELABUR 23/1,", 125),
        ("TAX INVOICE", 215),
        ("DATE: 14/10/2017", 300),
        ("MAKASSAR FRESH MARKET SDN BHD", 325),
    ])
    assert AdvancedRegexExtractor.extract_fields(tokens)["company"] == "GARDENIA BAKERIES (KL) SDN BHD"


def test_company_name_over_two_lines():
    tokens = _lines([
        ("POPULAR BOOK", 55),
        ("CO. (M) SDN BHD", 90),
        ("(CO. NO. 113825-W)", 127),
        ("NO 8, JALAN 7/118B, DESA TUN RAZAK", 172),
    ])
    assert AdvancedRegexExtractor.extract_fields(tokens)["company"] == "POPULAR BOOK CO. (M) SDN BHD"


def test_total_ignores_gst_summary_and_excluding_gst():
    tokens = _lines([
        ("SHOP SDN BHD", 50),
        ("TOTAL QTY: 6 82.68", 586),
        ("TOTAL SALES (EXCLUDING GST) : 78.00", 622),
        ("TOTAL GST : 4.68", 674),
        ("ROUNDING : 0.00", 707),
        ("TOTAL SALES (INCLUSIVE OF GST) : 82.68", 737),
        ("CASH : 82.68", 769),
        ("TAX CODE % AMT (RM) TAX (RM)", 877),
        ("SR 6 78.00 4.68", 900),
        ("TOTAL : 78.00 4.68", 928),
    ])
    assert AdvancedRegexExtractor.extract_fields(tokens)["total"] == "82.68"


def test_total_uses_rounded_amount():
    tokens = _lines([
        ("SHOP SDN BHD", 50),
        ("TOTAL SALES (INCLUSIVE GST) RM 22.91", 637),
        ("ROUNDING ADJUSTMENT RM -.01", 667),
        ("ROUNDING RM 22.90", 698),
        ("CASH RM 50.00", 728),
        ("CHANGE RM 27.10", 759),
    ])
    assert AdvancedRegexExtractor.extract_fields(tokens)["total"] == "22.90"


def test_total_skips_savings_and_negative_amounts():
    tokens = _lines([
        ("SHOP SDN BHD", 50),
        ("STAT-BUY 2@RM 13 -2.98", 554),
        ("TOTAL RM INCL. OF GST 30.10", 602),
        ("CASH -40.10", 626),
        ("CHANGE 10.00", 649),
        ("TOTAL SAVINGS -2.98", 816),
    ])
    assert AdvancedRegexExtractor.extract_fields(tokens)["total"] == "30.10"


def test_date_with_month_name_and_slashes():
    tokens = _lines([("SHOP SDN BHD", 50), ("DATE: 02/JAN/2017 10:15", 200)])
    assert AdvancedRegexExtractor.extract_fields(tokens)["date"] == "02/JAN/2017"


def test_paddleocr_v2_and_v3_results_give_same_tokens():
    from src.paddle_ocr_engine import PaddleOCREngine

    polys = [
        [[10, 10], [200, 10], [200, 40], [10, 40]],
        [[10, 300], [80, 300], [80, 330], [10, 330]],
        [[150, 302], [200, 302], [200, 330], [150, 330]],
    ]
    texts = ["ACME STORE SDN BHD", "TOTAL", "12.50"]
    v2 = [[[p, (t, 0.9)] for p, t in zip(polys, texts)]]
    v3 = [{"rec_texts": texts, "rec_scores": [0.9] * 3, "rec_polys": [np.array(p) for p in polys]}]

    tok2 = PaddleOCREngine.to_tokens(PaddleOCREngine.parse_results(v2), 250, 400)
    tok3 = PaddleOCREngine.to_tokens(PaddleOCREngine.parse_results(v3), 250, 400)
    assert [(t.text, t.box) for t in tok2] == [(t.text, t.box) for t in tok3]
    # "TOTAL" and "12.50" are separate boxes on the same row and must be paired
    assert AdvancedRegexExtractor.extract_fields(tok3)["total"] == "12.50"


if __name__ == "__main__":
    test_preprocessing_synthetic_image()
    test_advanced_regex_extraction()
    print("All tests passed successfully!")



# ---------------------------------------------------------------------------
# Real-data evaluation (src/real_eval.py) - uses a stand-in OCR, nothing is downloaded
# ---------------------------------------------------------------------------

def _fake_tokens(company, date, total):
    """Tokens in the format ReceiptExtractionPipeline.process() returns (boxes in [0, 1000])."""
    rows = [(company, 50), (f"DATE: {date}", 300), ("SUBTOTAL 1.00", 700), (f"TOTAL RM {total}", 800)]
    return [{"text": t, "box": [50, y, 900, y + 25], "confidence": 0.9} for t, y in rows]


class _FakePipeline:
    """Stands in for ReceiptExtractionPipeline: 'reads' receipts from a dict instead of running OCR."""

    def __init__(self, receipts):
        self.receipts = receipts
        self.calls = []

    def process(self, image_path):
        import os
        name = os.path.basename(image_path)
        self.calls.append(name)
        if name not in self.receipts:
            raise ValueError("cannot read image")
        return {"tokens": self.receipts[name], "preprocessing": {"scale_factor": 1.0, "skew_angle": 0.0}}


def test_load_ground_truth_reads_metadata_jsonl(tmp_path):
    import json
    from src.real_eval import load_ground_truth

    meta = tmp_path / "metadata.jsonl"
    row = {"file_name": "a.jpg",
           "ground_truth": json.dumps({"gt_parse": {"company": "ACME SDN BHD", "date": "01/02/2019",
                                                    "address": "x", "total": "12.50"}})}
    meta.write_text(json.dumps(row) + "\n\n", encoding="utf-8")
    assert load_ground_truth(str(meta)) == {"a.jpg": {"company": "ACME SDN BHD", "date": "01/02/2019", "total": "12.50"}}


def test_run_ocr_caches_resumes_and_survives_bad_image(tmp_path):
    import json
    from src.real_eval import run_ocr, extract_from_cache

    pipe = _FakePipeline({"a.jpg": _fake_tokens("ACME SDN BHD", "01/02/2019", "12.50")})
    cache = str(tmp_path / "cache")
    stats = run_ocr(["a.jpg", "bad.jpg"], str(tmp_path), cache, pipe, show_progress=False)
    assert stats == {"done": 2, "cached": 0, "errors": 1}
    assert "error" in json.load(open(tmp_path / "cache" / "bad.jpg.json"))

    # second run: everything comes from the cache, OCR is not run again
    stats = run_ocr(["a.jpg", "bad.jpg"], str(tmp_path), cache, pipe, show_progress=False)
    assert stats["done"] == 0 and stats["cached"] == 2
    assert pipe.calls == ["a.jpg", "bad.jpg"]

    preds = extract_from_cache(["a.jpg", "bad.jpg"], cache)
    assert preds["a.jpg"] == {"company": "ACME SDN BHD", "date": "01/02/2019", "total": "12.50"}
    assert preds["bad.jpg"] == {"company": "", "date": "", "total": ""}


def test_evaluate_real_data_scores_extracted_fields_and_writes_results(tmp_path, monkeypatch):
    import json
    import src.real_eval as real_eval

    truth = {
        "a.jpg": {"company": "ACME SDN BHD", "date": "01/02/2019", "total": "12.50"},
        "b.jpg": {"company": "BETA TRADING", "date": "05/06/2019", "total": "99.00"},
    }
    monkeypatch.setattr(real_eval, "download_sroie", lambda split, data_dir, limit: truth)
    pipe = _FakePipeline({
        "a.jpg": _fake_tokens("ACME SDN BHD", "01/02/2019", "12.50"),      # all correct
        "b.jpg": _fake_tokens("BETA TRADING", "05/06/2019", "98.00"),      # wrong total
    })
    out = tmp_path / "res" / "regex_baseline.json"
    results = real_eval.evaluate_real_data(limit=2, data_dir=str(tmp_path), cache_root=str(tmp_path / "c"),
                                           output_path=str(out), pipeline=pipe)

    assert results["test_size"] == 2 and results["ocr_errors"] == 0
    assert results["metrics"]["company"]["f1"] == 1.0
    assert results["metrics"]["date"]["f1"] == 1.0
    assert results["metrics"]["total"]["f1"] == 0.5
    saved = json.load(open(out))
    assert saved["model"] == "PaddleOCR + Advanced Regex" and "macro_avg" in saved["metrics"]


def test_fast_engine_uses_mobile_models_and_skips_line_orientation():
    from src.paddle_ocr_engine import PaddleOCREngine

    seen = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    PaddleOCREngine(fast=True)._build_v3(FakePaddleOCR)
    assert seen["text_detection_model_name"] == "PP-OCRv5_mobile_det"
    assert seen["text_recognition_model_name"] == "en_PP-OCRv5_mobile_rec"
    assert seen["use_textline_orientation"] is False

    seen.clear()
    PaddleOCREngine(fast=False)._build_v3(FakePaddleOCR)
    assert "text_detection_model_name" not in seen


def _write_cache(cache_dir, name, tokens):
    import json, os
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"tokens": tokens}, f)


def test_diagnose_misses_separates_ocr_misread_from_extractor_mistake(tmp_path, capsys):
    from src.real_eval import diagnose_misses

    cache = str(tmp_path)
    # a.jpg: OCR read the true company name, but the extractor chose another line
    _write_cache(cache, "a.jpg", [{"text": "WRONG LINE SDN BHD", "box": [0, 0, 1, 1]},
                                  {"text": "ACME TRADING", "box": [0, 50, 1, 51]}])
    # b.jpg: OCR misread the name ("ACNE" for "ACME"), nothing could have got it right
    _write_cache(cache, "b.jpg", [{"text": "ACNE TRADING", "box": [0, 0, 1, 1]}])
    truth = {"a.jpg": {"company": "ACME TRADING"}, "b.jpg": {"company": "ACME TRADING"}}
    preds = {"a.jpg": {"company": "WRONG LINE SDN BHD"}, "b.jpg": {"company": "ACNE TRADING"}}

    counts = diagnose_misses("company", ["a.jpg", "b.jpg"], truth, preds, cache)
    assert counts == {"wrong": 2, "ocr_problem": 1, "extractor_problem": 1, "no_ocr": 0}
    assert "closest OCR line" in capsys.readouterr().out


def test_diagnose_total_and_date_found_in_ocr_text(tmp_path):
    from src.real_eval import _text_was_read

    assert _text_was_read("total", "1,234.50", ["TOTAL RM 1,234.50"])
    assert not _text_was_read("total", "20.50", ["TOTAL RM 20.60"])
    assert _text_was_read("date", "15/01/2019", ["DATE 15-01-2019 10:20"])
    assert not _text_was_read("date", "15/01/2019", ["DATE 15-01-2O19"])


def test_hybrid_takes_best_method_per_field_and_falls_back():
    from src.real_eval import combine_predictions

    regex = {"company": "REGEX CO", "date": "", "total": "10.00"}
    layout = {"company": "LAYOUT CO", "date": "01/01/2019", "total": "11.00", "address": "x"}
    assert combine_predictions(regex, layout) == {"company": "LAYOUT CO", "date": "01/01/2019", "total": "10.00"}
    # LayoutLM found nothing -> fall back to the rules
    assert combine_predictions(regex, {})["company"] == "REGEX CO"


def test_compare_methods_scores_three_methods_on_cached_ocr(tmp_path):
    import json
    from src.real_eval import compare_methods

    data_dir, cache_root = tmp_path / "d", tmp_path / "c"
    (data_dir / "test").mkdir(parents=True)
    gt = {"company": "ACME SDN BHD", "date": "01/02/2019", "total": "12.50"}
    row = {"file_name": "a.jpg", "ground_truth": json.dumps({"gt_parse": gt})}
    (data_dir / "test" / "metadata.jsonl").write_text(json.dumps(row), encoding="utf-8")
    _write_cache(str(cache_root / "mobile"), "a.jpg", _fake_tokens("ACME SDN BHD", "01/02/2019", "12.50"))

    seen = {}

    def fake_layoutlm(words, boxes):
        seen["words"], seen["boxes"] = words, boxes
        return {"company": "ACME SDN BHD", "date": "01/02/2018", "total": "99.00", "address": ""}

    res = compare_methods(limit=1, data_dir=str(data_dir), cache_root=str(cache_root),
                          predict_fn=fake_layoutlm, output_path=str(tmp_path / "cmp.json"))
    m = res["metrics"]
    assert m["Advanced Regex"]["macro_avg"]["f1"] == 1.0
    assert m["LayoutLM"]["company"]["f1"] == 1.0 and m["LayoutLM"]["total"]["f1"] == 0.0
    # hybrid: company+date from LayoutLM (date wrong there), total from the rules (right)
    assert m["Hybrid (best per field)"]["total"]["f1"] == 1.0
    assert m["Hybrid (best per field)"]["date"]["f1"] == 0.0
    # every word is given its OCR line's box
    assert len(seen["words"]) == len(seen["boxes"]) and seen["boxes"][0] == [50, 50, 900, 75]


def test_compare_methods_asks_to_run_ocr_first(tmp_path):
    import json
    import pytest
    from src.real_eval import compare_methods

    (tmp_path / "test").mkdir()
    row = {"file_name": "a.jpg", "ground_truth": json.dumps({"gt_parse": {"company": "X", "date": "", "total": ""}})}
    (tmp_path / "test" / "metadata.jsonl").write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Step 4"):
        compare_methods(limit=1, data_dir=str(tmp_path), cache_root=str(tmp_path / "none"),
                        predict_fn=lambda w, b: {}, output_path=str(tmp_path / "o.json"))
