"""
inference.py

Single-receipt inference + demo helpers for the fine-tuned LayoutLM model.

This is additive glue on top of train_layoutlm.py (it reuses that module's
LABELS / ID2LABEL / extract_entity_spans so decoding stays identical to
training). It does NOT run training and does not import the training-only
helpers (tqdm / dataset building), so it loads even while those are in flux.

Pipeline:
    receipt image -> OCR (Tesseract) -> words + [0,1000] boxes
                  -> LayoutLM -> per-word BIO tags
                  -> entity spans -> {company, date, total, address}
"""

from typing import Dict, List, Tuple

from src.train_layoutlm import ID2LABEL, extract_entity_spans

TARGET_FIELDS = ["company", "date", "total", "address"]


def spans_to_fields(words: List[str], word_tags: List[str]) -> Dict[str, str]:
    """Group per-word BIO tags into {field: value}, taking the first span per field."""
    fields = {f: "" for f in TARGET_FIELDS}
    # Sort by start index so "first span on the page" wins per field.
    for field_type, start, end in sorted(extract_entity_spans(word_tags), key=lambda s: s[1]):
        key = field_type.lower()
        if key in fields and not fields[key]:
            fields[key] = " ".join(str(w) for w in words[start:end])
    return fields


def _encode(words: List[str], boxes: List[List[int]], tokenizer, max_length: int = 512):
    """Tokenize words, mirroring ReceiptDataset: track each word's first-subword position."""
    input_ids = [tokenizer.cls_token_id]
    bbox = [[0, 0, 0, 0]]
    first_subword_positions: List[Tuple[int, int]] = []

    for word_index, (word, box) in enumerate(zip(words, boxes)):
        tokens = tokenizer.tokenize(str(word))
        if not tokens:
            continue
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        available = max_length - len(input_ids) - 1  # reserve [SEP]
        if available <= 0 or len(token_ids) > available:
            break
        clamped = [min(1000, max(0, int(c))) for c in box]
        first_subword_positions.append((len(input_ids), word_index))
        for token_id in token_ids:
            input_ids.append(token_id)
            bbox.append(clamped)

    input_ids.append(tokenizer.sep_token_id)
    bbox.append([0, 0, 0, 0])
    return input_ids, bbox, first_subword_positions


def predict_fields(model, tokenizer, words: List[str], boxes: List[List[int]], device=None) -> Dict[str, str]:
    """Run the trained LayoutLM on OCR output and return the extracted fields."""
    import torch

    if device is None:
        device = next(model.parameters()).device

    input_ids, bbox, first_subword_positions = _encode(words, boxes, tokenizer)
    ii = torch.tensor([input_ids], dtype=torch.long, device=device)
    bb = torch.tensor([bbox], dtype=torch.long, device=device)
    am = torch.ones_like(ii)
    tt = torch.zeros_like(ii)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=ii, bbox=bb, attention_mask=am, token_type_ids=tt).logits
    preds = logits.argmax(dim=-1)[0].cpu().tolist()

    # Read each word's predicted tag from its first subword position (same as evaluate()).
    kept_words = [words[word_index] for _, word_index in first_subword_positions]
    word_tags = [ID2LABEL[preds[pos]] for pos, _ in first_subword_positions]
    return spans_to_fields(kept_words, word_tags)


def ocr_image(path: str) -> Tuple[List[str], List[List[int]]]:
    """Scanned image -> (words, boxes normalized to [0, 1000]). Requires pytesseract + tesseract-ocr."""
    import pytesseract
    from PIL import Image

    img = Image.open(path).convert("RGB")
    W, H = img.size
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(text)
        boxes.append([
            int(1000 * x / W), int(1000 * y / H),
            int(1000 * (x + w) / W), int(1000 * (y + h) / H),
        ])
    return words, boxes


def extract_from_image(model, tokenizer, path: str, device=None) -> Dict[str, str]:
    """Full image -> fields pipeline."""
    words, boxes = ocr_image(path)
    return predict_fields(model, tokenizer, words, boxes, device=device)


if __name__ == "__main__":
    # Self-check: span decoding is correct without needing a model or GPU.
    words = ["ACME", "STORE", "receipt", "12/05/2019", "TOTAL", "48.50"]
    tags = ["B-COMPANY", "I-COMPANY", "O", "B-DATE", "O", "B-TOTAL"]
    fields = spans_to_fields(words, tags)
    assert fields["company"] == "ACME STORE", fields
    assert fields["date"] == "12/05/2019", fields
    assert fields["total"] == "48.50", fields
    print("inference self-check passed:", fields)
