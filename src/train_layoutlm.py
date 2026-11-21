"""
Training and Evaluation Pipeline for LayoutLM Token Classification (Multimodal: Text + 2D Layout).

This script fine-tunes microsoft/layoutlm-base-uncased on SROIE receipts using
WordPiece text tokens and 2D spatial bounding boxes normalized to [0, 1000].
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
try:
    from torch.optim import AdamW
except ImportError:
    from transformers import AdamW

from transformers import (
    LayoutLMTokenizer,
    LayoutLMForTokenClassification,
    get_linear_schedule_with_warmup,
)

from src.dataset import LABELS, ID2LABEL, LABEL2ID
from src.metrics import compute_field_metrics, compute_delta_f1


class SROIELayoutDataset(Dataset):
    """PyTorch Dataset for LayoutLM (Text + 2D Bounding Boxes)."""

    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_samples = [self._encode_sample(item) for item in data]

    def _encode_sample(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        words = item["words"]
        bboxes = item["bboxes"]
        bio_tags = item["bio_tags"]

        input_ids = [self.tokenizer.cls_token_id]
        boxes = [[0, 0, 0, 0]]
        labels = [-100]

        for word, box, tag in zip(words, bboxes, bio_tags):
            subwords = self.tokenizer.tokenize(word)
            if not subwords:
                continue
            subword_ids = self.tokenizer.convert_tokens_to_ids(subwords)
            label_id = LABEL2ID.get(tag, 0)

            # Clamp coordinates to [0, 1000]
            clamped_box = [min(1000, max(0, c)) for c in box]

            # First subword gets the label
            input_ids.append(subword_ids[0])
            boxes.append(clamped_box)
            labels.append(label_id)

            # Continuation subwords get parent bbox and ignore index -100
            for sw_id in subword_ids[1:]:
                input_ids.append(sw_id)
                boxes.append(clamped_box)
                labels.append(-100)

            if len(input_ids) >= self.max_length - 1:
                break

        input_ids.append(self.tokenizer.sep_token_id)
        boxes.append([1000, 1000, 1000, 1000])
        labels.append(-100)

        # Truncate
        input_ids = input_ids[: self.max_length]
        boxes = boxes[: self.max_length]
        labels = labels[: self.max_length]

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        padding_length = self.max_length - len(input_ids)

        input_ids += [self.tokenizer.pad_token_id] * padding_length
        boxes += [[0, 0, 0, 0]] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        labels += [-100] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "bbox": torch.tensor(boxes, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "doc_id": item["id"],
        }

    def __len__(self):
        return len(self.encoded_samples)

    def __getitem__(self, idx):
        return self.encoded_samples[idx]


def extract_entities_from_layout_tokens(
    tokens: List[str],
    predicted_label_ids: List[int],
) -> Dict[str, str]:
    """Reconstruct entity strings from predicted LayoutLM BIO tags."""
    extracted = {"company": "", "date": "", "total": "", "address": ""}
    current_entity = None
    current_tokens = []

    for token, label_id in zip(tokens, predicted_label_ids):
        if label_id == -100:
            continue
        tag = ID2LABEL.get(label_id, "O")

        if tag.startswith("B-"):
            if current_entity and current_tokens:
                field = current_entity.lower()
                if field in extracted and not extracted[field]:
                    extracted[field] = " ".join(current_tokens)
            current_entity = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_entity and tag[2:] == current_entity:
            current_tokens.append(token)
        else:
            if current_entity and current_tokens:
                field = current_entity.lower()
                if field in extracted and not extracted[field]:
                    extracted[field] = " ".join(current_tokens)
            current_entity = None
            current_tokens = []

    if current_entity and current_tokens:
        field = current_entity.lower()
        if field in extracted and not extracted[field]:
            extracted[field] = " ".join(current_tokens)

    for k in extracted:
        extracted[k] = extracted[k].replace(" ##", "").replace("##", "").strip()

    return extracted


def train_layoutlm(
    train_file: str = "data/processed/train.json",
    test_file: str = "data/processed/test.json",
    output_path: str = "results/layoutlm_results.json",
    bert_results_path: str = "results/bert_results.json",
    model_name: str = "microsoft/layoutlm-base-uncased",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 5e-5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train and evaluate the multimodal LayoutLM model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing LayoutLM Training on device: {device}")

    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Loaded {len(train_data)} train samples and {len(test_data)} test samples.")

    tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
    model = LayoutLMForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    train_dataset = SROIELayoutDataset(train_data, tokenizer)
    test_dataset = SROIELayoutDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    print(f"Starting multimodal LayoutLM fine-tuning for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            bbox = batch["bbox"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

    # Evaluation
    print("Evaluating LayoutLM on test set...")
    model.eval()
    predictions = {}
    ground_truths = {item["id"]: item["entities"] for item in test_data}

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            bbox = batch["bbox"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            doc_ids = batch["doc_id"]

            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = torch.argmax(outputs.logits, dim=2).cpu().tolist()

            for i, doc_id in enumerate(doc_ids):
                token_list = tokenizer.convert_ids_to_tokens(input_ids[i])
                pred_tags = preds[i]
                extracted = extract_entities_from_layout_tokens(token_list, pred_tags)
                predictions[doc_id] = extracted

    metrics = compute_field_metrics(predictions, ground_truths)

    # Compute ΔF1 if BERT results exist
    deltas = {}
    if os.path.exists(bert_results_path):
        try:
            with open(bert_results_path, "r", encoding="utf-8") as f:
                bert_res = json.load(f)
            deltas = compute_delta_f1(bert_res.get("metrics", {}), metrics)
        except Exception as e:
            print(f"Note: Could not calculate delta vs BERT: {e}")

    output_data = {
        "model": "LayoutLM-base (Multimodal)",
        "modality": "Text + 2D Bounding Boxes",
        "epochs": epochs,
        "test_size": len(test_data),
        "metrics": metrics,
        "delta_vs_bert": deltas,
        "sample_predictions": {k: predictions[k] for k in list(predictions.keys())[:5]},
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 50)
    print("LayoutLM-base Evaluation Results:")
    print("=" * 50)
    for field in ["company", "date", "total", "macro_avg"]:
        m = metrics.get(field, {})
        d = deltas.get(field, 0.0)
        d_str = f" (ΔF1: {d:+.4f})" if deltas else ""
        print(f"  {field.upper():<12} | Precision: {m.get('precision', 0):.4f} | Recall: {m.get('recall', 0):.4f} | F1: {m.get('f1', 0):.4f}{d_str}")
    print("=" * 50)
    print(f"Results saved to {output_path}")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    train_layoutlm(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
