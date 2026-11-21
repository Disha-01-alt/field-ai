"""
Training and Evaluation Pipeline for BERT-base Token Classification (Text-Only Baseline).

This script fine-tunes bert-base-uncased on SROIE receipts using sequential
WordPiece tokens without any 2D bounding-box spatial coordinates.
"""

import os
import re
import sys
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
try:
    from torch.optim import AdamW
except ImportError:
    from transformers import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)

from src.dataset import LABELS, ID2LABEL, LABEL2ID
from src.metrics import compute_field_metrics, compute_delta_f1


class SROIETokenDataset(Dataset):
    """PyTorch Dataset for text-only BERT token classification."""

    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_samples = [self._encode_sample(item) for item in data]

    def _encode_sample(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        words = item["words"]
        bio_tags = item["bio_tags"]

        input_ids = [self.tokenizer.cls_token_id]
        labels = [-100]

        for word, tag in zip(words, bio_tags):
            subwords = self.tokenizer.tokenize(word)
            if not subwords:
                continue
            subword_ids = self.tokenizer.convert_tokens_to_ids(subwords)
            label_id = LABEL2ID.get(tag, 0)

            # Assign I-tag to continuation subwords so the full entity span is learned
            if tag.startswith("B-"):
                cont_tag = "I-" + tag[2:]
                cont_id = LABEL2ID.get(cont_tag, label_id)
            elif tag.startswith("I-"):
                cont_id = label_id
            else:
                cont_id = 0

            # First subword gets the actual label
            input_ids.append(subword_ids[0])
            labels.append(label_id)

            # Continuation subwords get the continuation tag
            for sw_id in subword_ids[1:]:
                input_ids.append(sw_id)
                labels.append(cont_id)

            if len(input_ids) >= self.max_length - 1:
                break

        input_ids.append(self.tokenizer.sep_token_id)
        labels.append(-100)

        # Truncate if needed
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)

        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        labels += [-100] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "doc_id": item["id"],
        }

    def __len__(self):
        return len(self.encoded_samples)

    def __getitem__(self, idx):
        return self.encoded_samples[idx]


def reassemble_token_sequence(tokens: List[str], tokenizer=None) -> str:
    """Reassemble WordPiece tokens into a natural string with proper punctuation."""
    if tokenizer and hasattr(tokenizer, "convert_tokens_to_string"):
        text = tokenizer.convert_tokens_to_string(tokens).strip()
    else:
        out = []
        for t in tokens:
            if t.startswith("##"):
                if out:
                    out[-1] += t[2:]
                else:
                    out.append(t[2:])
            else:
                out.append(t)
        text = " ".join(out)
    text = re.sub(r"\s*([/\-\.:,])\s*", r"\1", text)
    return text.strip()


def extract_entities_from_tokens(
    tokens: List[str],
    predicted_label_ids: List[int],
    tokenizer=None,
) -> Dict[str, str]:
    """Reconstruct entity strings from predicted BIO tags."""
    extracted = {"company": "", "date": "", "total": "", "address": ""}
    current_entity = None
    current_tokens = []

    for token, label_id in zip(tokens, predicted_label_ids):
        if label_id == -100 or token in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        tag = ID2LABEL.get(label_id, "O")

        if tag.startswith("B-"):
            if current_entity and current_tokens:
                field = current_entity.lower()
                if field in extracted and not extracted[field]:
                    extracted[field] = reassemble_token_sequence(current_tokens, tokenizer)
            current_entity = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_entity and tag[2:] == current_entity:
            current_tokens.append(token)
        else:
            if current_entity and current_tokens:
                field = current_entity.lower()
                if field in extracted and not extracted[field]:
                    extracted[field] = reassemble_token_sequence(current_tokens, tokenizer)
            current_entity = None
            current_tokens = []

    if current_entity and current_tokens:
        field = current_entity.lower()
        if field in extracted and not extracted[field]:
            extracted[field] = reassemble_token_sequence(current_tokens, tokenizer)

    return extracted


def train_bert(
    train_file: str = "data/processed/train.json",
    test_file: str = "data/processed/test.json",
    output_path: str = "results/bert_results.json",
    model_name: str = "bert-base-uncased",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train and evaluate the text-only BERT token classification model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing BERT Training on device: {device}")

    # Load preprocessed datasets
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Loaded {len(train_data)} train samples and {len(test_data)} test samples.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    train_dataset = SROIETokenDataset(train_data, tokenizer)
    test_dataset = SROIETokenDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    print(f"Starting training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

    # Evaluation using batch test loader
    print("Evaluating on test set...")
    model.eval()
    predictions = {}
    ground_truths = {item["id"]: item["entities"] for item in test_data}

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_ids = batch["doc_id"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=2).cpu().tolist()

            for i, doc_id in enumerate(doc_ids):
                token_list = tokenizer.convert_ids_to_tokens(input_ids[i])
                pred_tags = preds[i]
                extracted = extract_entities_from_tokens(token_list, pred_tags, tokenizer)
                predictions[doc_id] = extracted

    metrics = compute_field_metrics(predictions, ground_truths)

    output_data = {
        "model": "BERT-base (Text Only)",
        "modality": "1D Text Tokens",
        "epochs": epochs,
        "test_size": len(test_data),
        "metrics": metrics,
        "sample_predictions": {k: predictions[k] for k in list(predictions.keys())[:5]},
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 50)
    print("BERT-base Evaluation Results:")
    print("=" * 50)
    for field in ["company", "date", "total", "macro_avg"]:
        m = metrics.get(field, {})
        print(f"  {field.upper():<12} | Precision: {m.get('precision', 0):.4f} | Recall: {m.get('recall', 0):.4f} | F1: {m.get('f1', 0):.4f}")
    print("=" * 50)
    print(f"Results saved to {output_path}")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    train_bert(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
