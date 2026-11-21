
"""
train_layoutlm.py

LayoutLM token-classification training for SROIE receipt extraction.

Compatible with dataset.py output:
{
    "id": "...",
    "words": [...],
    "bboxes": [[x0, y0, x1, y1], ...],
    "ner_tags": [...],
    "bio_tags": [...],
    "entities": {...}
}

Important:
- Uses the existing train.json and test.json files.
- Creates validation data ONLY from train.json.
- Does not regenerate or modify the test split.
- Selects the best checkpoint using validation macro F1.
- Reports entity-level F1 for Company, Date, Total, and Address.
"""

import os
import json
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    LayoutLMTokenizerFast,
    LayoutLMForTokenClassification,
    get_linear_schedule_with_warmup,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/layoutlm-base-uncased"

TRAIN_FILE = "data/processed/train.json"
TEST_FILE = "data/processed/test.json"

OUTPUT_DIR = "outputs/layoutlm"

MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 12

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

VALIDATION_RATIO = 0.15
EARLY_STOPPING_PATIENCE = 3

SEED = 42

# Macro F1 is calculated over these three research fields.
# Address is still trained and reported separately.
MAIN_FIELDS = ["COMPANY", "DATE", "TOTAL"]

ALL_FIELDS = [
    "COMPANY",
    "DATE",
    "TOTAL",
    "ADDRESS",
]


# Must match dataset.py exactly.
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

LABEL2ID = {
    label: index
    for index, label in enumerate(LABELS)
}

ID2LABEL = {
    index: label
    for index, label in enumerate(LABELS)
}

NUM_LABELS = len(LABELS)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. LOAD JSON DATA
# ============================================================

def load_json(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"{path} must contain a JSON list."
        )

    print(f"Loaded {len(data)} documents from {path}")

    return data


def validate_sample(sample):

    required = [
        "id",
        "words",
        "bboxes",
        "ner_tags",
    ]

    for key in required:
        if key not in sample:
            raise ValueError(
                f"Document {sample.get('id', 'UNKNOWN')} "
                f"is missing '{key}'."
            )

    words = sample["words"]
    boxes = sample["bboxes"]
    tags = sample["ner_tags"]

    if not (
        len(words) == len(boxes) == len(tags)
    ):
        raise ValueError(
            f"Length mismatch in document {sample['id']}: "
            f"words={len(words)}, "
            f"bboxes={len(boxes)}, "
            f"ner_tags={len(tags)}"
        )

    for box in boxes:

        if len(box) != 4:
            raise ValueError(
                f"Invalid bounding box in {sample['id']}: {box}"
            )

        if any(
            coordinate < 0 or coordinate > 1000
            for coordinate in box
        ):
            raise ValueError(
                f"Bounding box outside [0, 1000] "
                f"in {sample['id']}: {box}"
            )

    for tag in tags:

        if tag < 0 or tag >= NUM_LABELS:
            raise ValueError(
                f"Invalid NER tag ID {tag} "
                f"in document {sample['id']}"
            )


# ============================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================

def split_train_validation(
    samples,
    validation_ratio=0.15,
    seed=42,
):

    if len(samples) < 2:
        raise ValueError(
            "At least two training documents are required."
        )

    indices = list(range(len(samples)))

    rng = random.Random(seed)
    rng.shuffle(indices)

    validation_size = max(
        1,
        int(len(samples) * validation_ratio)
    )

    validation_size = min(
        validation_size,
        len(samples) - 1
    )

    validation_indices = set(
        indices[:validation_size]
    )

    train_samples = [
        sample
        for index, sample in enumerate(samples)
        if index not in validation_indices
    ]

    validation_samples = [
        sample
        for index, sample in enumerate(samples)
        if index in validation_indices
    ]

    print("\nDataset split:")
    print(f"Training documents:   {len(train_samples)}")
    print(f"Validation documents: {len(validation_samples)}")

    return train_samples, validation_samples


# ============================================================
# 5. TOKENIZATION AND SUBWORD ALIGNMENT
# ============================================================


class ReceiptDataset(Dataset):

    def __init__(self, samples, tokenizer, max_length=512):

        self.features = []

        for sample in tqdm(samples, desc="Tokenizing receipts"):

            validate_sample(sample)

            words = sample["words"]
            boxes = get_boxes(sample)
            word_labels = [int(x) for x in sample["ner_tags"]]

            input_ids = [tokenizer.cls_token_id]
            attention_mask = [1]
            aligned_boxes = [[0, 0, 0, 0]]
            aligned_labels = [-100]

            first_subword_positions = []
            retained_word_labels = []

            # Tokenize each original OCR word independently.
            for word_index, (word, box, label) in enumerate(
                zip(words, boxes, word_labels)
            ):

                word_tokens = tokenizer.tokenize(str(word))

                if not word_tokens:
                    continue

                word_token_ids = tokenizer.convert_tokens_to_ids(
                    word_tokens
                )

                # Reserve one position for [SEP].
                available = max_length - len(input_ids) - 1

                if available <= 0:
                    break

                # Do not partially include a word.
                if len(word_token_ids) > available:
                    break

                box = [int(x) for x in box]

                # Index into the compact list of retained words.
                compact_word_index = len(retained_word_labels)

                # Remember the first subword's token position.
                first_subword_positions.append(
                    (len(input_ids), compact_word_index)
                )

                retained_word_labels.append(label)

                for subword_index, token_id in enumerate(word_token_ids):

                    input_ids.append(token_id)
                    attention_mask.append(1)
                    aligned_boxes.append(box)

                    if subword_index == 0:
                        aligned_labels.append(label)
                    else:
                        aligned_labels.append(-100)

            # Add [SEP].
            input_ids.append(tokenizer.sep_token_id)
            attention_mask.append(1)
            aligned_boxes.append([0, 0, 0, 0])
            aligned_labels.append(-100)

            if not first_subword_positions:
                raise ValueError(
                    f"No words tokenized in document {sample['id']}"
                )

            assert len(input_ids) == len(aligned_boxes)
            assert len(input_ids) == len(aligned_labels)

            # Verify that entity labels survived tokenization.
            if not any(label != 0 for label in retained_word_labels):
                raise ValueError(
                    f"All labels became O in document {sample['id']}"
                )

            self.features.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "bbox": aligned_boxes,
                "labels": aligned_labels,
                "first_subword_positions": first_subword_positions,
                "original_word_labels": retained_word_labels,
                "document_id": sample["id"],
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


# ============================================================
# 6. BATCH COLLATOR
# ============================================================

class LayoutLMCollator:

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):

        max_length = max(
            len(item["input_ids"])
            for item in batch
        )

        input_ids = []
        attention_masks = []
        token_type_ids = []
        bounding_boxes = []
        labels = []

        for item in batch:

            padding_length = (
                max_length - len(item["input_ids"])
            )

            input_ids.append(
                item["input_ids"]
                + [self.pad_token_id] * padding_length
            )

            attention_masks.append(
                item["attention_mask"]
                + [0] * padding_length
            )

            token_type_ids.append(
                [0] * len(item["input_ids"])
                + [0] * padding_length
            )

            bounding_boxes.append(
                item["bbox"]
                + [[0, 0, 0, 0]] * padding_length
            )

            labels.append(
                item["labels"]
                + [-100] * padding_length
            )

        return {

            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long
            ),

            "token_type_ids": torch.tensor(
                token_type_ids,
                dtype=torch.long
            ),

            "bbox": torch.tensor(
                bounding_boxes,
                dtype=torch.long
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.long
            ),

            "metadata": batch,
        }


# ============================================================
# 7. ENTITY-LEVEL METRICS
# ============================================================

def extract_entity_spans(tags):

    """
    Convert BIO tags into entity spans.

    Example:
        O, B-COMPANY, I-COMPANY, O

    Becomes:
        (COMPANY, start, end)
    """

    spans = []

    current_type = None
    current_start = None

    def close_entity(end_position):

        nonlocal current_type, current_start

        if current_type is not None:

            spans.append((
                current_type,
                current_start,
                end_position
            ))

        current_type = None
        current_start = None

    for index, tag in enumerate(tags):

        if tag == "O":

            close_entity(index)
            continue

        prefix, entity_type = tag.split("-", 1)

        if prefix == "B":

            close_entity(index)

            current_type = entity_type
            current_start = index

        elif prefix == "I":

            if current_type != entity_type:

                # Handle an invalid I-tag as a new entity.
                close_entity(index)

                current_type = entity_type
                current_start = index

    close_entity(len(tags))

    return set(spans)


def calculate_field_metrics(gold_documents, predicted_documents):

    results = {}

    for field in ALL_FIELDS:

        true_positive = 0
        false_positive = 0
        false_negative = 0

        for gold_tags, predicted_tags in zip(
            gold_documents,
            predicted_documents
        ):

            gold_spans = {
                span
                for span in extract_entity_spans(gold_tags)
                if span[0] == field
            }

            predicted_spans = {
                span
                for span in extract_entity_spans(predicted_tags)
                if span[0] == field
            }

            true_positive += len(
                gold_spans & predicted_spans
            )

            false_positive += len(
                predicted_spans - gold_spans
            )

            false_negative += len(
                gold_spans - predicted_spans
            )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        results[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

    macro_f1 = np.mean([
        results[field]["f1"]
        for field in MAIN_FIELDS
    ])

    results["MACRO_F1"] = float(macro_f1)

    return results


# ============================================================
# 8. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device):

    model.eval()

    gold_documents = []
    predicted_documents = []

    for batch in dataloader:

        metadata = batch.pop("metadata")

        model_inputs = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**model_inputs)

        predictions = torch.argmax(
            outputs.logits,
            dim=-1
        ).cpu().tolist()

        for row_index, item in enumerate(metadata):

            gold_tags = []
            predicted_tags = []

            for token_position, word_index in (
                item["first_subword_positions"]
            ):

                gold_id = item["original_word_labels"][word_index]

                predicted_id = predictions[
                    row_index
                ][token_position]

                gold_tags.append(
                    ID2LABEL[gold_id]
                )

                predicted_tags.append(
                    ID2LABEL[predicted_id]
                )

            gold_documents.append(gold_tags)
            predicted_documents.append(predicted_tags)

    return calculate_field_metrics(
        gold_documents,
        predicted_documents
    )


def print_metrics(title, metrics):

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    print(
        f"{'FIELD':<15}"
        f"{'PRECISION':>12}"
        f"{'RECALL':>12}"
        f"{'F1':>12}"
    )

    print("-" * 55)

    for field in ALL_FIELDS:

        result = metrics[field]

        print(
            f"{field:<15}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['f1']:>12.4f}"
        )

    print("-" * 55)

    print(
        f"{'MACRO F1':<15}"
        f"{'':>24}"
        f"{metrics['MACRO_F1']:>12.4f}"
    )

    print("=" * 65)


# ============================================================
# 9. TRAINING
# ============================================================

def train():

    set_seed(SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)
    print("Loading tokenizer:", MODEL_NAME)

    tokenizer = LayoutLMTokenizerFast.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Load existing dataset splits.
    # --------------------------------------------------------

    all_train_samples = load_json(TRAIN_FILE)
    test_samples = load_json(TEST_FILE)

    # Check for document overlap between train and test.
    train_ids = {
        sample["id"]
        for sample in all_train_samples
    }

    test_ids = {
        sample["id"]
        for sample in test_samples
    }

    overlap = train_ids & test_ids

    if overlap:
        raise ValueError(
            "Train/test document leakage detected. "
            f"Overlapping IDs: {sorted(overlap)[:10]}"
        )

    for sample in all_train_samples:
        validate_sample(sample)

    for sample in test_samples:
        validate_sample(sample)

    train_samples, validation_samples = (
        split_train_validation(
            all_train_samples,
            validation_ratio=VALIDATION_RATIO,
            seed=SEED
        )
    )

    # --------------------------------------------------------
    # Create tokenized datasets.
    # --------------------------------------------------------

    train_dataset = ReceiptDataset(
        train_samples,
        tokenizer,
        MAX_LENGTH
    )

    validation_dataset = ReceiptDataset(
        validation_samples,
        tokenizer,
        MAX_LENGTH
    )

    test_dataset = ReceiptDataset(
        test_samples,
        tokenizer,
        MAX_LENGTH
    )

    collator = LayoutLMCollator(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )

    # --------------------------------------------------------
    # Initialize LayoutLM.
    # --------------------------------------------------------

    print("\nLoading LayoutLM model...")

    model = LayoutLMForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_training_steps = (
        len(train_loader) * EPOCHS
    )

    warmup_steps = int(
        total_training_steps * WARMUP_RATIO
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    # --------------------------------------------------------
    # Training loop with validation checkpointing.
    # --------------------------------------------------------

    best_validation_f1 = -1.0
    epochs_without_improvement = 0

    history = []

    best_model_dir = os.path.join(
        OUTPUT_DIR,
        "best_model"
    )

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        for batch in train_loader:

            batch.pop("metadata")

            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**model_inputs)

            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / max(len(train_loader), 1)
        )

        print(
            f"Training loss: {average_loss:.4f}"
        )

        # ----------------------------------------------------
        # Validation evaluation.
        # ----------------------------------------------------

        validation_metrics = evaluate(
            model,
            validation_loader,
            device
        )

        print_metrics(
            "VALIDATION RESULTS",
            validation_metrics
        )

        validation_f1 = validation_metrics["MACRO_F1"]

        history.append({
            "epoch": epoch + 1,
            "training_loss": average_loss,
            "validation_macro_f1": validation_f1,
            "validation_metrics": validation_metrics,
        })

        # ----------------------------------------------------
        # Save best checkpoint.
        # ----------------------------------------------------

        if validation_f1 > best_validation_f1:

            best_validation_f1 = validation_f1

            epochs_without_improvement = 0

            print(
                "\nValidation improved. Saving best model..."
            )

            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

        else:

            epochs_without_improvement += 1

            print(
                "No validation improvement. "
                f"Patience: {epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping.
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # 10. FINAL TEST EVALUATION
    # ========================================================

    print("\nLoading best validation checkpoint...")

    best_model = LayoutLMForTokenClassification.from_pretrained(
        best_model_dir
    )

    best_model.to(device)

    test_metrics = evaluate(
        best_model,
        test_loader,
        device
    )

    print_metrics(
        "FINAL TEST RESULTS",
        test_metrics
    )

    # ========================================================
    # 11. SAVE EXPERIMENT RESULTS
    # ========================================================

    results = {

        "model": MODEL_NAME,

        "epochs_requested": EPOCHS,

        "best_validation_macro_f1": best_validation_f1,

        "test_metrics": test_metrics,

        "training_history": history,

        "configuration": {
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "validation_ratio": VALIDATION_RATIO,
            "seed": SEED,
        }
    }

    results_path = os.path.join(
        OUTPUT_DIR,
        "training_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        f"\nBest model saved at: {best_model_dir}"
    )

    print(
        f"Experiment results saved at: {results_path}"
    )


# ============================================================
# 12. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train()
