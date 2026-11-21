# Engineering Plan: Receipt Information Extraction with Layout-Aware Transformers

**Document Version:** 1.0.0  
**Scope:** Technical Architecture, Dataset Pipeline, Experimental Protocol, and Implementation Roadmap

---

## 1. Problem Definition & Objectives

The goal of this project is to build and benchmark an information extraction pipeline for scanned receipts that extracts four key fields:
- `COMPANY` (Merchant name)
- `DATE` (Transaction date)
- `TOTAL` (Grand total amount)
- `ADDRESS` (Merchant physical location, optional auxiliary target)

### Research Question
> **How much does 2D spatial layout information add over text alone?**  
> Formally evaluated as:
> $$\Delta F1_{\text{field}} = F1_{\text{LayoutLM}}(\text{field}) - F1_{\text{BERT}}(\text{field})$$

### The Experimental Hierarchy
To provide an uncompromising scientific comparison, we structure the evaluation along a three-tier progression:
1. **Tier 1 (Zero-Learning Baseline)**: OCR + Regular Expression / Spatial Rule Heuristics
2. **Tier 2 (Text-Only Deep Learning Baseline)**: Fine-tuned `bert-base-uncased` Token Classification
3. **Tier 3 (Multimodal Deep Learning Model)**: Fine-tuned `microsoft/layoutlm-base-uncased` Token Classification (Text + 2D Bounding Boxes)

All models will be trained on the identical training set, evaluated on the identical test set, and scored with the exact same unified evaluation harness (`metrics.py`).

---

## 2. Theoretical Foundations: LayoutLM (Xu et al., KDD 2020)

### 2.1 2D Spatial Coordinate Embedding
In conventional transformers, input embeddings only encode word content and 1D sequential order:
$$E_{\text{BERT}} = E_{\text{WordPiece}}(w_i) + E_{\text{1D\_position}}(i) + E_{\text{segment}}(s_i)$$

LayoutLM augments this representation by projecting normalized 2D spatial coordinates into trainable embedding tables:
$$E_{\text{LayoutLM}} = E_{\text{WordPiece}}(w_i) + E_{\text{1D\_position}}(i) + E_{x}(x_{0,i}) + E_{y}(y_{0,i}) + E_{x}(x_{1,i}) + E_{y}(y_{1,i})$$

Where $(x_{0,i}, y_{0,i})$ denotes the top-left coordinate and $(x_{1,i}, y_{1,i})$ denotes the bottom-right coordinate of token $i$.

### 2.2 Coordinate Normalization Protocol
To ensure invariance across variable scan resolutions and aspect ratios, all coordinate dimensions are scaled to the standard integer domain $[0, 1000]$:
$$x_{\text{norm}} = \min\left(1000, \max\left(0, \left\lfloor \frac{x}{W_{\text{image}}} \times 1000 \right\rfloor\right)\right)$$
$$y_{\text{norm}} = \min\left(1000, \max\left(0, \left\lfloor \frac{y}{H_{\text{image}}} \times 1000 \right\rfloor\right)\right)$$

Safety assertions require enforcing $x_0 \le x_1$ and $y_0 \le y_1$ to eliminate OCR bounding box inversions.

---

## 3. Dataset Architecture: ICDAR 2019 SROIE Task 3

### 3.1 Data Specification
- **Raw Data Structure**:
  - `img/`: High-resolution receipt scans (`.jpg`).
  - `box/`: OCR transcriptions with 8-point polygon bounding boxes: $[x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, \text{text}]$.
  - `entities/`: Ground truth JSON files with exact entity strings:
    ```json
    {
      "company": "BOOK XCESS SDN. BHD.",
      "date": "15/01/2019",
      "address": "NO. 5-1, JALAN MEDAN BUKIT INDAH...",
      "total": "96.40"
    }
    ```

### 3.2 Labeling Schema: BIO Token Classification
Token classification relies on the standard BIO tagging format:
- `O`: Outside of any named entity
- `B-COMPANY`, `I-COMPANY`
- `B-DATE`, `I-DATE`
- `B-TOTAL`, `I-TOTAL`
- `B-ADDRESS`, `I-ADDRESS`

### 3.3 Subword Tokenization Alignment
When a single OCR word token is fragmented by WordPiece subword tokenization (e.g., `"Starbucks"` $\to$ `["star", "##bucks"]`):
1. **Bounding Box Inheritance**: All subwords inherit the full word's bounding box coordinates $[x_0, y_0, x_1, y_1]$.
2. **Label Masking**:
   - The first subword receives the active label (e.g., `B-COMPANY` or `I-COMPANY`).
   - Subsequent subwords (`##bucks`) are assigned PyTorch's cross-entropy ignore index `-100` so that subword fragmentation does not artificially skew loss computation or inflate evaluation metrics.

---

## 4. Evaluation Protocol

### 4.1 Strict Entity-Level Extraction
Token-level micro F1 can be misleadingly elevated due to the preponderance of `O` tokens (~85% of receipt words). We therefore evaluate with **Entity-Level Exact Match F1**:
- Consecutive `B-` and `I-` predictions are grouped into extracted string spans.
- An extraction is counted as a True Positive (TP) if and only if:
  $$\text{Predicted\_Field\_Value} = \text{Ground\_Truth\_Value}$$
- Precision, Recall, and F1 are computed field-by-field:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  $$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 4.2 Sequence-Labeling F1 (seqeval)
In addition to exact string matching, standard span-level NER evaluation via the `seqeval` library is logged across epochs during model training.

---

## 5. Phased Implementation Roadmap

```text
Phase 0: Environment & Dependencies
├── Setup Python 3.10/3.11 virtual environment
├── Install PyTorch, Hugging Face Transformers, Datasets, Seqeval
└── Prepare project scaffolding and directory structure
      │
Phase 1: Dataset Pipeline & Preprocessing
├── Ingest SROIE dataset (train / test splits)
├── Normalize bounding boxes to [0, 1000]
├── Implement WordPiece <-> BIO span alignment algorithm
└── Unit tests for coordinate clipping and token alignment
      │
Phase 2: Evaluation Metric & Regex Baseline
├── Implement unified evaluation harness (src/metrics.py)
├── Implement deterministic regex & positional rules (src/baseline_regex.py)
└── Benchmark Baseline 1 on test split (populate Row 1 of results table)
      │
Phase 3: BERT Text-Only Baseline
├── Implement token classification training pipeline (src/train_bert.py / notebooks/FieldAI_Complete_Colab_Benchmark.ipynb)
├── Fine-tune bert-base-uncased on text tokens without coordinates
└── Score on test split using metrics.py (populate Row 2 of results table)
      │
Phase 4: LayoutLM Multimodal Implementation
├── Implement LayoutLM pipeline with 2D box embeddings (src/train_layoutlm.py / notebooks/FieldAI_Complete_Colab_Benchmark.ipynb)
├── Fine-tune microsoft/layoutlm-base-uncased
└── Score on test split using metrics.py (populate Row 3 of results table)
      │
Phase 5: Comparative Analysis & Report Synthesis
├── Calculate field-wise ΔF1 = F1_LayoutLM - F1_BERT
├── Generate comparative visualization plots & error analysis
└── Finalize README.md with complete empirical results
```

---

## 6. Detailed Phase Specifications

### Phase 0: Environment & Dependencies
- **Target Files**: `requirements.txt`, environment verification scripts.
- **Dependencies**:
  - `torch >= 2.0.0`
  - `transformers >= 4.30.0`
  - `datasets >= 2.12.0`
  - `seqeval >= 1.2.2`
  - `pytesseract >= 0.3.10`
  - `pillow >= 9.5.0`
  - `scikit-learn >= 1.2.0`
  - `matplotlib >= 3.7.0`
  - `pandas >= 2.0.0`

### Phase 1: Data Pipeline (`src/dataset.py`)
- Parse raw SROIE dataset:
  - Load polygon coordinates $[x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4]$ and convert to axis-aligned bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$.
  - Scale boxes to $[0, 1000]$.
  - Align word-level text with ground-truth entity values using longest common substring matching.
  - Generate Hugging Face compatible `DatasetDict` (`train`, `val`, `test`).

### Phase 2: Metrics & Baseline 1 (`src/metrics.py`, `src/baseline_regex.py`)
- `src/metrics.py`:
  - Input: Dictionary of predicted entity strings per sample `{id: {"company": ..., "date": ..., "total": ...}}`.
  - Output: Precision, Recall, F1 for each field, Macro F1, Micro F1.
- `src/baseline_regex.py`:
  - `DATE`: Pattern-matches `\d{1,2}[/-]\d{1,2}[/-]\d{2,4}` and alphanumeric variants (`15 JAN 2019`).
  - `TOTAL`: Regex `(?:TOTAL|AMOUNT DUE|NET)\s*[:\$]?\s*([0-9]+\.[0-9]{2})`.
  - `COMPANY`: Takes the topmost non-empty line of text.

### Phase 3: BERT Text-Only (`src/train_bert.py` & Step 5 in `FieldAI_Complete_Colab_Benchmark.ipynb`)
- Model: `AutoModelForTokenClassification.from_pretrained("bert-base-uncased")`.
- Inputs: `input_ids`, `attention_mask`, `labels`.
- Training parameters:
  - Learning rate: $3 \times 10^{-5}$
  - Batch size: 8 / 16
  - Optimizer: AdamW with linear warmup
  - Epochs: 5–10 with early stopping on validation entity F1.

### Phase 4: LayoutLM Multimodal (`src/train_layoutlm.py` & Step 6 in `FieldAI_Complete_Colab_Benchmark.ipynb`)
- Model: `AutoModelForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")`.
- Inputs: `input_ids`, `bbox`, `attention_mask`, `token_type_ids`, `labels`.
- Bounding box dimension: shape `(batch_size, sequence_length, 4)`.
- Training parameters:
  - Learning rate: $5 \times 10^{-5}$
  - Batch size: 8 / 16
  - Optimizer: AdamW with weight decay $0.01$
  - Epochs: 5–10 with early stopping on validation entity F1.

### Phase 5: Empirical Synthesis & Deliverables
- Compile final test metrics table across all 3 models.
- Compute field-wise $\Delta F1$ to empirically confirm or refute the layout hypothesis.
- Conduct qualitative error analysis on edge cases:
  - Subtotal vs. Grand Total misclassifications.
  - Multi-date receipts (e.g., printed date vs. transaction date).
  - OCR recognition errors on noisy thermal receipt scans.

---

## 7. Risk Management & Engineering Pitfalls

| Potential Issue | Technical Impact | Engineering Mitigation |
|---|---|---|
| **Subword Box Mismatch** | Subwords without spatial coordinates trigger embedding runtime errors | Duplicate parent word's 2D bounding box across all constituent subword tokens. |
| **Coordinate Out-of-Bounds** | Coordinates $< 0$ or $> 1000$ cause index errors in LayoutLM embedding table | Explicitly clamp: `box = [min(1000, max(0, c)) for c in box]`. |
| **Inverted BBoxes** | OCR box skew can lead to $x_0 > x_1$ or $y_0 > y_1$ | Enforce axis alignment: $x_0 = \min(x), x_1 = \max(x), y_0 = \min(y), y_1 = \max(y)$. |
| **Class Imbalance** | `O` class tokens dominate ~85% of loss | Mask subword continuation tokens with `-100`; evaluate on Entity-level F1 rather than token accuracy. |
| **Metric Divergence** | Comparing token-F1 against regex exact strings produces invalid comparison | All models must decode to `{company: str, date: str, total: str}` and pass through `metrics.py`. |
