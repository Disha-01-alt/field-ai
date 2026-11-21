# FieldAI: Receipt Information Extraction with Layout-Aware Transformers

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/transformers/)
[![Dataset: SROIE 2019](https://img.shields.io/badge/Dataset-ICDAR_2019_SROIE-green.svg)](https://rrc.cvc.uab.es/?ch=13)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

A deep learning benchmark investigating the impact of **2D spatial layout representations** on key information extraction (KIE) from scanned receipts and invoices.

---

## Abstract & Research Objective

Visually rich documents (VRDs)—such as commercial receipts and utility invoices—contain critical structured data (e.g., merchant name, transaction date, grand total). While standard Natural Language Processing (NLP) models treat documents as 1D sequential text, real-world document semantics heavily depend on **two-dimensional spatial layouts** (table alignment, spatial headers, key-value proximity).

This project conducts an empirical comparison across three model paradigms to quantify the exact contribution of 2D spatial layout information over pure textual signals:

$$\Delta F1_{\text{field}} = F1_{\text{LayoutLM}}(\text{field}) - F1_{\text{BERT}}(\text{field})$$

### Core Research Questions
1. **Marginal Utility of Layout**: How much performance gain ($\Delta F1$) is achieved when augmenting textual representations with 2D bounding-box spatial embeddings?
2. **Field-Specific Vulnerability**: Which document fields (e.g., `TOTAL` vs. `DATE` vs. `COMPANY`) exhibit the highest sensitivity to spatial configuration? For example, does spatial layout resolve recurring ambiguities between line-item prices, subtotals, and final totals?
3. **Robustness vs. Heuristics**: How do learned multimodal transformers compare against classical OCR + regular expression heuristics under document skew and OCR recognition noise?

---

## System Architecture

```text
               +----------------------------------+
               |     Scanned Receipt Image        |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |         OCR Engine               |
               |  (Text Tokens + Bounding Boxes)  |
               +----------------------------------+
                                |
        +-----------------------+-----------------------+
        |                                               |
        v                                               v
+-------------------------------+               +-------------------------------+
|    Rule-Based Baseline        |               |   Spatial Normalization       |
|    (Regex & Heuristic Rules)  |               |   (Scale (x, y) to [0, 1000])  |
+-------------------------------+               +-------------------------------+
        |                                               |
        |                       +-----------------------+-----------------------+
        |                       |                                               |
        |                       v                                               v
        |       +-------------------------------+               +-------------------------------+
        |       |    BERT-base (Text-Only)      |               |  LayoutLM-base (Multimodal)   |
        |       |    - WordPiece Tokenization   |               |  - WordPiece Tokenization     |
        |       |    - 1D Sequence Position     |               |  - 1D Sequence Position       |
        |       |    - No Bounding Box Context  |               |  - 2D Bounding Box Embeddings |
        |       +-------------------------------+               +-------------------------------+
        |                       |                                               |
        +-----------------------+-----------------------------------------------+
                                |
                                v
               +----------------------------------+
               |    Unified Evaluation Harness    |
               |     (seqeval & Exact Match)      |
               |   Precision | Recall | Micro/Macro F1 |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |   Field-Wise Gain Analysis       |
               |     ΔF1 = F1_LayoutLM - F1_BERT  |
               +----------------------------------+
```

---

## Dataset: ICDAR 2019 SROIE (Task 3)

We utilize the **ICDAR 2019 Scanned Receipts OCR and Information Extraction (SROIE)** benchmark dataset.

- **Images**: Scanned real-world receipts varying in rotation, illumination, resolution, and merchant formatting.
- **Annotations**:
  - OCR bounding boxes: Word-level text bounding boxes $[x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, \text{text}]$.
  - Key Information Extraction (KIE) ground truth:
    - `company`: Merchant / store name.
    - `date`: Transaction date.
    - `total`: Grand total currency amount.
    - `address`: Merchant physical location (optional auxiliary field).
- **Coordinate Normalization**: Following Xu et al. (KDD 2020), all bounding box coordinates are mapped onto an integer scale $[0, 1000]$:
  $$x_{\text{norm}} = \min\left(1000, \max\left(0, \left\lfloor \frac{x}{W} \times 1000 \right\rfloor\right)\right)$$
  $$y_{\text{norm}} = \min\left(1000, \max\left(0, \left\lfloor \frac{y}{H} \times 1000 \right\rfloor\right)\right)$$

---

## Model Progression & Methodology

### 1. Baseline 1: OCR + Regex (Rule-Based)
- **Mechanism**: Extracts textual tokens via OCR and evaluates structured pattern regular expressions:
  - `DATE`: Pattern-matching on standard receipt date conventions (`DD/MM/YYYY`, `YYYY-MM-DD`, month abbreviations).
  - `TOTAL`: Currency extraction filtered by proximity heuristics near keywords (`TOTAL`, `GRAND TOTAL`, `AMOUNT DUE`).
  - `COMPANY`: Top-of-page vertical positional heuristic selecting prominent initial text blocks.
- **Purpose**: Serves as the zero-learning operational baseline.

### 2. Baseline 2: BERT (Text-Only Transformer)
- **Backbone**: `bert-base-uncased` fine-tuned for token classification.
- **Input**: Sequential WordPiece tokens with standard 1D positional encodings without spatial layout features.
- **Target**: BIO sequence labeling (`B-COMPANY`, `I-COMPANY`, `B-DATE`, `I-DATE`, `B-TOTAL`, `I-TOTAL`, `O`).
- **Purpose**: Represents modern NLP performance without multimodal document grounding.

### 3. Core Model: LayoutLM (Multimodal: Text + 2D Layout)
- **Backbone**: `microsoft/layoutlm-base-uncased` fine-tuned for token classification.
- **Input**: Concatenated WordPiece tokens + 1D positional encodings + **2D spatial box embeddings** $[x_0, y_0, x_1, y_1]$.
- **Embedding Formulation**:
  $$E_i = E_{\text{text}}(w_i) + E_{\text{1D}}(i) + E_{x}(x_{0,i}) + E_{y}(y_{0,i}) + E_{x}(x_{1,i}) + E_{y}(y_{1,i})$$
- **Purpose**: Tests whether 2D structural inductive bias resolves semantic ambiguities (such as distinguishing subtotals from grand totals).

---

## Experimental Results

> *Evaluated on the frozen SROIE Task 3 test split (126 receipts) using strict entity-level exact matching.*

| Model Paradigm | Input Modality | Company F1 | Date F1 | Total F1 | Overall Macro F1 |
|:---|:---|:---:|:---:|:---:|:---:|
| **OCR + Regex Baseline** | Plain Text (Rules) | 55.56% | 96.36% | 61.11% | **71.01%** |
| **BERT-base** | 1D Text Tokens | *In Training* | *In Training* | *In Training* | *In Training* |
| **LayoutLM-base** | Text + 2D Bounding Boxes | *Pending* | *Pending* | *Pending* | *Pending* |
| **Empirical Delta ($\Delta F1$)** | *Layout Contribution* | **+Pending** | **+Pending** | **+Pending** | **+Pending** |

*(Results are actively updated as each model run completes on the frozen test split.)*

---

## Project Structure

```text
field-ai/
├── README.md                 # Project documentation and empirical benchmark
├── requirements.txt          # Python runtime dependencies
├── doc/
│   └── PLAN.md               # Detailed architectural design and engineering execution plan
├── data/                     # Dataset storage (SROIE raw & processed, gitignored)
│   ├── sroie/                # Raw SROIE box and key files
│   └── processed/            # Frozen train.json and test.json splits
├── src/
│   ├── dataset.py            # Dataset loaders, coordinate normalization, BIO alignment
│   ├── metrics.py            # Unified entity-level Precision/Recall/F1 evaluation engine
│   ├── baseline_regex.py     # Deterministic OCR + Regex extraction baseline
│   ├── train_bert.py         # 1D text-only BERT fine-tuning pipeline
│   ├── train_layoutlm.py     # Multimodal LayoutLM fine-tuning pipeline
│   └── visualize.py          # Benchmark plotting and SVG chart generation
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb    # Exploratory data analysis & baseline demonstration
│   ├── 02_bert_finetuning.ipynb     # Text-only BERT model fine-tuning & evaluation
│   └── 03_layoutlm_finetuning.ipynb # Multimodal LayoutLM fine-tuning & ablation study
└── results/                  # Serialized evaluation metrics, benchmark SVGs, and plots
```

---

## Installation & Setup

### 1. Clone & Environment Configuration
```bash
git clone https://github.com/your-username/field-ai.git
cd field-ai

# Create virtual environment (Python 3.10 or 3.11 recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. System OCR Engine (Optional if using pre-extracted OCR)
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y tesseract-ocr
```

---

## Execution Pipeline

### Step 1: Preprocess Data & Verify BIO Tagging
```bash
python src/dataset.py --dataset_dir data/sroie --output_dir data/processed
```

### Step 2: Run OCR + Regex Baseline
```bash
python src/baseline_regex.py --data_dir data/processed --output results/regex_baseline.json
```

### Step 3: Train & Evaluate Transformer Models
Training notebooks are ready for local or cloud GPU acceleration (Google Colab / Kaggle):
- Open `notebooks/02_bert_finetuning.ipynb` for the 1D text baseline.
- Open `notebooks/03_layoutlm_finetuning.ipynb` for the multimodal 2D layout model.

### Step 4: Aggregate Metrics & Generate Comparative Analysis
```bash
python src/metrics.py --results_dir results/ --generate_plots
```

---

## Primary References

1. **LayoutLM Reference Paper**:  
   Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., & Zhou, M. (2020). *LayoutLM: Pre-training of Text and Layout for Document Image Understanding*. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD 2020). [arXiv:1912.13318](https://arxiv.org/abs/1912.13318)
2. **SROIE Benchmark**:  
   Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., & Jawahar, C. V. (2019). *ICDAR 2019 Competition on Scanned Receipts OCR and Information Extraction (SROIE)*. [Official Task Overview](https://rrc.cvc.uab.es/?ch=13)
3. **Seqeval Evaluation Metric**:  
   Nakayama, H. (2018). *seqeval: A Python framework for sequence labeling evaluation*. [GitHub Repository](https://github.com/chakki-works/seqeval)

---

## License
This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
