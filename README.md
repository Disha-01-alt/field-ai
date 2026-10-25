
# FieldAI — Receipt Information Extraction

## Project Overview

| Category | Details |
|---|---|
| **Project Type** | Multimodal (Image + Text) / Deep Learning |
| **Domain** | Document AI, NLP, Information Extraction |
| **Objective** | Extract structured information from scanned receipts |
| **Target Fields** | Company Name, Date, Total Amount |
| **Datasets** | SROIE (ICDAR 2019) / CORD |
| **Models** | OCR + Regex, BERT, LayoutLM |

## Problem Statement

Receipts contain important information such as merchant names, transaction dates, and total amounts. Extracting these fields manually is time-consuming, while rule-based approaches can fail when receipt layouts vary or multiple dates and monetary values appear in the same document.

This project investigates whether incorporating the two-dimensional layout of text improves receipt information extraction compared with using textual information alone.

## Objectives

- Build an OCR + Regex baseline for receipt field extraction.
- Implement a text-only BERT token-classification model.
- Implement a layout-aware LayoutLM model using text and bounding-box coordinates.
- Fine-tune pretrained models on a receipt dataset.
- Compare model performance using field-wise Precision, Recall, and F1 scores.
- Analyze the contribution of layout information to extraction accuracy.

## Dataset

### 1. SROIE — ICDAR 2019

A dataset for scanned receipt text recognition and key information extraction.

- Dataset: https://rrc.cvc.uab.es/?ch=13

### 2. CORD — Consolidated Receipt Dataset

A receipt dataset containing structured annotations for receipt understanding and information extraction.

- Dataset: https://huggingface.co/datasets/naver-clova-ix/cord-v2

The final dataset will be selected after reviewing annotation compatibility and the baseline paper's experimental setup.

## Methodology

### Phase 1: Literature Review

- Study the baseline research paper.
- Identify its dataset, methodology, evaluation metrics, and reported results.
- Define the target fields and experimental setup.

### Phase 2: Dataset Preparation and OCR

- Load receipt images and ground-truth annotations.
- Extract text using an OCR tool such as Tesseract or PaddleOCR.
- Obtain word-level bounding boxes.
- Align OCR output with the annotated fields.
- Prepare training, validation, and test splits.

### Phase 3: OCR + Regex Baseline

- Extract text from receipt images.
- Use regular expressions and rules to identify dates and monetary values.
- Implement rules for merchant name extraction.
- Evaluate predictions against ground-truth annotations.

### Phase 4: Transformer-Based Models

**BERT — Text-Only Approach**

- Use OCR-extracted words as model input.
- Apply token classification to identify company names, dates, and totals.
- Fine-tune pretrained BERT weights on the receipt dataset.

**LayoutLM — Text + Layout Approach**

- Use OCR-extracted words and their bounding-box coordinates.
- Fine-tune a pretrained LayoutLM token-classification model.
- Evaluate whether spatial information improves field extraction.

Where suitable checkpoints are available, pretrained inference may also be evaluated before fine-tuning.

### Phase 5: Evaluation and Comparison

- Compare OCR + Regex, fine-tuned BERT, and fine-tuned LayoutLM.
- Calculate Precision, Recall, and F1 for each field.
- Analyze common errors, including confusion between subtotal and total.
- Measure the improvement associated with layout information.
- Record training and inference time where feasible.

### Phase 6: Results and Demonstration

- Prepare comparison tables and visualizations.
- Document findings and limitations.
- Optionally develop a demo that accepts a receipt image and returns structured JSON.

## System Architecture

```text
Receipt Image
     |
     v
 OCR Extraction
 (Text + Bounding Boxes)
     |
     +---------------------------+
     |                           |
     v                           v
OCR + Regex                 Transformer Models
 Baseline                        |
                         +------+------+
                         |             |
                         v             v
                       BERT         LayoutLM
                    (Text Only)   (Text + Layout)
                         |             |
                         +------+------+
                                |
                                v
                       Extracted Fields
                                |
                                v
                       Evaluation Metrics
```

## Evaluation Metrics

The models will be evaluated using Precision, Recall, and F1 score.

- **Precision:** Proportion of predicted field instances that are correct.
- **Recall:** Proportion of ground-truth field instances successfully extracted.
- **F1 Score:** Harmonic mean of Precision and Recall.

The F1 score is calculated as:

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

### Measuring Layout Contribution

The improvement from adding layout information will be measured by:

\[
\Delta F1 = F1_{\text{LayoutLM}} - F1_{\text{BERT}}
\]

A positive difference indicates better performance by LayoutLM under the selected evaluation setup.

## Results

| Model | Merchant F1 | Date F1 | Total F1 |
|---|---:|---:|---:|
| OCR + Regex | TBD | TBD | TBD |
| Fine-tuned BERT | TBD | TBD | TBD |
| Fine-tuned LayoutLM | TBD | TBD | TBD |

*Results will be updated after running the experiments.*

For a fair comparison, the models will use consistent dataset splits, field definitions, and evaluation procedures.

## Technology Stack

- **Programming Language:** Python
- **OCR:** Tesseract or PaddleOCR
- **Deep Learning:** PyTorch
- **Transformer Models:** Hugging Face Transformers
- **Dataset Processing:** Hugging Face Datasets, pandas
- **Evaluation:** scikit-learn and custom extraction metrics
- **Visualization:** Matplotlib

## Expected Deliverables

- Receipt dataset preparation pipeline
- OCR-based extraction baseline
- Fine-tuned BERT model
- Fine-tuned LayoutLM model
- Evaluation scripts and comparison results
- Field-wise error analysis
- Optional receipt extraction demo

## Reference

Xu et al. (2020). *LayoutLM: Pre-training of Text and Layout for Document Image Understanding.*

Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD 2020).

- Paper: https://arxiv.org/abs/1912.13318

## Project Status

**Status:** Planning and experimentation in progress.

The final dataset, model configurations, experimental results, and conclusions will be documented as the project develops.
