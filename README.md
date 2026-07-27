# product-category-classifier
An OCR + ML pipeline that extracts text from FMCG product packaging and predicts product categories — grounded in Open Food Facts data and real retail workflows.
cat > README.md << 'EOF'
# Product Category Classifier from Package Images

> **Status: Work in Progress** — This project is under active development. Current progress covers data preparation and a baseline text classification model. OCR and image-based prediction are in progress.

## Overview

This project mirrors a real-world enterprise workflow (validated during my time at NielsenIQ): given a product's packaging, extract text from the image using OCR, then predict the product's category using a machine learning classifier — with confidence-ranked suggestions rather than a single rigid guess.

The pipeline is being built in stages:

**Image → OCR (text extraction) → ML Classifier → Predicted Category (with confidence ranking)**

## Dataset

Data is sourced from [Open Food Facts](https://world.openfoodfacts.org/), a free, open, crowdsourced database of food products worldwide (3M+ products). The dataset provides both product text fields (name, category, ingredients) and product images, linked by barcode — making it well-suited for an OCR + classification pipeline.

Rather than downloading the full multi-GB database, data is streamed via the [Hugging Face](https://huggingface.co/datasets/openfoodfacts/product-database) `openfoodfacts/product-database` dataset and a working sample is extracted and cleaned locally.

## Progress So Far

### ✅ Stage 1 — Data Preparation

- Streamed a working sample from the Open Food Facts dataset (avoided downloading the full ~7.7GB database)
- Extracted clean text fields (`product_name`, `ingredients_text`) from multi-language nested fields
- Cleaned mislabeled/missing category values (including disguised `"null"` strings)
- Merged duplicate French/English category labels
- Filtered to a clean, labeled working set of 6,165 products across 10 top-level categories

### ✅ Stage 2 — Baseline Text Classification Model

- Built a TF-IDF + Logistic Regression baseline model predicting product category from product name
- Train/test split (80/20) with an 83% baseline accuracy
- Diagnosed class imbalance across categories using precision/recall analysis (e.g. strong performance on high-frequency categories like Snacks and Beverages, weaker recall on low-frequency categories like Frozen Foods)

### ⏳ Stage 3 — OCR Integration (In Progress)

- Extract product images using barcode-linked URLs
- Run EasyOCR on package images to extract text
- Feed OCR-extracted text into the trained classifier to validate the full image → category pipeline

### 🔜 Stage 4 — Transformer Model Comparison (Planned)

- Compare baseline TF-IDF + Logistic Regression performance against a pretrained transformer-based text classifier

### 🔜 Stage 5 — Confidence-Ranked Output & Wrap-Up (Planned)

- Return ranked category suggestions with confidence scores instead of a single prediction
- Final write-up of results and findings

## Tech Stack

- Python, pandas
- Hugging Face `datasets` (streaming data access)
- scikit-learn (TF-IDF, Logistic Regression)
- EasyOCR (upcoming)
- Transformers (upcoming)

## Notes

This README will be updated as each stage is completed. Code and notebooks will be added to this repository progressively as the project develops.
EOF
