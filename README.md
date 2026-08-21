# Product Category Classifier from Package Images

An OCR + ML pipeline that extracts text from FMCG product packaging 
and predicts product categories — grounded in Open Food Facts data 
and real retail workflows.

## Overview

This project mirrors a real-world enterprise workflow (validated during 
my time at NielsenIQ): given a product's packaging image, extract visible 
text using OCR, then predict the product's category using a machine learning 
classifier — with confidence-ranked suggestions rather than a single rigid guess.

**Pipeline:**
Image → OCR (text extraction) → ML Classifier → Predicted Category (with confidence ranking)

## Dataset

Data is sourced from [Open Food Facts](https://world.openfoodfacts.org/), 
a free, open, crowdsourced database of food products worldwide (3M+ products). 
Rather than downloading the full multi-GB database, data is streamed via 
[Hugging Face](https://huggingface.co/datasets/openfoodfacts/product-database) 
and a working sample is extracted and cleaned locally.

Final working dataset: **6,033 products across 10 categories**

## Stages

### ✅ Stage 1 — Data Preparation
- Streamed a working sample from Open Food Facts (avoided downloading the full ~7.7GB database)
- Extracted clean text fields from multi-language nested fields
- Cleaned mislabeled and missing category values including disguised null strings
- Merged duplicate French/English category labels
- Filtered to 10 top-level categories

### ✅ Stage 2 — Baseline Text Classification Model
- Built a TF-IDF + Logistic Regression baseline model predicting product 
  category from product name
- 80/20 train/test split
- **83% accuracy on clean text**
- Diagnosed class imbalance: strong performance on high-frequency categories 
  (Snacks, Beverages), weaker recall on low-frequency ones (Frozen Foods)

### ✅ Stage 3 — OCR Integration
- Fetched real product images via Open Food Facts API using barcode-linked URLs
- Downloaded ~30 product images locally
- Ran EasyOCR on each image with a 0.65 confidence threshold to filter noise
- Fed OCR-extracted text into the trained Stage 2 classifier
- **53% accuracy on OCR text** vs 83% on clean text
- Key finding: the bottleneck is OCR input quality, not the classifier

### ✅ Stage 4 — Transformer Model Comparison
- Used DistilBERT as a pretrained feature extractor to generate 768-dimensional 
  embeddings for each product name
- Trained Logistic Regression on top of these embeddings
- Tested on both clean text and OCR text

| Method | Clean Text | OCR Text |
|---|---|---|
| TF-IDF + Logistic Regression | 83% | 53% |
| DistilBERT + Logistic Regression | 75% | 50% |

- Key finding: TF-IDF outperformed DistilBERT on this dataset. Product names 
  are short and keyword-heavy — contextual understanding adds no advantage 
  when category keywords are already obvious. A fine-tuned DistilBERT trained 
  specifically on food data would likely close this gap.

### ✅ Stage 5 — Confidence-Ranked Output
- Added confidence scores to predictions — top 3 category guesses ranked by probability
- Full end-to-end pipeline: image in, ranked predictions out
- Example outputs:
  - "DELUXE ICE CREAM COOKIES N' CREAM" → Desserts 97.3%
  - "milk dutch hot cocoa chocolate" → Beverages 69.9%
  - "Sorrel Ginger" → Plant-based foods 32.3% (low confidence — little text extracted)
- Confidence scores expose the real bottleneck: when OCR extracts rich text, 
  predictions are confident and correct. When little text is extracted, 
  confidence drops accordingly.

## Tech Stack

- Python, pandas
- Hugging Face `datasets` (streaming data access)
- scikit-learn (TF-IDF, Logistic Regression)
- EasyOCR (text extraction from product images)
- Transformers / DistilBERT (pretrained feature extraction)
- PyTorch (DistilBERT backend)
- Open Food Facts API (barcode-linked image URLs)

## Key Findings

- Clean text classification works well at 83% with a simple TF-IDF baseline
- OCR noise is the main bottleneck — not the classifier
- Transformer models are not always better — for short, keyword-heavy text, 
  TF-IDF wins
- Confidence scores make the tool more honest and production-ready

## Next Steps

- Image preprocessing (denoising, contrast adjustment) before OCR to recover accuracy
- Fine-tune DistilBERT specifically on food product data
- Deploy as a simple web app where users can upload a product image and get predictions
