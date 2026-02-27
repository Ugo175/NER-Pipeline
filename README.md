# E-Commerce Title Aspect Extraction Pipeline  
**Scalable NLP + Data Engineering Project (2M+ Listings)**

---

## 📌 Overview

This project builds a scalable pipeline to extract structured product aspects from **2,000,000+ noisy e-commerce listing titles** (German automotive marketplace data).

The goal is to transform unstructured, inconsistent, real-world listing titles into structured entity outputs such as:

- `Hersteller` (Manufacturer)  
- `Kompatible_Fahrzeug_Marke` (Compatible Vehicle Make)  
- `Kompatibles_Fahrzeug_Modell` (Compatible Vehicle Model)  
- `Produktart` (Product Type)  
- `Herstellernummer` (MPN)  
- and more  

The challenge combines **large-scale data processing, strict formatting constraints, and Named Entity Recognition (NER)** under real-world noise conditions.

---


---

## Key Challenges

### 1️⃣ Real-World Noisy Data

- Misspellings  
- Abbreviations (e.g., "WaPu")  
- Inconsistent tokenization (`Zahnriemensatz+WaPu`)  
- Mixed-language terms  
- Duplicate tokens  
- Human annotation inconsistencies  

### 2️⃣ Strict Entity Reconstruction Rules

- Empty tags indicate continuation  
- Same tag ≠ continuation  
- Multi-token entities must be reconstructed using ASCII space  
- Duplicate tokens must be preserved  
- Submission formatting must follow exact constraints  

### 3️⃣ Scale

- 2,000,000 records  
- Gzip compressed  
- UTF-8 encoded  
- TAB-separated  
- CSV-style quoting  
- Embedded TAB characters possible  

This required careful memory management and deterministic transformation logic.

---

## Machine Learning Approach

The project uses a Named Entity Recognition (NER) pipeline:

- Custom token-level tagging logic  
- Context-aware feature engineering  
- Sequence modeling (planned: BiLSTM / Transformer-based model)  
- Evaluation using precision, recall, and F1-score  
- Handling annotation inconsistencies in training data  

Special care was taken to:

- Preserve raw tokens (no normalization allowed)  
- Avoid unwanted NA conversion in pandas  
- Maintain strict token ordering  

---

## Data Engineering Focus

This project emphasizes:

- Stream-based ingestion of compressed datasets  
- Schema validation and integrity checks  
- Deterministic entity reconstruction logic  
- Controlled serialization (no CSV quoting in submission)  
- Reproducible data pipeline design  
- Logging and verification checks before submission  

The system mimics production ETL workflows rather than a simple notebook experiment.

---

## Evaluation Strategy

- Token-level performance metrics  
- Entity-level reconstruction accuracy  
- Submission format validation tests  
- Edge-case handling tests (duplicate tokens, multi-token spans, empty tags)  

---


---

## Results (To Be Updated)

- Model Performance:
  - Precision:
  - Recall:
  - F1-score:
- Submission consistency validation: ✅  
- Large-scale inference tested on full dataset: ✅  

---

## What This Project Demonstrates

- Handling messy, human-generated production data  
- Designing scalable NLP pipelines  
- Combining data engineering and machine learning  
- Working under strict output constraints  
- Building deterministic, reproducible systems  

---

## Future Improvements

- Transformer-based contextual embeddings  
- Active learning for ambiguous tags  
- Automated data quality scoring  
- Pipeline orchestration (Airflow / Prefect)  
- Distributed processing (Spark)  

