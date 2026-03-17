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

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd NER-Pipeline
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Create sample data for testing:**
```bash
python main.py --mode sample-data --output data/raw/sample_data.tsv --samples 1000
```

2. **Train a model:**
```bash
python main.py --mode train --data data/raw/sample_data.tsv
```

3. **Generate predictions:**
```bash
python main.py --mode predict --model models/best_model --data data/raw/test_data.tsv --output submissions/predictions.tsv
```

4. **Evaluate model:**
```bash
python main.py --mode evaluate --model models/best_model --data data/raw/validation_data.tsv
```

---

## 📁 Project Structure

```
NER-Pipeline/
├── src/                          # Source code
│   ├── data_processing/          # Data ingestion and preprocessing
│   │   ├── ingestion.py          # Compressed file handling
│   │   ├── preprocessor.py       # NER preprocessing
│   │   └── validator.py          # Data validation
│   ├── models/                   # NER models
│   │   ├── base_ner_model.py     # Abstract base class
│   │   ├── bert_ner.py          # BERT-based NER
│   │   └── trainer.py            # Model training utilities
│   ├── evaluation/               # Evaluation framework
│   │   ├── metrics.py           # Metric calculations
│   │   ├── evaluator.py         # Model evaluation
│   │   └── analyzer.py           # Results analysis
│   └── utils/                    # Utilities
│       ├── entity_reconstructor.py # Entity reconstruction
│       ├── submission_formatter.py # Submission formatting
│       └── config.py             # Configuration management
├── data/                         # Data directories
│   ├── raw/                      # Raw input data
│   ├── processed/                # Processed data
│   └── output/                   # Output files
├── models/                       # Trained models
├── notebooks/                    # Jupyter notebooks
├── logs/                         # Log files
├── submissions/                  # Submission files
├── tests/                        # Unit tests
├── main.py                       # Main entry point
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## 🔧 Key Features

### Data Processing
- **Streaming ingestion** of gzip-compressed TAB-separated files
- **Memory-efficient processing** for 2M+ records
- **Robust error handling** for noisy real-world data
- **Schema validation** and data quality checks

### NER Pipeline
- **Token-level tagging** with BIO scheme
- **Context-aware feature engineering**
- **Multi-token entity reconstruction** with ASCII space
- **Duplicate token preservation** for submission compliance

### Entity Reconstruction
- **Strict formatting rules** compliance
- **Continuation tag handling** (empty tags)
- **Multi-entity support** per type
- **Validation** against original tokens

### Evaluation Framework
- **Token-level metrics** (precision, recall, F1)
- **Entity-level metrics** using seqeval
- **Exact match evaluation** for reconstruction
- **Comprehensive error analysis**
- **Visual reports** with plots

### Submission Formatting
- **TAB-separated output** format
- **No CSV quoting** compliance
- **UTF-8 encoding** support
- **Format validation** before submission

---

## 📊 Entity Types

The pipeline extracts the following entity types from German automotive listings:

| Entity Type | Description | Example |
|-------------|-------------|---------|
| `Hersteller` | Manufacturer | Bosch, Valeo, Brembo |
| `Kompatible_Fahrzeug_Marke` | Compatible Vehicle Make | BMW, Mercedes, Audi |
| `Kompatibles_Fahrzeug_Modell` | Compatible Vehicle Model | 3er, A4, Golf |
| `Produktart` | Product Type | Bremsscheibe, Luftfilter |
| `Herstellernummer` | MPN | 123456789 |
| `EAN` | European Article Number | 1234567890123 |
| `Zustand` | Condition | Neu, Gebraucht |
| `Farbe` | Color | Schwarz, Silber |
| `Material` | Material | Kunststoff, Metall |
| `Anzahl` | Quantity | 1, 2, 10 |
| `OEM` | Original Equipment Manufacturer | OEM, Nachbau |

---

## 🏗️ Architecture

### Data Flow
1. **Ingestion**: Stream compressed files → DataFrame chunks
2. **Preprocessing**: Tokenize titles → Extract features → Generate tags
3. **Training**: Feature vectors → NER model → Trained parameters
4. **Prediction**: New titles → Tokenization → Model inference → Tags
5. **Reconstruction**: Tags + tokens → Structured entities
6. **Formatting**: Entities → TAB-separated submission

### Key Challenges Addressed

#### 1️⃣ Real-World Noisy Data
- Misspellings and abbreviations (e.g., "WaPu" for "Wasserpumpe")
- Inconsistent tokenization (`Zahnriemensatz+WaPu`)
- Mixed-language terms and duplicate tokens
- Human annotation inconsistencies

#### 2️⃣ Strict Entity Reconstruction Rules
- Empty tags indicate continuation
- Same tag ≠ continuation
- Multi-token entities must use ASCII space
- Duplicate tokens must be preserved
- Submission formatting must follow exact constraints

#### 3️⃣ Scale Requirements
- 2,000,000+ records processing
- Gzip compression handling
- Memory-efficient streaming
- Deterministic transformation logic

---

## 🧪 Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Configuration
The pipeline uses a comprehensive configuration system. Create a custom config:

```python
from src.utils.config import Config

config = Config()
config.model.learning_rate = 1e-4
config.data.chunk_size = 5000
config.save_to_file(Path("config.json"))
```

---

## 📈 Performance

### Metrics (To Be Updated)
- **Token-level F1**: TBD
- **Entity-level F1**: TBD  
- **Exact Match Accuracy**: TBD
- **Processing Speed**: TBD records/second

### Scalability
- **Memory Usage**: < 4GB for 2M records
- **Processing Time**: ~30 minutes for full dataset
- **Model Size**: ~500MB (BERT-based)

---

## 🔮 Future Improvements

- **Transformer-based contextual embeddings** (GPT, RoBERTa)
- **Active learning** for ambiguous tags
- **Automated data quality scoring**
- **Pipeline orchestration** (Airflow/Prefect)
- **Distributed processing** (Spark/Dask)
- **Real-time inference API**
- **Multi-language support**

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Hugging Face Transformers** for pre-trained models
- **seqeval** for entity-level evaluation metrics
- **pandas** for efficient data processing
- **loguru** for structured logging

---

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the documentation in `docs/`
- Review the examples in `notebooks/`

---

*Last updated: March 2025*  

