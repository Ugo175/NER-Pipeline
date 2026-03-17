"""
Data processing module for NER Pipeline

Handles ingestion, preprocessing, and validation of e-commerce data
with support for large-scale compressed datasets.
"""

from .ingestion import DataIngestor
from .preprocessor import NERPreprocessor
from .validator import DataValidator

__all__ = ['DataIngestor', 'NERPreprocessor', 'DataValidator']
