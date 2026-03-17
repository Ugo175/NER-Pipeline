"""
Configuration management for NER Pipeline

Centralized configuration settings for data processing, model training,
and submission formatting.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Data paths
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    output_dir: Path = Path("data/output")
    
    # File formats
    input_file_extension: str = ".gz"
    output_file_extension: str = ".tsv"
    
    # Processing parameters
    chunk_size: int = 10000
    batch_size: int = 32
    max_sequence_length: int = 512
    
    # Data validation
    required_columns: List[str] = field(default_factory=lambda: [
        'title', 'Hersteller', 'Kompatible_Fahrzeug_Marke', 'Kompatibles_Fahrzeug_Modell',
        'Produktart', 'Herstellernummer', 'EAN', 'Zustand', 'Farbe', 'Material', 'Anzahl', 'OEM'
    ])
    
    # Encoding settings
    encoding: str = "utf-8"
    delimiter: str = "\t"
    quotechar: str = '"'


@dataclass
class ModelConfig:
    """Configuration for NER model."""
    
    # Model architecture
    model_type: str = "bert"  # Options: bert, roberta, distilbert, custom
    model_name: str = "bert-base-german-cased"
    
    # Training parameters
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Regularization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    
    # Evaluation
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    
    # Model paths
    model_dir: Path = Path("models")
    checkpoint_dir: Path = Path("models/checkpoints")
    best_model_dir: Path = Path("models/best")
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001


@dataclass
class EntityConfig:
    """Configuration for entity types and processing."""
    
    # Entity types for German automotive e-commerce
    entity_types: List[str] = field(default_factory=lambda: [
        'Hersteller',
        'Kompatible_Fahrzeug_Marke',
        'Kompatibles_Fahrzeug_Modell',
        'Produktart',
        'Herstellernummer',
        'EAN',
        'Zustand',
        'Farbe',
        'Material',
        'Anzahl',
        'OEM'
    ])
    
    # Entity reconstruction settings
    preserve_duplicates: bool = True
    use_ascii_space: bool = True
    empty_tag_means_continuation: bool = True
    
    # Special handling
    multi_entity_separator: str = "|"
    max_entities_per_type: int = 10
    
    # Tag scheme
    tag_scheme: str = "BIO"  # Options: BIO, BIOES


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "precision", "recall", "f1", "accuracy"
    ])
    
    # Evaluation parameters
    eval_steps: int = 500
    eval_batch_size: int = 32
    
    # Output
    output_eval_dir: Path = Path("data/eval")
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    
    # Thresholds
    confidence_threshold: float = 0.5
    entity_level_threshold: float = 0.5
    
    # Cross-validation
    num_folds: int = 5
    stratify_by_entity: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    # Log levels
    log_level: str = "INFO"
    file_log_level: str = "DEBUG"
    
    # Log paths
    log_dir: Path = Path("logs")
    log_file: str = "ner_pipeline.log"
    
    # Log rotation
    max_log_size_mb: int = 100
    backup_count: int = 5
    
    # Console logging
    console_log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    file_log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"


@dataclass
class SubmissionConfig:
    """Configuration for submission formatting."""
    
    # Output settings
    output_dir: Path = Path("submissions")
    file_prefix: str = "submission"
    
    # Format requirements
    delimiter: str = "\t"
    include_header: bool = True
    encoding: str = "utf-8"
    
    # Validation
    validate_format: bool = True
    strict_validation: bool = True
    
    # Required columns (in order)
    required_columns: List[str] = field(default_factory=lambda: [
        'id',
        'Hersteller',
        'Kompatible_Fahrzeug_Marke',
        'Kompatibles_Fahrzeug_Modell',
        'Produktart',
        'Herstellernummer',
        'EAN',
        'Zustand',
        'Farbe',
        'Material',
        'Anzahl',
        'OEM'
    ])


class Config:
    """
    Main configuration class that combines all configuration sections.
    """
    
    def __init__(
        self,
        config_file: Optional[Path] = None,
        **kwargs
    ):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to JSON configuration file
            **kwargs: Override configuration parameters
        """
        # Initialize default configurations
        self.data = DataConfig()
        self.model = ModelConfig()
        self.entities = EntityConfig()
        self.evaluation = EvaluationConfig()
        self.logging = LoggingConfig()
        self.submission = SubmissionConfig()
        
        # Load from file if provided
        if config_file and config_file.exists():
            self.load_from_file(config_file)
        
        # Apply overrides
        self._apply_overrides(kwargs)
        
        # Create directories
        self._create_directories()
    
    def load_from_file(self, config_file: Path) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Update configuration sections
            if 'data' in config_dict:
                self._update_dataclass(self.data, config_dict['data'])
            if 'model' in config_dict:
                self._update_dataclass(self.model, config_dict['model'])
            if 'entities' in config_dict:
                self._update_dataclass(self.entities, config_dict['entities'])
            if 'evaluation' in config_dict:
                self._update_dataclass(self.evaluation, config_dict['evaluation'])
            if 'logging' in config_dict:
                self._update_dataclass(self.logging, config_dict['logging'])
            if 'submission' in config_dict:
                self._update_dataclass(self.submission, config_dict['submission'])
                
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_file}: {e}")
    
    def save_to_file(self, config_file: Path) -> None:
        """
        Save current configuration to JSON file.
        
        Args:
            config_file: Path to save configuration
        """
        config_dict = {
            'data': self._dataclass_to_dict(self.data),
            'model': self._dataclass_to_dict(self.model),
            'entities': self._dataclass_to_dict(self.entities),
            'evaluation': self._dataclass_to_dict(self.evaluation),
            'logging': self._dataclass_to_dict(self.logging),
            'submission': self._dataclass_to_dict(self.submission)
        }
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Apply configuration overrides.
        
        Args:
            overrides: Dictionary of overrides
        """
        for key, value in overrides.items():
            if hasattr(self, key):
                section = getattr(self, key)
                if isinstance(value, dict) and hasattr(section, '__dict__'):
                    self._update_dataclass(section, value)
                else:
                    setattr(self, key, value)
    
    def _update_dataclass(self, dataclass_obj, updates: Dict[str, Any]) -> None:
        """
        Update dataclass with dictionary values.
        
        Args:
            dataclass_obj: Dataclass object to update
            updates: Dictionary of updates
        """
        for key, value in updates.items():
            if hasattr(dataclass_obj, key):
                # Handle Path objects
                if isinstance(getattr(dataclass_obj, key), Path):
                    setattr(dataclass_obj, key, Path(value))
                else:
                    setattr(dataclass_obj, key, value)
    
    def _dataclass_to_dict(self, dataclass_obj) -> Dict[str, Any]:
        """
        Convert dataclass to dictionary.
        
        Args:
            dataclass_obj: Dataclass object
            
        Returns:
            Dictionary representation
        """
        result = {}
        for key, value in dataclass_obj.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.data.data_dir,
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.data.output_dir,
            self.model.model_dir,
            self.model.checkpoint_dir,
            self.model.best_model_dir,
            self.evaluation.output_eval_dir,
            self.logging.log_dir,
            self.submission.output_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'data': self._dataclass_to_dict(self.data),
            'model': self._dataclass_to_dict(self.model),
            'entities': self._dataclass_to_dict(self.entities),
            'evaluation': self._dataclass_to_dict(self.evaluation),
            'logging': self._dataclass_to_dict(self.logging),
            'submission': self._dataclass_to_dict(self.submission)
        }


# Global configuration instance
config = Config()
