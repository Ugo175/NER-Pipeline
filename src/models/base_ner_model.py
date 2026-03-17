"""
Base NER model class

Abstract base class for all NER models in the pipeline.
Provides common interface and utility methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from loguru import logger


class BaseNERModel(ABC):
    """
    Abstract base class for NER models.
    
    All NER models should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, entity_types: List[str], **kwargs):
        """
        Initialize base NER model.
        
        Args:
            entity_types: List of entity types for the model
            **kwargs: Additional model-specific parameters
        """
        self.entity_types = entity_types
        self.model_params = kwargs
        self.is_trained = False
        self.logger = logger.bind(component=self.__class__.__name__)
        
        # Generate label mappings
        self.label_to_id, self.id_to_label = self._create_label_mappings()
    
    def _create_label_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Create label to ID and ID to label mappings.
        
        Returns:
            Tuple of (label_to_id, id_to_label) dictionaries
        """
        labels = ['O']  # Outside tag
        
        for entity_type in self.entity_types:
            labels.append(f'B-{entity_type}')  # Begin
            labels.append(f'I-{entity_type}')  # Inside
        
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        
        return label_to_id, id_to_label
    
    @abstractmethod
    def train(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train the NER model.
        
        Args:
            train_data: Training dataset
            validation_data: Optional validation dataset
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[List[str]]:
        """
        Predict entity tags for given texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of predicted tag sequences
        """
        pass
    
    @abstractmethod
    def predict_tokens(self, tokens_list: List[List[str]]) -> List[List[str]]:
        """
        Predict entity tags for tokenized inputs.
        
        Args:
            tokens_list: List of token sequences
            
        Returns:
            List of predicted tag sequences
        """
        pass
    
    def save_model(self, model_path: Path) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model metadata
        metadata = {
            'entity_types': self.entity_types,
            'model_params': self.model_params,
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label,
            'model_class': self.__class__.__name__
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Model metadata saved to {metadata_path}")
    
    def load_model(self, model_path: Path) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load metadata
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.entity_types = metadata['entity_types']
            self.model_params = metadata['model_params']
            self.label_to_id = metadata['label_to_id']
            self.id_to_label = {int(k): v for k, v in metadata['id_to_label'].items()}
        
        self.is_trained = True
        self.logger.info(f"Model loaded from {model_path}")
    
    def validate_input(self, texts: List[str]) -> bool:
        """
        Validate input texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            True if input is valid
        """
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings")
        
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All items in input list must be strings")
        
        return True
    
    def preprocess_texts(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess texts into tokens.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of token sequences
        """
        from ..data_processing.preprocessor import NERPreprocessor
        
        preprocessor = NERPreprocessor()
        tokens_list = []
        
        for text in texts:
            tokens = preprocessor.tokenize_title(text)
            tokens_list.append(tokens)
        
        return tokens_list
    
    def postprocess_predictions(
        self, 
        predictions: List[List[str]], 
        tokens_list: List[List[str]]
    ) -> List[List[str]]:
        """
        Postprocess model predictions.
        
        Args:
            predictions: Raw model predictions
            tokens_list: Original token sequences
            
        Returns:
            Postprocessed predictions
        """
        # Ensure predictions match token lengths
        processed_predictions = []
        
        for pred_seq, token_seq in zip(predictions, tokens_list):
            if len(pred_seq) != len(token_seq):
                # Truncate or pad predictions to match token length
                if len(pred_seq) > len(token_seq):
                    processed_pred = pred_seq[:len(token_seq)]
                else:
                    processed_pred = pred_seq + ['O'] * (len(token_seq) - len(pred_seq))
            else:
                processed_pred = pred_seq
            
            processed_predictions.append(processed_pred)
        
        return processed_predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_class': self.__class__.__name__,
            'entity_types': self.entity_types,
            'num_labels': len(self.label_to_id),
            'is_trained': self.is_trained,
            'model_params': self.model_params
        }
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        from ..evaluation.evaluator import NEREvaluator
        
        evaluator = NEREvaluator(self.entity_types)
        metrics = evaluator.evaluate_model(self, test_data)
        
        return metrics
    
    def predict_single(self, text: str) -> List[str]:
        """
        Predict entities for a single text.
        
        Args:
            text: Input text
            
        Returns:
            List of predicted tags
        """
        predictions = self.predict([text])
        return predictions[0] if predictions else []
    
    def predict_with_confidence(
        self, 
        texts: List[str]
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Predict entities with confidence scores.
        
        Args:
            texts: List of input texts
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        # Default implementation - override in subclasses if confidence scores are available
        predictions = self.predict(texts)
        
        # Create dummy confidence scores
        confidence_scores = []
        for pred_seq in predictions:
            conf_scores = [1.0] * len(pred_seq)  # Perfect confidence as default
            confidence_scores.append(conf_scores)
        
        return predictions, confidence_scores
    
    def analyze_predictions(
        self, 
        texts: List[str], 
        predictions: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze model predictions.
        
        Args:
            texts: List of input texts
            predictions: Optional pre-computed predictions
            
        Returns:
            Dictionary with prediction analysis
        """
        if predictions is None:
            predictions = self.predict(texts)
        
        analysis = {
            'total_texts': len(texts),
            'total_tokens': sum(len(pred) for pred in predictions),
            'entity_counts': {entity_type: 0 for entity_type in self.entity_types},
            'o_tag_count': 0,
            'average_tokens_per_text': 0
        }
        
        # Count entities
        for pred_seq in predictions:
            for tag in pred_seq:
                if tag == 'O':
                    analysis['o_tag_count'] += 1
                elif tag.startswith('B-') or tag.startswith('I-'):
                    entity_type = tag[2:]
                    if entity_type in analysis['entity_counts']:
                        analysis['entity_counts'][entity_type] += 1
        
        # Calculate averages
        if len(texts) > 0:
            analysis['average_tokens_per_text'] = analysis['total_tokens'] / len(texts)
        
        return analysis
