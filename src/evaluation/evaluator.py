"""
NER model evaluator

Comprehensive evaluation pipeline for NER models including data loading,
prediction generation, and metric calculation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import json
from loguru import logger

from .metrics import NERMetricCalculator
from ..data_processing.preprocessor import NERPreprocessor
from ..utils.entity_reconstructor import EntityReconstructor


class NEREvaluator:
    """
    Comprehensive evaluator for NER models.
    
    Handles evaluation of trained models on test data with comprehensive metrics.
    """
    
    def __init__(
        self, 
        entity_types: List[str],
        preprocessor: Optional[NERPreprocessor] = None,
        reconstructor: Optional[EntityReconstructor] = None
    ):
        """
        Initialize NER evaluator.
        
        Args:
            entity_types: List of entity types for evaluation
            preprocessor: Optional preprocessor for tokenization
            reconstructor: Optional reconstructor for entity reconstruction
        """
        self.entity_types = entity_types
        self.preprocessor = preprocessor or NERPreprocessor()
        self.reconstructor = reconstructor or EntityReconstructor()
        self.metric_calculator = NERMetricCalculator(entity_types)
        self.logger = logger.bind(component="NEREvaluator")
    
    def evaluate_predictions(
        self,
        true_tags: List[List[str]],
        pred_tags: List[List[str]],
        tokens: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions against ground truth.
        
        Args:
            true_tags: List of true tag sequences
            pred_tags: List of predicted tag sequences
            tokens: Optional list of token sequences for entity reconstruction
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        if len(true_tags) != len(pred_tags):
            raise ValueError("Length of true and predicted tags must match")
        
        self.logger.info(f"Evaluating {len(true_tags)} sequences")
        
        # Calculate basic metrics
        metrics = self.metric_calculator.calculate_comprehensive_metrics(
            true_tags, pred_tags
        )
        
        # Calculate entity reconstruction metrics if tokens provided
        if tokens is not None:
            if len(tokens) != len(true_tags):
                raise ValueError("Length of tokens must match tags")
            
            true_entities = []
            pred_entities = []
            
            for token_seq, true_tag_seq, pred_tag_seq in zip(tokens, true_tags, pred_tags):
                true_entity_dict = self.reconstructor.reconstruct_entities(token_seq, true_tag_seq)
                pred_entity_dict = self.reconstructor.reconstruct_entities(token_seq, pred_tag_seq)
                
                true_entities.append(true_entity_dict)
                pred_entities.append(pred_entity_dict)
            
            # Add exact match metrics
            exact_match_metrics = self.metric_calculator.calculate_exact_match_metrics(
                true_entities, pred_entities
            )
            metrics['exact_match'] = exact_match_metrics
            metrics['reconstructed_entities'] = {
                'true_entities': true_entities,
                'pred_entities': pred_entities
            }
        
        # Add evaluation metadata
        metrics['evaluation_metadata'] = {
            'num_sequences': len(true_tags),
            'entity_types': self.entity_types,
            'total_tokens': sum(len(seq) for seq in true_tags)
        }
        
        self.logger.info("Evaluation completed successfully")
        return metrics
    
    def evaluate_from_dataframe(
        self,
        df: pd.DataFrame,
        true_tag_columns: List[str],
        pred_tag_columns: List[str],
        token_column: str = 'tokens'
    ) -> Dict[str, Any]:
        """
        Evaluate predictions from DataFrame format.
        
        Args:
            df: DataFrame with true and predicted tags
            true_tag_columns: Column names for true tags
            pred_tag_columns: Column names for predicted tags
            token_column: Column name for tokens
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract data from DataFrame
        true_tags = []
        pred_tags = []
        tokens = []
        
        for _, row in df.iterrows():
            true_seq = [row[col] for col in true_tag_columns if pd.notna(row[col])]
            pred_seq = [row[col] for col in pred_tag_columns if pd.notna(row[col])]
            token_seq = row.get(token_column, [])
            
            if isinstance(token_seq, str):
                token_seq = token_seq.split(' ')
            
            true_tags.append(true_seq)
            pred_tags.append(pred_seq)
            tokens.append(token_seq)
        
        return self.evaluate_predictions(true_tags, pred_tags, tokens)
    
    def evaluate_model(
        self,
        model,
        test_data: pd.DataFrame,
        text_column: str = 'title',
        tag_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained NER model with predict method
            test_data: Test dataset
            text_column: Column name for text data
            tag_columns: Column names for true tags
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating model on {len(test_data)} test examples")
        
        # Prepare test data
        tokens_list = []
        true_tags_list = []
        
        for _, row in test_data.iterrows():
            text = row[text_column]
            tokens = self.preprocessor.tokenize_title(text)
            tokens_list.append(tokens)
            
            # Extract true tags if provided
            if tag_columns:
                true_tags = []
                for i, token in enumerate(tokens):
                    if i < len(tag_columns):
                        tag = row[tag_columns[i]] if pd.notna(row[tag_columns[i]]) else 'O'
                        true_tags.append(tag)
                    else:
                        true_tags.append('O')
                true_tags_list.append(true_tags)
        
        # Generate predictions
        pred_tags_list = []
        for tokens in tokens_list:
            # This assumes model has a predict method that takes tokens
            # Adjust based on your actual model interface
            try:
                pred_tags = model.predict(tokens)
                pred_tags_list.append(pred_tags)
            except Exception as e:
                self.logger.error(f"Error predicting for tokens {tokens}: {e}")
                pred_tags_list.append(['O'] * len(tokens))
        
        # Evaluate
        true_tags_to_use = true_tags_list if true_tags_list else None
        metrics = self.evaluate_predictions(
            true_tags_to_use, pred_tags_list, tokens_list
        )
        
        return metrics
    
    def cross_validate(
        self,
        model_class,
        data: pd.DataFrame,
        cv_folds: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model_class: Class of the model to evaluate
            data: Dataset for cross-validation
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with cross-validation results
        """
        from sklearn.model_selection import StratifiedKFold
        
        self.logger.info(f"Starting {cv_folds}-fold cross-validation")
        
        # Create stratified folds (simplified - you might want more sophisticated splitting)
        # For NER, you typically want to split by documents rather than individual tokens
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        cv_results = {
            'fold_results': [],
            'mean_metrics': {},
            'std_metrics': {},
            'cv_folds': cv_folds
        }
        
        # For simplicity, we'll use a dummy stratification based on sequence length
        # In practice, you'd want more sophisticated document-level splitting
        sequence_lengths = [len(seq.split()) for seq in data['title']]
        length_bins = pd.qcut(sequence_lengths, q=5, labels=False, duplicates='drop')
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(data, length_bins)):
            self.logger.info(f"Processing fold {fold + 1}/{cv_folds}")
            
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            
            # Train model
            model = model_class()
            model.fit(train_data)
            
            # Evaluate
            fold_metrics = self.evaluate_model(model, val_data)
            fold_metrics['fold'] = fold
            cv_results['fold_results'].append(fold_metrics)
        
        # Calculate mean and std across folds
        if cv_results['fold_results']:
            all_metrics = {}
            for fold_result in cv_results['fold_results']:
                for metric_name, value in fold_result.get('summary', {}).items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
            
            for metric_name, values in all_metrics.items():
                cv_results['mean_metrics'][metric_name] = np.mean(values)
                cv_results['std_metrics'][metric_name] = np.std(values)
        
        self.logger.info("Cross-validation completed")
        return cv_results
    
    def save_evaluation_results(
        self, 
        metrics: Dict[str, Any], 
        output_path: Path
    ) -> None:
        """
        Save evaluation results to file.
        
        Args:
            metrics: Evaluation metrics dictionary
            output_path: Path to save results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        json_metrics = self._prepare_metrics_for_json(metrics)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_metrics, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Evaluation results saved to {output_path}")
    
    def generate_evaluation_report(
        self, 
        metrics: Dict[str, Any], 
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            metrics: Evaluation metrics dictionary
            output_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        report = self.metric_calculator.generate_evaluation_report(metrics)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Evaluation report saved to {output_path}")
        
        return report
    
    def _prepare_metrics_for_json(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metrics for JSON serialization.
        
        Args:
            metrics: Raw metrics dictionary
            
        Returns:
            JSON-serializable metrics dictionary
        """
        json_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                json_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            elif isinstance(value, dict):
                json_metrics[key] = self._prepare_metrics_for_json(value)
            elif isinstance(value, list):
                json_metrics[key] = [
                    self._prepare_metrics_for_json(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                json_metrics[key] = value
        
        return json_metrics
    
    def analyze_errors(
        self, 
        true_tags: List[List[str]], 
        pred_tags: List[List[str]], 
        tokens: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors in detail.
        
        Args:
            true_tags: List of true tag sequences
            pred_tags: List of predicted tag sequences
            tokens: List of token sequences
            
        Returns:
            Dictionary with error analysis
        """
        error_analysis = {
            'total_errors': 0,
            'error_types': defaultdict(int),
            'confusion_pairs': defaultdict(int),
            'error_examples': []
        }
        
        for seq_idx, (true_seq, pred_seq, token_seq) in enumerate(zip(true_tags, pred_tags, tokens)):
            seq_errors = []
            
            for token_idx, (true_tag, pred_tag, token) in enumerate(zip(true_seq, pred_seq, token_seq)):
                if true_tag != pred_tag:
                    error_analysis['total_errors'] += 1
                    
                    # Categorize error type
                    if true_tag == 'O' and pred_tag != 'O':
                        error_type = 'false_positive'
                    elif true_tag != 'O' and pred_tag == 'O':
                        error_type = 'false_negative'
                    else:
                        error_type = 'misclassification'
                    
                    error_analysis['error_types'][error_type] += 1
                    
                    # Track confusion pairs
                    confusion_pair = f"{true_tag} -> {pred_tag}"
                    error_analysis['confusion_pairs'][confusion_pair] += 1
                    
                    # Store example (limit to prevent memory issues)
                    if len(error_analysis['error_examples']) < 100:
                        seq_errors.append({
                            'token': token,
                            'true_tag': true_tag,
                            'pred_tag': pred_tag,
                            'error_type': error_type,
                            'position': token_idx
                        })
            
            if seq_errors:
                error_analysis['error_examples'].append({
                    'sequence_id': seq_idx,
                    'tokens': token_seq,
                    'errors': seq_errors
                })
        
        return dict(error_analysis)
