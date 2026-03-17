"""
NER metrics calculation module

Computes token-level and entity-level metrics for NER models
including precision, recall, F1-score, and accuracy.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score
from loguru import logger


class NERMetricCalculator:
    """
    Calculates comprehensive NER metrics including token-level and entity-level scores.
    """
    
    def __init__(self, entity_types: List[str]):
        """
        Initialize metric calculator.
        
        Args:
            entity_types: List of entity types for evaluation
        """
        self.entity_types = entity_types
        self.logger = logger.bind(component="NERMetricCalculator")
    
    def calculate_token_level_metrics(
        self, 
        y_true: List[List[str]], 
        y_pred: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Calculate token-level classification metrics.
        
        Args:
            y_true: List of true tag sequences
            y_pred: List of predicted tag sequences
            
        Returns:
            Dictionary with token-level metrics
        """
        # Flatten sequences for sklearn metrics
        y_true_flat = [tag for sequence in y_true for tag in sequence]
        y_pred_flat = [tag for sequence in y_pred for tag in sequence]
        
        # Get unique labels
        labels = sorted(list(set(y_true_flat + y_pred_flat)))
        
        # Calculate classification report
        try:
            report = classification_report(
                y_true_flat, 
                y_pred_flat, 
                labels=labels,
                output_dict=True,
                zero_division=0
            )
            
            # Extract overall metrics
            metrics = {
                'token_precision': report['weighted avg']['precision'],
                'token_recall': report['weighted avg']['recall'],
                'token_f1': report['weighted avg']['f1-score'],
                'token_accuracy': report['accuracy'],
                'classification_report': report,
                'labels': labels
            }
            
            # Calculate per-entity metrics
            entity_metrics = {}
            for entity_type in self.entity_types:
                b_tag = f'B-{entity_type}'
                i_tag = f'I-{entity_type}'
                
                for tag in [b_tag, i_tag]:
                    if tag in report:
                        if entity_type not in entity_metrics:
                            entity_metrics[entity_type] = {}
                        entity_metrics[entity_type][tag] = {
                            'precision': report[tag]['precision'],
                            'recall': report[tag]['recall'],
                            'f1-score': report[tag]['f1-score'],
                            'support': report[tag]['support']
                        }
            
            metrics['per_entity'] = entity_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating token-level metrics: {e}")
            metrics = {'error': str(e)}
        
        return metrics
    
    def calculate_entity_level_metrics(
        self, 
        y_true: List[List[str]], 
        y_pred: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Calculate entity-level metrics using seqeval.
        
        Args:
            y_true: List of true tag sequences
            y_pred: List of predicted tag sequences
            
        Returns:
            Dictionary with entity-level metrics
        """
        try:
            # Calculate seqeval metrics
            precision = seq_precision_score(y_true, y_pred)
            recall = seq_recall_score(y_true, y_pred)
            f1 = seq_f1_score(y_true, y_pred)
            
            # Get detailed classification report
            report = seq_classification_report(y_true, y_pred, output_dict=True)
            
            metrics = {
                'entity_precision': precision,
                'entity_recall': recall,
                'entity_f1': f1,
                'entity_classification_report': report
            }
            
            # Extract per-entity type metrics
            entity_metrics = {}
            for entity_type in self.entity_types:
                if entity_type in report:
                    entity_metrics[entity_type] = {
                        'precision': report[entity_type]['precision'],
                        'recall': report[entity_type]['recall'],
                        'f1-score': report[entity_type]['f1-score'],
                        'support': report[entity_type]['support']
                    }
            
            metrics['per_entity_entity_level'] = entity_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating entity-level metrics: {e}")
            metrics = {'error': str(e)}
        
        return metrics
    
    def calculate_exact_match_metrics(
        self, 
        true_entities: List[Dict[str, List[str]]], 
        pred_entities: List[Dict[str, List[str]]]
    ) -> Dict[str, Any]:
        """
        Calculate exact match metrics for entity reconstruction.
        
        Args:
            true_entities: List of true entity dictionaries
            pred_entities: List of predicted entity dictionaries
            
        Returns:
            Dictionary with exact match metrics
        """
        if len(true_entities) != len(pred_entities):
            raise ValueError("Length of true and predicted entities must match")
        
        exact_matches = 0
        partial_matches = 0
        total_examples = len(true_entities)
        
        entity_type_matches = defaultdict(lambda: {'exact': 0, 'partial': 0, 'total': 0})
        
        for i, (true_dict, pred_dict) in enumerate(zip(true_entities, pred_entities)):
            example_exact_match = True
            example_partial_match = False
            
            for entity_type in self.entity_types:
                true_entities_list = set(true_dict.get(entity_type, []))
                pred_entities_list = set(pred_dict.get(entity_type, []))
                
                entity_type_matches[entity_type]['total'] += 1
                
                # Exact match for this entity type
                if true_entities_list == pred_entities_list:
                    entity_type_matches[entity_type]['exact'] += 1
                # Partial match (some overlap)
                elif true_entities_list & pred_entities_list:
                    entity_type_matches[entity_type]['partial'] += 1
                    example_partial_match = True
                else:
                    example_exact_match = False
            
            if example_exact_match:
                exact_matches += 1
            if example_partial_match:
                partial_matches += 1
        
        # Calculate metrics
        exact_match_accuracy = exact_matches / total_examples
        partial_match_accuracy = partial_matches / total_examples
        
        # Per-entity type metrics
        per_entity_metrics = {}
        for entity_type in self.entity_types:
            total = entity_type_matches[entity_type]['total']
            if total > 0:
                per_entity_metrics[entity_type] = {
                    'exact_match_rate': entity_type_matches[entity_type]['exact'] / total,
                    'partial_match_rate': entity_type_matches[entity_type]['partial'] / total,
                    'total_examples': total
                }
        
        metrics = {
            'exact_match_accuracy': exact_match_accuracy,
            'partial_match_accuracy': partial_match_accuracy,
            'total_examples': total_examples,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'per_entity_exact_match': per_entity_metrics
        }
        
        return metrics
    
    def calculate_confusion_matrix(
        self, 
        y_true: List[List[str]], 
        y_pred: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Calculate confusion matrix for token-level predictions.
        
        Args:
            y_true: List of true tag sequences
            y_pred: List of predicted tag sequences
            
        Returns:
            Dictionary with confusion matrix data
        """
        # Flatten sequences
        y_true_flat = [tag for sequence in y_true for tag in sequence]
        y_pred_flat = [tag for sequence in y_pred for tag in sequence]
        
        # Get unique labels
        labels = sorted(list(set(y_true_flat + y_pred_flat)))
        
        try:
            cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
            
            # Calculate per-class metrics
            per_class_metrics = {}
            for i, label in enumerate(labels):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                per_class_metrics[label] = {
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_negatives': int(tn),
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            
            metrics = {
                'confusion_matrix': cm.tolist(),
                'labels': labels,
                'per_class_metrics': per_class_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating confusion matrix: {e}")
            metrics = {'error': str(e)}
        
        return metrics
    
    def calculate_comprehensive_metrics(
        self, 
        y_true: List[List[str]], 
        y_pred: List[List[str]],
        true_entities: Optional[List[Dict[str, List[str]]]] = None,
        pred_entities: Optional[List[Dict[str, List[str]]]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics combining all evaluation approaches.
        
        Args:
            y_true: List of true tag sequences
            y_pred: List of predicted tag sequences
            true_entities: Optional true entity dictionaries
            pred_entities: Optional predicted entity dictionaries
            
        Returns:
            Dictionary with comprehensive metrics
        """
        metrics = {}
        
        # Token-level metrics
        metrics['token_level'] = self.calculate_token_level_metrics(y_true, y_pred)
        
        # Entity-level metrics
        metrics['entity_level'] = self.calculate_entity_level_metrics(y_true, y_pred)
        
        # Confusion matrix
        metrics['confusion_matrix'] = self.calculate_confusion_matrix(y_true, y_pred)
        
        # Exact match metrics (if entities provided)
        if true_entities is not None and pred_entities is not None:
            metrics['exact_match'] = self.calculate_exact_match_metrics(true_entities, pred_entities)
        
        # Summary metrics
        metrics['summary'] = self._create_summary_metrics(metrics)
        
        return metrics
    
    def _create_summary_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create summary of key metrics.
        
        Args:
            metrics: Comprehensive metrics dictionary
            
        Returns:
            Summary metrics dictionary
        """
        summary = {}
        
        # Token-level summary
        if 'token_level' in metrics and 'error' not in metrics['token_level']:
            summary['token_f1'] = metrics['token_level']['token_f1']
            summary['token_precision'] = metrics['token_level']['token_precision']
            summary['token_recall'] = metrics['token_level']['token_recall']
        
        # Entity-level summary
        if 'entity_level' in metrics and 'error' not in metrics['entity_level']:
            summary['entity_f1'] = metrics['entity_level']['entity_f1']
            summary['entity_precision'] = metrics['entity_level']['entity_precision']
            summary['entity_recall'] = metrics['entity_level']['entity_recall']
        
        # Exact match summary
        if 'exact_match' in metrics:
            summary['exact_match_accuracy'] = metrics['exact_match']['exact_match_accuracy']
        
        return summary
    
    def generate_evaluation_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            metrics: Comprehensive metrics dictionary
            
        Returns:
            Formatted evaluation report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("NER MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        
        # Summary section
        if 'summary' in metrics:
            report_lines.append("\nSUMMARY METRICS:")
            report_lines.append("-" * 20)
            summary = metrics['summary']
            
            for metric_name, value in summary.items():
                report_lines.append(f"{metric_name:25}: {value:.4f}")
        
        # Token-level details
        if 'token_level' in metrics and 'error' not in metrics['token_level']:
            report_lines.append("\nTOKEN-LEVEL METRICS:")
            report_lines.append("-" * 25)
            token_metrics = metrics['token_level']
            
            report_lines.append(f"{'Precision':<15}: {token_metrics['token_precision']:.4f}")
            report_lines.append(f"{'Recall':<15}: {token_metrics['token_recall']:.4f}")
            report_lines.append(f"{'F1-Score':<15}: {token_metrics['token_f1']:.4f}")
            report_lines.append(f"{'Accuracy':<15}: {token_metrics['token_accuracy']:.4f}")
        
        # Entity-level details
        if 'entity_level' in metrics and 'error' not in metrics['entity_level']:
            report_lines.append("\nENTITY-LEVEL METRICS:")
            report_lines.append("-" * 25)
            entity_metrics = metrics['entity_level']
            
            report_lines.append(f"{'Precision':<15}: {entity_metrics['entity_precision']:.4f}")
            report_lines.append(f"{'Recall':<15}: {entity_metrics['entity_recall']:.4f}")
            report_lines.append(f"{'F1-Score':<15}: {entity_metrics['entity_f1']:.4f}")
        
        # Exact match details
        if 'exact_match' in metrics:
            report_lines.append("\nEXACT MATCH METRICS:")
            report_lines.append("-" * 25)
            exact_metrics = metrics['exact_match']
            
            report_lines.append(f"{'Exact Match':<15}: {exact_metrics['exact_match_accuracy']:.4f}")
            report_lines.append(f"{'Partial Match':<15}: {exact_metrics['partial_match_accuracy']:.4f}")
            report_lines.append(f"{'Total Examples':<15}: {exact_metrics['total_examples']}")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)
