"""
Results analyzer for NER Pipeline

Provides detailed analysis and visualization of NER model performance
including error analysis, per-entity analysis, and comparative studies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import json
from collections import defaultdict, Counter
from loguru import logger


class ResultsAnalyzer:
    """
    Analyzes and visualizes NER model results.
    
    Provides detailed analysis of model performance, error patterns,
    and per-entity type performance.
    """
    
    def __init__(self, entity_types: List[str]):
        """
        Initialize results analyzer.
        
        Args:
            entity_types: List of entity types for analysis
        """
        self.entity_types = entity_types
        self.logger = logger.bind(component="ResultsAnalyzer")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def analyze_per_entity_performance(
        self, 
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze performance per entity type.
        
        Args:
            metrics: Comprehensive metrics dictionary
            
        Returns:
            Dictionary with per-entity analysis
        """
        entity_analysis = {}
        
        # Extract token-level per-entity metrics
        if 'token_level' in metrics and 'per_entity' in metrics['token_level']:
            token_per_entity = metrics['token_level']['per_entity']
            for entity_type in self.entity_types:
                entity_analysis[entity_type] = {
                    'token_metrics': {}
                }
                
                if entity_type in token_per_entity:
                    for tag, tag_metrics in token_per_entity[entity_type].items():
                        entity_analysis[entity_type]['token_metrics'][tag] = {
                            'precision': tag_metrics['precision'],
                            'recall': tag_metrics['recall'],
                            'f1-score': tag_metrics['f1-score'],
                            'support': tag_metrics['support']
                        }
        
        # Extract entity-level per-entity metrics
        if 'entity_level' in metrics and 'per_entity_entity_level' in metrics['entity_level']:
            entity_per_entity = metrics['entity_level']['per_entity_entity_level']
            for entity_type in self.entity_types:
                if entity_type not in entity_analysis:
                    entity_analysis[entity_type] = {}
                
                if entity_type in entity_per_entity:
                    entity_analysis[entity_type]['entity_metrics'] = entity_per_entity[entity_type]
        
        # Extract exact match per-entity metrics
        if 'exact_match' in metrics and 'per_entity_exact_match' in metrics['exact_match']:
            exact_per_entity = metrics['exact_match']['per_entity_exact_match']
            for entity_type in self.entity_types:
                if entity_type not in entity_analysis:
                    entity_analysis[entity_type] = {}
                
                if entity_type in exact_per_entity:
                    entity_analysis[entity_type]['exact_match_metrics'] = exact_per_entity[entity_type]
        
        return entity_analysis
    
    def analyze_error_patterns(
        self, 
        error_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze error patterns and common mistakes.
        
        Args:
            error_analysis: Error analysis dictionary from evaluator
            
        Returns:
            Dictionary with detailed error pattern analysis
        """
        patterns = {
            'error_distribution': {},
            'top_confusion_pairs': {},
            'error_context_analysis': {},
            'entity_specific_errors': defaultdict(lambda: defaultdict(int))
        }
        
        # Error type distribution
        total_errors = error_analysis.get('total_errors', 0)
        if total_errors > 0:
            for error_type, count in error_analysis.get('error_types', {}).items():
                patterns['error_distribution'][error_type] = {
                    'count': count,
                    'percentage': (count / total_errors) * 100
                }
        
        # Top confusion pairs
        confusion_pairs = error_analysis.get('confusion_pairs', {})
        top_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        patterns['top_confusion_pairs'] = {
            pair: {'count': count, 'percentage': (count / total_errors) * 100}
            for pair, count in top_pairs
        } if total_errors > 0 else {}
        
        # Entity-specific errors
        for example in error_analysis.get('error_examples', []):
            for error in example.get('errors', []):
                true_tag = error['true_tag']
                pred_tag = error['pred_tag']
                
                # Extract entity type from tags
                true_entity = true_tag.split('-')[-1] if '-' in true_tag else 'O'
                pred_entity = pred_tag.split('-')[-1] if '-' in pred_tag else 'O'
                
                patterns['entity_specific_errors'][true_entity][pred_entity] += 1
        
        return dict(patterns)
    
    def create_performance_plots(
        self, 
        metrics: Dict[str, Any], 
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Create performance visualization plots.
        
        Args:
            metrics: Comprehensive metrics dictionary
            output_dir: Directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {}
        
        # 1. Overall metrics bar plot
        if 'summary' in metrics:
            plot_paths['overall_metrics'] = self._plot_overall_metrics(
                metrics['summary'], output_dir
            )
        
        # 2. Per-entity performance heatmap
        if 'token_level' in metrics and 'per_entity' in metrics['token_level']:
            plot_paths['entity_performance'] = self._plot_entity_performance(
                metrics['token_level']['per_entity'], output_dir
            )
        
        # 3. Confusion matrix heatmap
        if 'confusion_matrix' in metrics:
            plot_paths['confusion_matrix'] = self._plot_confusion_matrix(
                metrics['confusion_matrix'], output_dir
            )
        
        # 4. Error distribution pie chart
        if 'error_analysis' in metrics:
            plot_paths['error_distribution'] = self._plot_error_distribution(
                metrics['error_analysis'], output_dir
            )
        
        return plot_paths
    
    def _plot_overall_metrics(self, summary_metrics: Dict[str, float], output_dir: Path) -> Path:
        """Create overall metrics bar plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_names = list(summary_metrics.keys())
        metrics_values = list(summary_metrics.values())
        
        bars = ax.bar(metrics_names, metrics_values)
        ax.set_title('Overall Model Performance Metrics', fontsize=16, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = output_dir / 'overall_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_entity_performance(self, per_entity_metrics: Dict, output_dir: Path) -> Path:
        """Create per-entity performance heatmap."""
        # Prepare data for heatmap
        entity_f1_scores = {}
        
        for entity_type in self.entity_types:
            if entity_type in per_entity_metrics:
                # Calculate average F1 for this entity type
                f1_scores = []
                for tag, metrics in per_entity_metrics[entity_type].items():
                    if 'f1-score' in metrics:
                        f1_scores.append(metrics['f1-score'])
                
                if f1_scores:
                    entity_f1_scores[entity_type] = np.mean(f1_scores)
                else:
                    entity_f1_scores[entity_type] = 0.0
            else:
                entity_f1_scores[entity_type] = 0.0
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        entities = list(entity_f1_scores.keys())
        scores = list(entity_f1_scores.values())
        
        # Create a single-row heatmap
        heatmap_data = np.array(scores).reshape(1, -1)
        
        sns.heatmap(heatmap_data, 
                   xticklabels=entities, 
                   yticklabels=['F1-Score'],
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   vmin=0, 
                   vmax=1,
                   ax=ax)
        
        ax.set_title('Per-Entity Type Performance (F1-Score)', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = output_dir / 'entity_performance_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_confusion_matrix(self, confusion_data: Dict, output_dir: Path) -> Path:
        """Create confusion matrix heatmap."""
        if 'confusion_matrix' not in confusion_data:
            return None
        
        cm = np.array(confusion_data['confusion_matrix'])
        labels = confusion_data['labels']
        
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Create heatmap
        sns.heatmap(cm, 
                   xticklabels=labels, 
                   yticklabels=labels,
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   ax=ax)
        
        ax.set_title('Token-Level Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_error_distribution(self, error_analysis: Dict, output_dir: Path) -> Path:
        """Create error distribution pie chart."""
        error_types = error_analysis.get('error_types', {})
        
        if not error_types:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = list(error_types.keys())
        sizes = list(error_types.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(sizes, 
                                          labels=labels, 
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          startangle=90)
        
        ax.set_title('Error Type Distribution', fontsize=16, fontweight='bold')
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        output_path = output_dir / 'error_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_comprehensive_report(
        self, 
        metrics: Dict[str, Any], 
        output_dir: Path
    ) -> Path:
        """
        Generate comprehensive HTML report.
        
        Args:
            metrics: Comprehensive metrics dictionary
            output_dir: Directory to save report
            
        Returns:
            Path to generated report
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots
        plot_paths = self.create_performance_plots(metrics, output_dir)
        
        # Analyze results
        entity_analysis = self.analyze_per_entity_performance(metrics)
        
        # Generate HTML report
        html_content = self._generate_html_report(
            metrics, entity_analysis, plot_paths, output_dir
        )
        
        report_path = output_dir / 'comprehensive_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Comprehensive report generated: {report_path}")
        return report_path
    
    def _generate_html_report(
        self, 
        metrics: Dict, 
        entity_analysis: Dict, 
        plot_paths: Dict[str, Path], 
        output_dir: Path
    ) -> str:
        """Generate HTML content for comprehensive report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>NER Model Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .plot-container { text-align: center; margin: 20px 0; }
                .plot-container img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>NER Model Evaluation Report</h1>
        """
        
        # Summary metrics
        if 'summary' in metrics:
            html += "<h2>Summary Metrics</h2>"
            html += "<div class='metric-card'>"
            html += "<table>"
            html += "<tr><th>Metric</th><th>Value</th></tr>"
            for metric, value in metrics['summary'].items():
                html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
            html += "</table>"
            html += "</div>"
        
        # Plots
        if plot_paths:
            html += "<h2>Performance Visualizations</h2>"
            for plot_name, plot_path in plot_paths.items():
                if plot_path:
                    relative_path = plot_path.name
                    html += f"<div class='plot-container'>"
                    html += f"<h3>{plot_name.replace('_', ' ').title()}</h3>"
                    html += f"<img src='{relative_path}' alt='{plot_name}'>"
                    html += "</div>"
        
        # Entity analysis
        if entity_analysis:
            html += "<h2>Per-Entity Performance</h2>"
            for entity_type, analysis in entity_analysis.items():
                html += f"<h3>{entity_type}</h3>"
                html += "<div class='metric-card'>"
                
                if 'entity_metrics' in analysis:
                    html += "<h4>Entity-Level Metrics</h4>"
                    html += "<table>"
                    html += "<tr><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>"
                    em = analysis['entity_metrics']
                    html += f"<tr><td>{em.get('precision', 0):.4f}</td><td>{em.get('recall', 0):.4f}</td>"
                    html += f"<td>{em.get('f1-score', 0):.4f}</td><td>{em.get('support', 0)}</td></tr>"
                    html += "</table>"
                
                html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def save_analysis_results(
        self, 
        analysis: Dict[str, Any], 
        output_path: Path
    ) -> None:
        """
        Save analysis results to JSON file.
        
        Args:
            analysis: Analysis results dictionary
            output_path: Path to save results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare for JSON serialization
        json_analysis = self._prepare_for_json(analysis)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_analysis, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Analysis results saved to {output_path}")
    
    def _prepare_for_json(self, obj: Any) -> Any:
        """Prepare object for JSON serialization."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
