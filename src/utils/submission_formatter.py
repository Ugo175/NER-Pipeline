"""
Submission formatter for NER Pipeline competition

Handles formatting of predictions according to strict submission requirements:
- TAB-separated format
- No CSV quoting
- Proper handling of empty fields
- UTF-8 encoding
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from loguru import logger


class SubmissionFormatter:
    """
    Formats NER predictions for competition submission.
    
    Ensures compliance with strict formatting requirements:
    - TAB-separated values
    - No CSV quoting in output
    - Proper UTF-8 encoding
    - Correct column ordering
    """
    
    # Required columns in specific order
    REQUIRED_COLUMNS = [
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
    ]
    
    def __init__(self, validate_format: bool = True):
        """
        Initialize submission formatter.
        
        Args:
            validate_format: Whether to validate format before submission
        """
        self.validate_format = validate_format
        self.logger = logger.bind(component="SubmissionFormatter")
    
    def format_predictions(
        self, 
        predictions: List[Dict[str, Any]],
        record_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Format predictions into submission DataFrame.
        
        Args:
            predictions: List of prediction dictionaries
            record_ids: Optional list of record IDs
            
        Returns:
            Formatted DataFrame ready for submission
        """
        if record_ids is not None and len(record_ids) != len(predictions):
            raise ValueError("Length of record_ids must match predictions")
        
        # Initialize submission DataFrame
        submission_data = []
        
        for i, prediction in enumerate(predictions):
            row = {}
            
            # Add ID
            if record_ids is not None:
                row['id'] = str(record_ids[i])
            else:
                row['id'] = str(i)
            
            # Add entity columns
            for entity_type in self.REQUIRED_COLUMNS[1:]:  # Skip 'id'
                entities = prediction.get(entity_type, [])
                if isinstance(entities, list):
                    if entities:
                        # Join multiple entities with pipe
                        row[entity_type] = '|'.join(str(e) for e in entities)
                    else:
                        row[entity_type] = ''
                else:
                    # Single entity or string
                    row[entity_type] = str(entities) if entities else ''
            
            submission_data.append(row)
        
        # Create DataFrame with correct column order
        submission_df = pd.DataFrame(submission_data, columns=self.REQUIRED_COLUMNS)
        
        if self.validate_format:
            self._validate_submission_format(submission_df)
        
        return submission_df
    
    def save_submission(
        self, 
        submission_df: pd.DataFrame, 
        output_path: Path,
        include_header: bool = True
    ) -> None:
        """
        Save submission DataFrame to file with proper formatting.
        
        Args:
            submission_df: Formatted submission DataFrame
            output_path: Path to save submission file
            include_header: Whether to include header row
        """
        if self.validate_format:
            self._validate_submission_format(submission_df)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Write with TAB separation, no quoting, UTF-8 encoding
            submission_df.to_csv(
                output_path,
                sep='\t',
                index=False,
                header=include_header,
                quoting=0,  # QUOTE_MINIMAL - no quoting unless needed
                quotechar='"',
                doublequote=False,
                escapechar=None,
                encoding='utf-8'
            )
            
            self.logger.info(f"Submission saved to {output_path}")
            
            # Verify the file was written correctly
            self._verify_output_file(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving submission: {e}")
            raise
    
    def _validate_submission_format(self, df: pd.DataFrame) -> None:
        """
        Validate submission DataFrame format.
        
        Args:
            df: DataFrame to validate
        """
        errors = []
        
        # Check columns
        if list(df.columns) != self.REQUIRED_COLUMNS:
            errors.append(f"Column mismatch. Expected: {self.REQUIRED_COLUMNS}, Got: {list(df.columns)}")
        
        # Check for prohibited characters
        for col in df.columns:
            if col in df.columns:
                # Check for TAB characters in data
                tab_count = df[col].astype(str).str.contains('\t').sum()
                if tab_count > 0:
                    errors.append(f"Column '{col}' contains {tab_count} TAB characters")
                
                # Check for newline characters
                newline_count = df[col].astype(str).str.contains(r'[\n\r]').sum()
                if newline_count > 0:
                    errors.append(f"Column '{col}' contains {newline_count} newline characters")
        
        # Check data types
        for col in df.columns:
            if col in df.columns:
                non_string_count = df.apply(lambda row: not isinstance(row[col], str), axis=1).sum()
                if non_string_count > 0:
                    errors.append(f"Column '{col}' has {non_string_count} non-string values")
        
        if errors:
            error_msg = "Submission format validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info("Submission format validation passed")
    
    def _verify_output_file(self, file_path: Path) -> None:
        """
        Verify that output file was written correctly.
        
        Args:
            file_path: Path to verify
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Output file not created: {file_path}")
        
        try:
            # Read back and verify structure
            test_df = pd.read_csv(
                file_path,
                sep='\t',
                encoding='utf-8',
                dtype=str  # Preserve as strings
            )
            
            if list(test_df.columns) != self.REQUIRED_COLUMNS:
                raise ValueError(f"File columns don't match expected: {list(test_df.columns)}")
            
            self.logger.info(f"Output file verification passed: {len(test_df)} records")
            
        except Exception as e:
            self.logger.error(f"Output file verification failed: {e}")
            raise
    
    def create_sample_submission(self, output_path: Path, num_records: int = 5) -> None:
        """
        Create a sample submission file for testing.
        
        Args:
            output_path: Path to save sample submission
            num_records: Number of sample records to create
        """
        sample_data = []
        
        for i in range(num_records):
            row = {'id': str(i)}
            
            # Add sample entities
            for entity_type in self.REQUIRED_COLUMNS[1:]:
                if i % 3 == 0:
                    row[entity_type] = f"Sample{entity_type}_{i}"
                elif i % 3 == 1:
                    row[entity_type] = f"Sample{entity_type}_{i}|Sample{entity_type}_{i+1}"
                else:
                    row[entity_type] = ''
            
            sample_data.append(row)
        
        sample_df = pd.DataFrame(sample_data, columns=self.REQUIRED_COLUMNS)
        self.save_submission(sample_df, output_path)
        
        self.logger.info(f"Sample submission created: {output_path}")
    
    def get_submission_stats(self, submission_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the submission.
        
        Args:
            submission_df: Submission DataFrame
            
        Returns:
            Dictionary with submission statistics
        """
        stats = {
            'total_records': len(submission_df),
            'columns': list(submission_df.columns),
            'file_size_estimate_mb': len(submission_df.to_csv(sep='\t', index=False).encode('utf-8')) / (1024 * 1024)
        }
        
        # Entity statistics
        entity_stats = {}
        for entity_type in self.REQUIRED_COLUMNS[1:]:
            non_empty = submission_df[entity_type].astype(str).str.strip() != ''
            entity_stats[entity_type] = {
                'non_empty_count': non_empty.sum(),
                'empty_count': (~non_empty).sum(),
                'has_multiple_entities': submission_df[entity_type].astype(str).str.contains('\|').sum()
            }
        
        stats['entity_statistics'] = entity_stats
        
        return stats
    
    def merge_predictions_with_metadata(
        self, 
        predictions: List[Dict[str, Any]],
        metadata: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge predictions with original metadata if available.
        
        Args:
            predictions: List of prediction dictionaries
            metadata: Optional DataFrame with original data
            
        Returns:
            Merged DataFrame
        """
        submission_df = self.format_predictions(predictions)
        
        if metadata is not None:
            # Try to merge on common columns
            common_cols = set(submission_df.columns) & set(metadata.columns)
            if common_cols:
                self.logger.info(f"Merging on common columns: {common_cols}")
                # This is a simplified merge - adjust based on your needs
                pass
        
        return submission_df
