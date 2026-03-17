"""
Data validation module for NER Pipeline

Provides validation and integrity checks for data processing,
entity reconstruction, and submission formatting.
"""

import pandas as pd
from typing import List, Dict, Any, Set, Optional
import re
from loguru import logger


class DataValidator:
    """
    Validator for NER pipeline data and outputs.
    
    Ensures data integrity and compliance with submission requirements.
    """
    
    # Valid entity types for German automotive e-commerce data
    VALID_ENTITY_TYPES = {
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
    }
    
    def __init__(self):
        """Initialize the data validator."""
        self.logger = logger.bind(component="DataValidator")
        self.validation_errors = []
    
    def validate_input_schema(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate input DataFrame schema.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if schema is valid
        """
        errors = []
        
        # Check for required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for empty DataFrame
        if df.empty:
            errors.append("DataFrame is empty")
        
        # Check for null values in critical columns
        for col in required_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    errors.append(f"Column '{col}' has {null_count} null values")
        
        if errors:
            self.validation_errors.extend(errors)
            self.logger.error(f"Schema validation failed: {errors}")
            return False
        
        self.logger.info("Schema validation passed")
        return True
    
    def validate_token_tag_alignment(self, tokens: List[str], tags: List[str]) -> bool:
        """
        Validate that tokens and tags are properly aligned.
        
        Args:
            tokens: List of tokens
            tags: List of tags
            
        Returns:
            True if properly aligned
        """
        if len(tokens) != len(tags):
            error = f"Token count ({len(tokens)}) != tag count ({len(tags)})"
            self.validation_errors.append(error)
            self.logger.error(error)
            return False
        
        # Validate BIO tag format
        bio_pattern = re.compile(r'^(O|B-[A-Z_]+|I-[A-Z_]+)?$')
        
        for i, tag in enumerate(tags):
            if pd.isna(tag) or tag == '':
                continue  # Empty tags are valid (continuation)
            
            if not bio_pattern.match(str(tag)):
                error = f"Invalid BIO tag at position {i}: {tag}"
                self.validation_errors.append(error)
                self.logger.warning(error)
                return False
        
        return True
    
    def validate_entity_reconstruction(
        self, 
        entities: Dict[str, List[str]],
        original_tokens: List[str]
    ) -> bool:
        """
        Validate reconstructed entities against original tokens.
        
        Args:
            entities: Dictionary of reconstructed entities
            original_tokens: Original token list
            
        Returns:
            True if reconstruction is valid
        """
        errors = []
        
        # Check entity types
        invalid_types = set(entities.keys()) - self.VALID_ENTITY_TYPES
        if invalid_types:
            errors.append(f"Invalid entity types: {invalid_types}")
        
        # Check that all reconstructed entities use valid tokens
        all_reconstructed_tokens = []
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_tokens = entity.split(' ')
                all_reconstructed_tokens.extend(entity_tokens)
        
        # Check for tokens that might have been lost (considering duplicates)
        original_token_counts = {}
        for token in original_tokens:
            original_token_counts[token] = original_token_counts.get(token, 0) + 1
        
        reconstructed_token_counts = {}
        for token in all_reconstructed_tokens:
            reconstructed_token_counts[token] = reconstructed_token_counts.get(token, 0) + 1
        
        # Check for missing tokens (allowing for tokens not tagged)
        for token, count in original_token_counts.items():
            reconstructed_count = reconstructed_token_counts.get(token, 0)
            if reconstructed_count > count:
                errors.append(f"Too many instances of token '{token}': {reconstructed_count} > {count}")
        
        if errors:
            self.validation_errors.extend(errors)
            self.logger.error(f"Entity reconstruction validation failed: {errors}")
            return False
        
        self.logger.info("Entity reconstruction validation passed")
        return True
    
    def validate_submission_format(
        self, 
        submission_df: pd.DataFrame,
        expected_columns: List[str]
    ) -> bool:
        """
        Validate submission format according to competition requirements.
        
        Args:
            submission_df: DataFrame to validate
            expected_columns: Expected column names
            
        Returns:
            True if format is valid
        """
        errors = []
        
        # Check columns
        if list(submission_df.columns) != expected_columns:
            errors.append(f"Column mismatch. Expected: {expected_columns}, Got: {list(submission_df.columns)}")
        
        # Check for proper formatting (no CSV quoting, TAB separation)
        for col in expected_columns:
            if col not in submission_df.columns:
                continue
            
            # Check for embedded TAB characters
            tab_count = submission_df[col].astype(str).str.contains('\t').sum()
            if tab_count > 0:
                errors.append(f"Column '{col}' contains {tab_count} embedded TAB characters")
            
            # Check for newline characters
            newline_count = submission_df[col].astype(str).str.contains(r'[\n\r]').sum()
            if newline_count > 0:
                errors.append(f"Column '{col}' contains {newline_count} newline characters")
        
        # Check data types (should be strings)
        for col in expected_columns:
            if col in submission_df.columns:
                # Check if any values are not strings
                non_string_count = submission_df[col].apply(lambda x: not isinstance(x, str)).sum()
                if non_string_count > 0:
                    errors.append(f"Column '{col}' has {non_string_count} non-string values")
        
        if errors:
            self.validation_errors.extend(errors)
            self.logger.error(f"Submission format validation failed: {errors}")
            return False
        
        self.logger.info("Submission format validation passed")
        return True
    
    def validate_bio_sequence(self, tags: List[str]) -> bool:
        """
        Validate BIO tag sequence for consistency.
        
        Args:
            tags: List of BIO tags
            
        Returns:
            True if sequence is valid
        """
        errors = []
        
        prev_tag_type = None
        
        for i, tag in enumerate(tags):
            if pd.isna(tag) or tag == '' or tag == 'O':
                prev_tag_type = None
                continue
            
            if tag.startswith('I-'):
                current_tag_type = tag[2:]
                
                # I-tag should not follow a different tag type
                if prev_tag_type is not None and current_tag_type != prev_tag_type:
                    errors.append(f"Invalid I-tag at position {i}: {tag} follows different tag type")
                
                prev_tag_type = current_tag_type
            elif tag.startswith('B-'):
                prev_tag_type = tag[2:]
        
        if errors:
            self.validation_errors.extend(errors)
            self.logger.error(f"BIO sequence validation failed: {errors}")
            return False
        
        return True
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'null_counts': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Check for common data issues
        issues = []
        
        # Check for empty strings
        for col in df.columns:
            empty_count = (df[col] == '').sum()
            if empty_count > 0:
                issues.append(f"Column '{col}' has {empty_count} empty strings")
        
        # Check for extremely long strings (potential data issues)
        for col in df.select_dtypes(include=['object']).columns:
            max_length = df[col].astype(str).str.len().max()
            if max_length > 1000:  # Arbitrary threshold
                issues.append(f"Column '{col}' has very long strings (max: {max_length})")
        
        quality_report['issues'] = issues
        
        if issues:
            self.logger.warning(f"Data quality issues found: {issues}")
        else:
            self.logger.info("Data quality check passed")
        
        return quality_report
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation errors.
        
        Returns:
            Dictionary with validation summary
        """
        return {
            'total_errors': len(self.validation_errors),
            'errors': self.validation_errors.copy(),
            'is_valid': len(self.validation_errors) == 0
        }
    
    def clear_errors(self):
        """Clear all validation errors."""
        self.validation_errors.clear()
