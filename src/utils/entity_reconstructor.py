"""
Entity reconstruction module following strict formatting rules

Handles reconstruction of entities from token-level predictions
with proper handling of continuation tags, duplicate tokens, and formatting constraints.
"""

from typing import List, Dict, Tuple, Optional
import pandas as pd
from loguru import logger


class EntityReconstructor:
    """
    Reconstructs entities from token-level predictions following strict rules.
    
    Key rules:
    - Empty tags indicate continuation
    - Same tag ≠ continuation  
    - Multi-token entities must be reconstructed using ASCII space
    - Duplicate tokens must be preserved
    - Submission formatting must follow exact constraints
    """
    
    # Valid entity types for German automotive e-commerce data
    ENTITY_TYPES = [
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
    
    def __init__(self, preserve_duplicates: bool = True):
        """
        Initialize the entity reconstructor.
        
        Args:
            preserve_duplicates: Whether to preserve duplicate tokens in output
        """
        self.preserve_duplicates = preserve_duplicates
        self.logger = logger.bind(component="EntityReconstructor")
    
    def reconstruct_entities(
        self, 
        tokens: List[str], 
        predictions: List[str]
    ) -> Dict[str, List[str]]:
        """
        Reconstruct entities from tokens and predictions following strict rules.
        
        Args:
            tokens: List of tokens in original order
            predictions: List of predicted tags (BIO format or empty for continuation)
            
        Returns:
            Dictionary mapping entity types to list of reconstructed entities
        """
        if len(tokens) != len(predictions):
            raise ValueError(f"Token count ({len(tokens)}) != prediction count ({len(predictions)})")
        
        # Initialize output
        entities = {entity_type: [] for entity_type in self.ENTITY_TYPES}
        
        # Track current entity being built
        current_tokens = []
        current_entity_type = None
        previous_tag = None
        
        for i, (token, prediction) in enumerate(zip(tokens, predictions)):
            # Handle empty/NA predictions (continuation rule)
            if pd.isna(prediction) or prediction == '':
                if current_tokens and current_entity_type:
                    # Continue current entity
                    current_tokens.append(token)
                continue
            
            # Parse BIO tag
            if prediction == 'O':
                # End current entity
                if current_tokens and current_entity_type:
                    entity_text = ' '.join(current_tokens)
                    entities[current_entity_type].append(entity_text)
                    current_tokens = []
                    current_entity_type = None
                previous_tag = prediction
                continue
            
            # Handle B- and I- tags
            if prediction.startswith('B-'):
                # Begin new entity - first end current entity if exists
                if current_tokens and current_entity_type:
                    entity_text = ' '.join(current_tokens)
                    entities[current_entity_type].append(entity_text)
                
                # Start new entity
                current_entity_type = prediction[2:]
                current_tokens = [token]
                
            elif prediction.startswith('I-'):
                # Continue entity
                predicted_entity_type = prediction[2:]
                
                if current_entity_type is None:
                    # Invalid I-tag without B-tag, treat as B-tag
                    current_entity_type = predicted_entity_type
                    current_tokens = [token]
                elif predicted_entity_type == current_entity_type:
                    # Same entity type, continue
                    current_tokens.append(token)
                else:
                    # Different entity type, end current and start new
                    if current_tokens:
                        entity_text = ' '.join(current_tokens)
                        entities[current_entity_type].append(entity_text)
                    
                    current_entity_type = predicted_entity_type
                    current_tokens = [token]
            
            else:
                # Invalid tag format, treat as outside
                self.logger.warning(f"Invalid tag format at position {i}: {prediction}")
                if current_tokens and current_entity_type:
                    entity_text = ' '.join(current_tokens)
                    entities[current_entity_type].append(entity_text)
                    current_tokens = []
                    current_entity_type = None
            
            previous_tag = prediction
        
        # Handle final entity
        if current_tokens and current_entity_type:
            entity_text = ' '.join(current_tokens)
            entities[current_entity_type].append(entity_text)
        
        return entities
    
    def reconstruct_with_continuation_tags(
        self, 
        tokens: List[str], 
        predictions: List[str],
        reference_tags: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Advanced reconstruction that handles continuation tags properly.
        
        Args:
            tokens: List of tokens
            predictions: List of predicted tags
            reference_tags: Optional reference tags for continuation logic
            
        Returns:
            Dictionary of reconstructed entities
        """
        if reference_tags is not None and len(reference_tags) != len(tokens):
            raise ValueError("Reference tags length must match tokens length")
        
        entities = {entity_type: [] for entity_type in self.ENTITY_TYPES}
        
        current_tokens = []
        current_entity_type = None
        
        for i, (token, prediction) in enumerate(zip(tokens, predictions)):
            # Determine if this is a continuation
            is_continuation = False
            
            if reference_tags is not None:
                # Use reference tags to determine continuation
                ref_tag = reference_tags[i]
                if pd.isna(ref_tag) or ref_tag == '':
                    is_continuation = True
            else:
                # Use prediction itself - empty means continuation
                if pd.isna(prediction) or prediction == '':
                    is_continuation = True
            
            if is_continuation:
                if current_tokens and current_entity_type:
                    current_tokens.append(token)
                continue
            
            # Normal BIO processing
            if prediction == 'O':
                if current_tokens and current_entity_type:
                    entity_text = ' '.join(current_tokens)
                    entities[current_entity_type].append(entity_text)
                    current_tokens = []
                    current_entity_type = None
                continue
            
            if prediction.startswith('B-'):
                if current_tokens and current_entity_type:
                    entity_text = ' '.join(current_tokens)
                    entities[current_entity_type].append(entity_text)
                
                current_entity_type = prediction[2:]
                current_tokens = [token]
                
            elif prediction.startswith('I-'):
                predicted_type = prediction[2:]
                
                if current_entity_type is None:
                    current_entity_type = predicted_type
                    current_tokens = [token]
                elif predicted_type == current_entity_type:
                    current_tokens.append(token)
                else:
                    if current_tokens:
                        entity_text = ' '.join(current_tokens)
                        entities[current_entity_type].append(entity_text)
                    
                    current_entity_type = predicted_type
                    current_tokens = [token]
        
        # Final entity
        if current_tokens and current_entity_type:
            entity_text = ' '.join(current_tokens)
            entities[current_entity_type].append(entity_text)
        
        return entities
    
    def validate_reconstruction(
        self, 
        entities: Dict[str, List[str]], 
        original_tokens: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate reconstructed entities against original tokens.
        
        Args:
            entities: Reconstructed entities dictionary
            original_tokens: Original token list
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check entity types
        for entity_type in entities.keys():
            if entity_type not in self.ENTITY_TYPES:
                errors.append(f"Invalid entity type: {entity_type}")
        
        # Check that reconstructed entities use valid tokens
        all_reconstructed_tokens = []
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_tokens = entity.split(' ')
                all_reconstructed_tokens.extend(entity_tokens)
        
        # Count tokens in original and reconstructed
        original_counts = {}
        for token in original_tokens:
            original_counts[token] = original_counts.get(token, 0) + 1
        
        reconstructed_counts = {}
        for token in all_reconstructed_tokens:
            reconstructed_counts[token] = reconstructed_counts.get(token, 0) + 1
        
        # Check for token count mismatches
        for token, original_count in original_counts.items():
            reconstructed_count = reconstructed_counts.get(token, 0)
            if reconstructed_count > original_count:
                errors.append(f"Too many instances of '{token}': {reconstructed_count} > {original_count}")
        
        # Check for invalid tokens in reconstruction
        for token in reconstructed_counts.keys():
            if token not in original_counts:
                errors.append(f"Invalid token in reconstruction: '{token}'")
        
        return len(errors) == 0, errors
    
    def format_for_submission(
        self, 
        entities: Dict[str, List[str]],
        record_id: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Format entities for submission according to competition requirements.
        
        Args:
            entities: Dictionary of entities
            record_id: Optional record ID
            
        Returns:
            Dictionary formatted for submission
        """
        submission_row = {}
        
        if record_id is not None:
            submission_row['id'] = str(record_id)
        
        # Format each entity type - join multiple entities with '|'
        for entity_type in self.ENTITY_TYPES:
            entity_list = entities.get(entity_type, [])
            if entity_list:
                # Join multiple entities of same type with pipe
                submission_row[entity_type] = '|'.join(entity_list)
            else:
                submission_row[entity_type] = ''
        
        return submission_row
    
    def batch_reconstruct(
        self, 
        batch_data: List[Tuple[List[str], List[str]]]
    ) -> List[Dict[str, List[str]]]:
        """
        Reconstruct entities for a batch of examples.
        
        Args:
            batch_data: List of (tokens, predictions) tuples
            
        Returns:
            List of entity dictionaries
        """
        results = []
        
        for i, (tokens, predictions) in enumerate(batch_data):
            try:
                entities = self.reconstruct_entities(tokens, predictions)
                results.append(entities)
            except Exception as e:
                self.logger.error(f"Error reconstructing batch item {i}: {e}")
                # Return empty entities on error
                empty_entities = {entity_type: [] for entity_type in self.ENTITY_TYPES}
                results.append(empty_entities)
        
        return results
