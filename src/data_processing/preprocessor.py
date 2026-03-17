"""
NER preprocessing pipeline for token-level tagging

Handles tokenization, feature engineering, and preparation of data
for Named Entity Recognition models with strict formatting constraints.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
from loguru import logger


class NERPreprocessor:
    """
    Preprocessor for NER pipeline with focus on German e-commerce data.
    
    Handles tokenization, feature extraction, and preparation of data
    while preserving original token structure and handling noise.
    """
    
    # Entity types for German automotive e-commerce data
    ENTITY_TYPES = [
        'Hersteller',  # Manufacturer
        'Kompatible_Fahrzeug_Marke',  # Compatible Vehicle Make
        'Kompatibles_Fahrzeug_Modell',  # Compatible Vehicle Model
        'Produktart',  # Product Type
        'Herstellernummer',  # MPN
        'EAN',  # European Article Number
        'Zustand',  # Condition
        'Farbe',  # Color
        'Material',  # Material
        'Anzahl',  # Quantity
        'OEM'  # Original Equipment Manufacturer
    ]
    
    def __init__(self, preserve_duplicates: bool = True):
        """
        Initialize the NER preprocessor.
        
        Args:
            preserve_duplicates: Whether to preserve duplicate tokens (required for submission)
        """
        self.preserve_duplicates = preserve_duplicates
        self.logger = logger.bind(component="NERPreprocessor")
        
        # Common German automotive abbreviations and patterns
        self.abbreviations = {
            'WaPu': 'Wasserpumpe',
            'Bremss': 'Bremssattel',
            'Brems': 'Bremse',
            'Zahnriemen': 'Zahnriemen',
            'Stossd': 'Stossdämpfer',
            'Kuppl': 'Kupplung'
        }
    
    def tokenize_title(self, title: str) -> List[str]:
        """
        Tokenize e-commerce title while preserving original structure.
        
        Args:
            title: Raw title string
            
        Returns:
            List of tokens preserving original order and duplicates
        """
        if not title or pd.isna(title):
            return []
        
        # Handle special cases: TAB characters, multiple spaces
        title = title.replace('\t', ' ')  # Replace TABs with spaces
        
        # Split on whitespace but preserve empty tokens from consecutive spaces
        # This is important for maintaining original structure
        tokens = re.split(r'(\s+)', title)
        
        # Filter out pure whitespace tokens but keep their position information
        # We need to be careful here about the reconstruction rules
        clean_tokens = []
        for token in tokens:
            if token.strip():  # Non-whitespace token
                clean_tokens.append(token.strip())
        
        return clean_tokens
    
    def extract_features(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Extract features for each token.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of feature dictionaries for each token
        """
        features = []
        
        for i, token in enumerate(tokens):
            feature_dict = {
                'token': token,
                'position': i,
                'is_first': i == 0,
                'is_last': i == len(tokens) - 1,
                'length': len(token),
                'is_numeric': token.isdigit(),
                'is_alphanumeric': bool(re.match(r'^[a-zA-Z0-9]+$', token)),
                'has_special_chars': bool(re.search(r'[^a-zA-Z0-9\s]', token)),
                'is_uppercase': token.isupper(),
                'is_lowercase': token.islower(),
                'is_title_case': token.istitle(),
                'contains_digit': bool(re.search(r'\d', token)),
                'is_abbreviation': token in self.abbreviations,
                'is_common_automotive_term': self._is_automotive_term(token)
            }
            
            # Context features
            if i > 0:
                feature_dict['prev_token'] = tokens[i-1]
                feature_dict['prev_length'] = len(tokens[i-1])
            else:
                feature_dict['prev_token'] = '<START>'
                feature_dict['prev_length'] = 0
            
            if i < len(tokens) - 1:
                feature_dict['next_token'] = tokens[i+1]
                feature_dict['next_length'] = len(tokens[i+1])
            else:
                feature_dict['next_token'] = '<END>'
                feature_dict['next_length'] = 0
            
            features.append(feature_dict)
        
        return features
    
    def _is_automotive_term(self, token: str) -> bool:
        """
        Check if token is a common automotive term.
        
        Args:
            token: Token to check
            
        Returns:
            True if token is automotive term
        """
        automotive_terms = {
            'bremsen', 'bremse', 'bremsbelag', 'bremssattel', 'bremsflüssigkeit',
            'filter', 'luftfilter', 'ölfilter', 'kraftstofffilter', 'innenraumfilter',
            'kupplung', 'kupplungsscheibe', 'schwungrad', 'zahnriemen', 'steuerkette',
            'stossdämpfer', 'federbein', 'querlenker', 'spurstange', 'lenkgetriebe',
            'wasserpumpe', 'thermostat', 'kühler', 'heizung', 'klimaanlage',
            'anlasser', 'lichtmaschine', 'zündkerze', 'glühkerze', 'batterie',
            'scheibenwischer', 'scheinwerfer', 'rücklicht', 'blinker', 'spiegel'
        }
        
        return token.lower() in automotive_terms
    
    def parse_bio_tags(self, tag_sequence: List[str]) -> List[Tuple[str, str]]:
        """
        Parse BIO (Begin, Inside, Outside) tag sequence.
        
        Args:
            tag_sequence: List of BIO tags
            
        Returns:
            List of (entity_type, entity_text) tuples
        """
        entities = []
        current_entity = []
        current_type = None
        
        for i, tag in enumerate(tag_sequence):
            if tag == 'O' or pd.isna(tag) or tag == '':
                # End of entity
                if current_entity:
                    entities.append((current_type, ' '.join(current_entity)))
                    current_entity = []
                    current_type = None
                continue
            
            # Parse BIO tag
            if tag.startswith('B-'):
                # Start new entity
                if current_entity:
                    entities.append((current_type, ' '.join(current_entity)))
                
                current_type = tag[2:]
                current_entity = []
            
            elif tag.startswith('I-'):
                # Continue current entity
                if current_type is None:
                    # Invalid I-tag without B-tag, treat as B-tag
                    current_type = tag[2:]
                elif tag[2:] != current_type:
                    # Different entity type, start new entity
                    entities.append((current_type, ' '.join(current_entity)))
                    current_type = tag[2:]
                    current_entity = []
            
            # Add token to current entity
            # Note: We don't have the actual token here, this needs to be called
            # with both tokens and tags together in practice
        
        # Add final entity
        if current_entity:
            entities.append((current_type, ' '.join(current_entity)))
        
        return entities
    
    def reconstruct_entities(
        self, 
        tokens: List[str], 
        tags: List[str]
    ) -> Dict[str, List[str]]:
        """
        Reconstruct entities from tokens and tags following strict rules.
        
        Args:
            tokens: List of tokens
            tags: List of corresponding tags
            
        Returns:
            Dictionary mapping entity types to list of reconstructed entities
        """
        if len(tokens) != len(tags):
            raise ValueError(f"Token count ({len(tokens)}) != tag count ({len(tags)})")
        
        entities = {entity_type: [] for entity_type in self.ENTITY_TYPES}
        
        current_tokens = []
        current_type = None
        
        for token, tag in zip(tokens, tags):
            if pd.isna(tag) or tag == '' or tag == 'O':
                # End of entity or outside
                if current_tokens and current_type:
                    entity_text = ' '.join(current_tokens)
                    if current_type in entities:
                        entities[current_type].append(entity_text)
                
                current_tokens = []
                current_type = None
                continue
            
            # Parse tag
            if tag.startswith('B-'):
                # Begin new entity
                if current_tokens and current_type:
                    # Save previous entity
                    entity_text = ' '.join(current_tokens)
                    if current_type in entities:
                        entities[current_type].append(entity_text)
                
                current_type = tag[2:]
                current_tokens = [token]
            
            elif tag.startswith('I-'):
                # Continue entity
                tag_type = tag[2:]
                if current_type is None:
                    # Invalid I-tag, treat as B-tag
                    current_type = tag_type
                    current_tokens = [token]
                elif tag_type == current_type:
                    # Same entity type, continue
                    current_tokens.append(token)
                else:
                    # Different entity type, start new entity
                    if current_tokens:
                        entity_text = ' '.join(current_tokens)
                        if current_type in entities:
                            entities[current_type].append(entity_text)
                    
                    current_type = tag_type
                    current_tokens = [token]
        
        # Handle final entity
        if current_tokens and current_type:
            entity_text = ' '.join(current_tokens)
            if current_type in entities:
                entities[current_type].append(entity_text)
        
        return entities
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame,
        title_column: str = 'title',
        tag_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare training data from raw DataFrame.
        
        Args:
            df: Raw DataFrame with titles and tags
            title_column: Name of the title column
            tag_columns: List of tag column names (if different from default)
            
        Returns:
            Processed DataFrame ready for training
        """
        if tag_columns is None:
            tag_columns = [col for col in df.columns if col != title_column]
        
        processed_data = []
        
        for idx, row in df.iterrows():
            title = row[title_column]
            tokens = self.tokenize_title(title)
            
            if not tokens:
                continue
            
            # Extract features
            features = self.extract_features(tokens)
            
            # Prepare tags for each token
            # This assumes tags are provided in a specific format
            # You may need to adjust this based on your actual data format
            token_tags = []
            for i, token in enumerate(tokens):
                # Get tag for this token from appropriate column
                # This is a simplified approach - adjust based on your data structure
                tag = 'O'  # Default
                if i < len(tag_columns):
                    tag = row[tag_columns[i]] if pd.notna(row[tag_columns[i]]) else 'O'
                
                token_tags.append(tag)
            
            # Create training example
            for i, (token, feature_dict, tag) in enumerate(zip(tokens, features, token_tags)):
                example = {
                    'record_id': idx,
                    'token_id': i,
                    'token': token,
                    'tag': tag,
                    **feature_dict
                }
                processed_data.append(example)
        
        return pd.DataFrame(processed_data)
