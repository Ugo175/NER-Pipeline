"""
Data ingestion module for handling compressed TAB-separated files

Supports streaming ingestion of large gzip-compressed datasets with
proper handling of UTF-8 encoding and CSV-style quoting.
"""

import gzip
import pandas as pd
from typing import Iterator, Optional, List, Dict, Any
from pathlib import Path
import logging
from loguru import logger


class DataIngestor:
    """
    Handles ingestion of compressed TAB-separated e-commerce data files.
    
    Designed for streaming large datasets (2M+ records) with memory efficiency.
    """
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize the data ingestor.
        
        Args:
            chunk_size: Number of records to process in each chunk
        """
        self.chunk_size = chunk_size
        self.logger = logger.bind(component="DataIngestor")
    
    def read_compressed_file(
        self, 
        file_path: Path, 
        delimiter: str = '\t',
        encoding: str = 'utf-8',
        quotechar: str = '"'
    ) -> Iterator[pd.DataFrame]:
        """
        Stream read a gzip-compressed TAB-separated file.
        
        Args:
            file_path: Path to the compressed file
            delimiter: Field delimiter (default: TAB)
            encoding: File encoding (default: UTF-8)
            quotechar: Quote character for CSV-style quoting
            
        Yields:
            DataFrame chunks of specified size
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.suffix == '.gz':
            raise ValueError(f"Expected .gz file, got: {file_path.suffix}")
        
        self.logger.info(f"Starting streaming read of {file_path}")
        
        try:
            with gzip.open(file_path, 'rt', encoding=encoding) as f:
                # Read header first
                header = f.readline().strip().split(delimiter)
                
                chunk = []
                for line_num, line in enumerate(f, start=2):  # Start at 2 since header is line 1
                    try:
                        # Handle CSV-style quoting by using pandas csv reader
                        # but we need to process line by line for memory efficiency
                        if line.strip():
                            chunk.append(line.strip())
                            
                            if len(chunk) >= self.chunk_size:
                                yield self._parse_chunk(chunk, header, delimiter, quotechar)
                                chunk = []
                    
                    except Exception as e:
                        self.logger.warning(f"Error parsing line {line_num}: {e}")
                        continue
                
                # Yield remaining chunk
                if chunk:
                    yield self._parse_chunk(chunk, header, delimiter, quotechar)
                    
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def _parse_chunk(
        self, 
        lines: List[str], 
        header: List[str], 
        delimiter: str,
        quotechar: str
    ) -> pd.DataFrame:
        """
        Parse a chunk of lines into a DataFrame.
        
        Args:
            lines: List of raw lines from file
            header: Column headers
            delimiter: Field delimiter
            quotechar: Quote character
            
        Returns:
            Parsed DataFrame
        """
        from io import StringIO
        
        # Join lines and use pandas read_csv for proper CSV parsing
        chunk_text = '\n'.join(lines)
        
        try:
            df = pd.read_csv(
                StringIO(chunk_text),
                sep=delimiter,
                quotechar=quotechar,
                header=None,
                names=header,
                dtype=str,  # Preserve all data as strings to avoid NA conversion
                keep_default_na=False  # Critical: don't convert empty strings to NaN
            )
            return df
        except Exception as e:
            self.logger.error(f"Error parsing chunk: {e}")
            # Fallback: simple split without CSV parsing
            data = [line.split(delimiter) for line in lines]
            return pd.DataFrame(data, columns=header)
    
    def validate_schema(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate DataFrame schema against required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if schema is valid, False otherwise
        """
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        self.logger.info(f"Schema validation passed. Columns: {list(df.columns)}")
        return True
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get basic information about the data file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        info = {
            'file_path': str(file_path),
            'file_size_bytes': file_path.stat().st_size,
            'file_size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
            'is_compressed': file_path.suffix == '.gz'
        }
        
        # Try to get line count for uncompressed files
        if not info['is_compressed']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    info['total_lines'] = sum(1 for _ in f)
            except Exception as e:
                self.logger.warning(f"Could not count lines: {e}")
        
        return info
