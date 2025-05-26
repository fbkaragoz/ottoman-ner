"""
Abstract base classes and utilities for data I/O in Ottoman NER
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import datasets
from ..configs import DataConfig


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
    
    @abstractmethod
    def load(self, data_config: DataConfig) -> datasets.DatasetDict:
        """
        Load dataset according to configuration.
        
        Args:
            data_config: Data configuration object
            
        Returns:
            DatasetDict with train/dev/test splits
        """
        pass
    
    @abstractmethod
    def validate_data(self, data_config: DataConfig) -> bool:
        """
        Validate that the data configuration is correct.
        
        Args:
            data_config: Data configuration object
            
        Returns:
            True if valid, raises exception otherwise
        """
        pass


class BaseDataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def process(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Process a batch of examples.
        
        Args:
            examples: Batch of examples from datasets
            
        Returns:
            Processed batch of examples
        """
        pass


class BaseDataConverter(ABC):
    """Abstract base class for data format converters."""
    
    @abstractmethod
    def convert(self, input_file: str, output_file: str, **kwargs) -> int:
        """
        Convert data from one format to another.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            **kwargs: Additional conversion parameters
            
        Returns:
            Number of examples converted
        """
        pass


def create_label_mappings(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create bidirectional label mappings.
    
    Args:
        labels: List of label strings
        
    Returns:
        Tuple of (label2id, id2label) mappings
    """
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def extract_labels_from_data(file_paths: List[str], encoding: str = 'utf-8') -> List[str]:
    """
    Extract unique labels from CONLL files.
    
    Args:
        file_paths: List of CONLL file paths
        encoding: File encoding
        
    Returns:
        Sorted list of unique labels
    """
    labels = set()
    
    for file_path in file_paths:
        if not file_path or not Path(file_path).exists():
            continue
            
        with open(file_path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        labels.add(parts[1])
    
    return sorted(list(labels))


def validate_bio_sequence(labels: List[str]) -> List[str]:
    """
    Validate and fix BIO tag sequence.
    
    Args:
        labels: List of BIO labels
        
    Returns:
        Fixed list of BIO labels
    """
    fixed_labels = []
    
    for i, label in enumerate(labels):
        if label.startswith('I-'):
            entity_type = label[2:]
            # Check if previous label is compatible
            if i == 0 or not (fixed_labels[-1] == f'B-{entity_type}' or 
                             fixed_labels[-1] == f'I-{entity_type}'):
                # Fix by converting I- to B-
                fixed_labels.append(f'B-{entity_type}')
            else:
                fixed_labels.append(label)
        else:
            fixed_labels.append(label)
    
    return fixed_labels 