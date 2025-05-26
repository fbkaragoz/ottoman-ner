"""
Ottoman NER Data Processing Module
"""

# Abstract base classes
from .io_utils import BaseDatasetLoader, BaseDataProcessor, BaseDataConverter
from .io_utils import create_label_mappings, extract_labels_from_data, validate_bio_sequence

# Concrete implementations
from .readers import CoNLLDatasetLoader, JSONDatasetLoader, HuggingFaceDatasetLoader, get_dataset_loader
from .converters import ConllToLabelStudio, LabelStudioToConll
from .preprocessors import DataPreprocessor
from .analyzers import EntityAnalyzer

__all__ = [
    # Abstract base classes
    'BaseDatasetLoader',
    'BaseDataProcessor', 
    'BaseDataConverter',
    
    # Utility functions
    'create_label_mappings',
    'extract_labels_from_data',
    'validate_bio_sequence',
    
    # Concrete implementations
    'CoNLLDatasetLoader',
    'JSONDatasetLoader',
    'HuggingFaceDatasetLoader',
    'get_dataset_loader',
    
    # Legacy components (maintained for backward compatibility)
    'ConllToLabelStudio', 
    'LabelStudioToConll', 
    'DataPreprocessor',
    'EntityAnalyzer'
] 