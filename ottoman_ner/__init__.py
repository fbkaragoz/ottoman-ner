"""
Ottoman Turkish Named Entity Recognition Package

A professional, modular, and configuration-driven package for Ottoman Turkish NER.
"""

__version__ = "2.0.0"

# Configuration system
from .configs import DataConfig, ModelConfig, TrainingConfig, EvaluationConfig

# Models
from .models import BaseNerModel, HuggingFaceNerModel, get_model

# Data processing
from .data import (
    BaseDatasetLoader, CoNLLDatasetLoader, get_dataset_loader,
    create_label_mappings, extract_labels_from_data, validate_bio_sequence
)

# Training
from .training import HuggingFaceModelTrainer

# Evaluation  
from .evaluation import OttomanNEREvaluator

# Legacy components (maintained for backward compatibility)
from .data import ConllToLabelStudio, LabelStudioToConll, DataPreprocessor, EntityAnalyzer
from .training import NERDataset

# Utilities
from .utils import setup_logging

__all__ = [
    # Version
    '__version__',
    
    # Configuration system
    'DataConfig',
    'ModelConfig', 
    'TrainingConfig',
    'EvaluationConfig',
    
    # Models
    'BaseNerModel',
    'HuggingFaceNerModel',
    'get_model',
    
    # Data processing
    'BaseDatasetLoader',
    'CoNLLDatasetLoader', 
    'get_dataset_loader',
    'create_label_mappings',
    'extract_labels_from_data',
    'validate_bio_sequence',
    
    # Training
    'HuggingFaceModelTrainer',
    
    # Evaluation
    'OttomanNEREvaluator',
    
    # Legacy components
    'ConllToLabelStudio',
    'LabelStudioToConll', 
    'DataPreprocessor',
    'EntityAnalyzer',
    'NERDataset',
    
    # Utilities
    'setup_logging'
]

# Package metadata
__author__ = "Ottoman NER Team"
__email__ = "contact@ottoman-ner.org"
__description__ = "A professional package for Ottoman Turkish Named Entity Recognition"
__url__ = "https://github.com/ottoman-ner/ottoman-ner"
