"""
Ottoman NER Training Module
"""

# New configuration-driven trainer
from .trainer import HuggingFaceModelTrainer

# Legacy components (maintained for backward compatibility)
from .dataset import NERDataset

__all__ = [
    'HuggingFaceModelTrainer',
    'NERDataset'  # Legacy
] 