from .core import NERPredictor
from .model_config import AVAILABLE_MODELS

# Import IO utilities
from .io import load_conll_data, write_conll_data

# Import evaluation utilities
from .evaluation import get_predictions_in_conll_format, align_predictions_with_tokens

__version__ = "0.2.0"
__author__ = "Ottoman NER Team"
__description__ = "Named Entity Recognition for Ottoman Turkish texts"
__all__ = [
    "NERPredictor", 
    "AVAILABLE_MODELS",
    "load_conll_data",
    "write_conll_data", 
    "get_predictions_in_conll_format",
    "align_predictions_with_tokens"
]
