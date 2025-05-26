"""
Evaluation utilities for Ottoman NER package.
"""

from .alignment import get_predictions_in_conll_format, align_predictions_with_tokens
from .evaluator import OttomanNEREvaluator, evaluate_model

__all__ = [
    "get_predictions_in_conll_format", 
    "align_predictions_with_tokens",
    "OttomanNEREvaluator",
    "evaluate_model"
] 