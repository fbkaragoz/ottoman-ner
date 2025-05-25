"""
Evaluation utilities for Ottoman NER package.
"""

from .alignment import get_predictions_in_conll_format, align_predictions_with_tokens

__all__ = ["get_predictions_in_conll_format", "align_predictions_with_tokens"] 