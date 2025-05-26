"""
Ottoman Turkish Named Entity Recognition Toolkit

A simple and modern toolkit for Ottoman Turkish NER tasks.
"""

__version__ = "2.0.0"
__author__ = "Fatih Burak Karagöz"
__email__ = "fatihburakkarag@gmail.com"

# Main interface
from .core import OttomanNER

# Essential utilities
from .utils import setup_logging

__all__ = [
    "OttomanNER",
    "setup_logging",
    "__version__"
]
