"""
Logging utilities for Ottoman NER
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for Ottoman NER.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log messages
        
    Returns:
        Configured logger instance
    """
    # Default format
    if format_string is None:
        if include_timestamp:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[]
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: Union[str, int]):
    """
    Set the logging level for all loggers.
    
    Args:
        level: Logging level
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    logging.getLogger().setLevel(level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(level) 