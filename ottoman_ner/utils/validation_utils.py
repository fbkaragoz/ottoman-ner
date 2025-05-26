"""
Validation utilities for Ottoman NER
"""

from pathlib import Path
from typing import Union, List


def validate_file_exists(file_path: Union[str, Path], description: str = "File") -> Path:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to file
        description: Description of the file for error messages
        
    Returns:
        Path object if file exists
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{description} is not a file: {path}")
    return path


def validate_directory_exists(dir_path: Union[str, Path], description: str = "Directory") -> Path:
    """
    Validate that a directory exists.
    
    Args:
        dir_path: Path to directory
        description: Description of the directory for error messages
        
    Returns:
        Path object if directory exists
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_dir():
        raise ValueError(f"{description} is not a directory: {path}")
    return path


def validate_files_exist(file_paths: List[Union[str, Path]], description: str = "Files") -> List[Path]:
    """
    Validate that multiple files exist.
    
    Args:
        file_paths: List of file paths
        description: Description of the files for error messages
        
    Returns:
        List of Path objects if all files exist
    """
    validated_paths = []
    for file_path in file_paths:
        validated_paths.append(validate_file_exists(file_path, description))
    return validated_paths


def create_directory_if_not_exists(dir_path: Union[str, Path], description: str = "Directory") -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        dir_path: Path to directory
        description: Description of the directory for logging
        
    Returns:
        Path object
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_extension(file_path: Union[str, Path], allowed_extensions: List[str]) -> Path:
    """
    Validate file extension.
    
    Args:
        file_path: Path to file
        allowed_extensions: List of allowed extensions (e.g., ['.json', '.yaml'])
        
    Returns:
        Path object if extension is valid
        
    Raises:
        ValueError: If extension is not allowed
    """
    path = Path(file_path)
    if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
        raise ValueError(f"File extension '{path.suffix}' not allowed. Allowed: {allowed_extensions}")
    return path 