"""
Configuration utilities for Ottoman NER
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union
from dataclasses import asdict, is_dataclass


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def save_config(
    config: Union[Dict[str, Any], Any], 
    config_path: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary or dataclass
        config_path: Path to save configuration
        format: Output format ('json' or 'yaml')
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dict if needed
    if is_dataclass(config):
        config_dict = asdict(config)
    else:
        config_dict = config
    
    if format.lower() == 'json':
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    elif format.lower() in ['yaml', 'yml']:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    else:
        raise ValueError(f"Unsupported format: {format}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        if config:
            merged = _deep_merge(merged, config)
    
    return merged


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config_structure(config: Dict[str, Any], required_keys: Dict[str, type]) -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        required_keys: Dictionary of required keys and their expected types
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
        
        if not isinstance(config[key], expected_type):
            raise ValueError(f"Configuration key '{key}' must be of type {expected_type.__name__}, got {type(config[key]).__name__}")
    
    return True 