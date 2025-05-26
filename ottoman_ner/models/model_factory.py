"""
Model factory for creating NER model instances
"""

from typing import Dict, Any
from .base import BaseNerModel
from .huggingface_model import HuggingFaceNerModel
from ..configs import ModelConfig


def get_model(model_type: str = "huggingface", model_config: ModelConfig = None, **kwargs) -> BaseNerModel:
    """
    Factory function to get appropriate model instance.
    
    Args:
        model_type: Type of model ('huggingface', 'custom', etc.)
        model_config: Model configuration object
        **kwargs: Additional arguments for the model
        
    Returns:
        Appropriate model instance
    """
    if model_config is None:
        model_config = ModelConfig(**kwargs)#TODO: code is unreachable
    
    models = {
        'huggingface': HuggingFaceNerModel,
        'hf': HuggingFaceNerModel,  # Alias
        'transformers': HuggingFaceNerModel,  # Alias
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](model_config)


def create_model_from_config(config_dict: Dict[str, Any]) -> BaseNerModel:
    """
    Create model from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_type = config_dict.pop('model_type', 'huggingface')
    model_config = ModelConfig(**config_dict)
    return get_model(model_type, model_config) 