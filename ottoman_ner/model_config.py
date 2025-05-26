"""
Model configurations and mappings for Ottoman NER.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

AVAILABLE_MODELS = {
    "latin": "fatihburakkaragoz/ottoman-ner-latin",
    "arabic": "fatihburakkaragoz/ottoman-ner-arabic", 
    "unified": "fatihburakkaragoz/ottoman-ner-unified"
}

# Local model paths (for development/testing)
LOCAL_MODELS = {
    "latin": "./models/ottoman_ner_latin",
    "arabic": "./models/ottoman_ner_arabic",
    "unified": "./models/ottoman_ner_unified"
}

# Model metadata
MODEL_INFO = {
    "latin": {
        "description": "Ottoman Turkish NER model for Latin script texts",
        "languages": ["ota-Latn"],
        "entities": ["PER", "LOC", "ORG"],
        "base_model": "dbmdz/bert-base-turkish-cased"
    },
    "arabic": {
        "description": "Ottoman Turkish NER model for Arabic script texts", 
        "languages": ["ota-Arab"],
        "entities": ["PER", "LOC", "ORG"],
        "base_model": "dbmdz/bert-base-turkish-cased"
    },
    "unified": {
        "description": "Unified Ottoman Turkish NER model for both scripts",
        "languages": ["ota-Latn", "ota-Arab"],
        "entities": ["PER", "LOC", "ORG"],
        "base_model": "dbmdz/bert-base-turkish-cased"
    }
}


@dataclass
class ModelConfig:
    """Configuration class for Ottoman NER models."""
    
    # Model settings
    model_name: str = "dbmdz/bert-base-turkish-cased"
    max_length: int = 512
    num_labels: int = 11  # O + 2*5 entity types (B-, I-)
    
    # Training settings
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 3
    weight_decay: float = 0.01
    
    # Entity labels
    entity_types: List[str] = None
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = ['PER', 'LOC', 'ORG', 'TIT', 'WORK']
    
    @property
    def label_list(self) -> List[str]:
        """Get the complete label list including BIO tags."""
        labels = ['O']
        for entity_type in self.entity_types:
            labels.extend([f'B-{entity_type}', f'I-{entity_type}'])
        return labels
    
    @property
    def label2id(self) -> Dict[str, int]:
        """Get label to ID mapping."""
        return {label: i for i, label in enumerate(self.label_list)}
    
    @property
    def id2label(self) -> Dict[int, str]:
        """Get ID to label mapping."""
        return {i: label for label, i in self.label2id.items()}


def get_model_path(model_name: str, use_local: bool = False) -> str:
    """Get the model path for a given model name."""
    if use_local:
        if model_name in LOCAL_MODELS:
            return LOCAL_MODELS[model_name]
        else:
            raise ValueError(f"Local model '{model_name}' not found. Available: {list(LOCAL_MODELS.keys())}")
    else:
        if model_name in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(AVAILABLE_MODELS.keys())}")

def list_available_models() -> dict:
    """List all available models with their information."""
    return {
        "remote_models": AVAILABLE_MODELS,
        "local_models": LOCAL_MODELS,
        "model_info": MODEL_INFO
    }
