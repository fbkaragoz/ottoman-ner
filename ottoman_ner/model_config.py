"""
Model configurations and mappings for Ottoman NER.
"""

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
