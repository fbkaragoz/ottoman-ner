"""
Model configuration for Ottoman NER
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model architecture and loading."""
    
    # Model identification
    model_name_or_path: str = "dbmdz/bert-base-turkish-cased"
    model_type: str = "bert"
    
    # Model architecture
    num_labels: Optional[int] = None
    id2label: Optional[Dict[int, str]] = None
    label2id: Optional[Dict[str, int]] = None
    
    # Model loading options
    cache_dir: Optional[str] = None
    revision: str = "main"
    use_auth_token: Optional[str] = None
    ignore_mismatched_sizes: bool = True
    
    # Model-specific parameters
    hidden_dropout_prob: Optional[float] = None
    attention_probs_dropout_prob: Optional[float] = None
    classifier_dropout: Optional[float] = None
    
    # Optimization settings
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = False
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert cache_dir to string if it's a Path
        if self.cache_dir:
            self.cache_dir = str(Path(self.cache_dir))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model loading."""
        config_dict = {
            'num_labels': self.num_labels,
            'id2label': self.id2label,
            'label2id': self.label2id,
            'cache_dir': self.cache_dir,
            'revision': self.revision,
            'use_auth_token': self.use_auth_token,
            'ignore_mismatched_sizes': self.ignore_mismatched_sizes,
            'trust_remote_code': self.trust_remote_code
        }
        
        # Add model-specific parameters if provided
        if self.hidden_dropout_prob is not None:
            config_dict['hidden_dropout_prob'] = self.hidden_dropout_prob
        if self.attention_probs_dropout_prob is not None:
            config_dict['attention_probs_dropout_prob'] = self.attention_probs_dropout_prob
        if self.classifier_dropout is not None:
            config_dict['classifier_dropout'] = self.classifier_dropout
        
        # Remove None values
        return {k: v for k, v in config_dict.items() if v is not None}
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if self.num_labels is not None and self.num_labels <= 0:
            raise ValueError("num_labels must be positive")
        
        if self.id2label and self.label2id:
            if len(self.id2label) != len(self.label2id):
                raise ValueError("id2label and label2id must have the same length")
        
        return True
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "ModelConfig":
        """Create ModelConfig from a pretrained model."""
        return cls(model_name_or_path=model_name_or_path, **kwargs) 