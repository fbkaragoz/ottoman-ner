"""
Abstract base class for NER models
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import torch
from ..configs import ModelConfig


class BaseNerModel(ABC):
    """Abstract base class for NER models."""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.device = None
    
    @abstractmethod
    def load_model(
        self, 
        num_labels: int, 
        id2label: Dict[int, str], 
        label2id: Dict[str, int]
    ) -> None:
        """
        Load and initialize the model.
        
        Args:
            num_labels: Number of labels
            id2label: Mapping from label IDs to label strings
            label2id: Mapping from label strings to label IDs
        """
        pass
    
    @abstractmethod
    def predict(self, tokenized_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Make predictions on tokenized inputs.
        
        Args:
            tokenized_inputs: Dictionary with tokenized inputs
            
        Returns:
            Prediction logits or probabilities
        """
        pass
    
    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            inputs: Model inputs
            
        Returns:
            Model outputs
        """
        pass
    
    @abstractmethod
    def save_model(self, output_dir: str) -> None:
        """
        Save the model to a directory.
        
        Args:
            output_dir: Directory to save the model
        """
        pass
    
    @abstractmethod
    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
        """
        pass
    
    def to(self, device: Union[str, torch.device]) -> "BaseNerModel":
        """Move model to device."""
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self
    
    def eval(self) -> "BaseNerModel":
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        return self
    
    def train(self) -> "BaseNerModel":
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
        return self
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.__class__.__name__,
            'model_name_or_path': self.model_config.model_name_or_path,
            'num_labels': self.model_config.num_labels,
            'device': str(self.device) if self.device else None,
            'is_loaded': self.is_loaded
        } 