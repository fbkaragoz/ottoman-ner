"""
HuggingFace-based NER model implementation
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForTokenClassification
import logging

from .base import BaseNerModel
from ..configs import ModelConfig

logger = logging.getLogger(__name__)


class HuggingFaceNerModel(BaseNerModel):
    """Concrete implementation using HuggingFace transformers."""
    
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.id2label = None
        self.label2id = None
    
    def load_model(
        self, 
        num_labels: int, 
        id2label: Dict[int, str], 
        label2id: Dict[str, int]
    ) -> None:
        """Load and initialize the HuggingFace model."""
        logger.info(f"Loading model: {self.model_config.model_name_or_path}")
        
        # Store label mappings
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.label2id = label2id
        
        # Update model config
        self.model_config.num_labels = num_labels
        self.model_config.id2label = self.id2label
        self.model_config.label2id = self.label2id
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name_or_path,
            use_fast=self.model_config.use_fast_tokenizer,
            trust_remote_code=self.model_config.trust_remote_code,
            cache_dir=self.model_config.cache_dir
        )
        
        # Load model
        model_kwargs = self.model_config.to_dict()
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_config.model_name_or_path,
            **model_kwargs
        )
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Model has {num_labels} labels: {list(label2id.keys())}")
    
    def predict(self, tokenized_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make predictions on tokenized inputs."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        with torch.no_grad():
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Get predictions
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        return predictions
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        
        return {
            'logits': outputs.logits,
            'loss': outputs.loss if hasattr(outputs, 'loss') else None
        }
    
    def save_model(self, output_dir: str) -> None:
        """Save the model to a directory."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Cannot save.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save label mappings
        label_mappings = {
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()}
        }
        
        with open(output_path / "label_mappings.json", "w", encoding='utf-8') as f:
            json.dump(label_mappings, f, indent=2, ensure_ascii=False)
        
        # Save model configuration
        config_dict = {
            "model_name_or_path": self.model_config.model_name_or_path,
            "model_type": self.model_config.model_type,
            "num_labels": self.model_config.num_labels,
            "labels": list(self.label2id.keys())
        }
        
        with open(output_path / "model_config.json", "w", encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model saved to {output_path}")
    
    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from a checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load label mappings if available
        label_mappings_file = checkpoint_path / "label_mappings.json"
        if label_mappings_file.exists():
            with open(label_mappings_file, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
                self.label2id = mappings.get('label2id', {})
                self.id2label = {int(k): v for k, v in mappings.get('id2label', {}).items()}
        
        # Update model config path
        self.model_config.model_name_or_path = str(checkpoint_path)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Model loaded from checkpoint: {checkpoint_path}")
    
    def tokenize_and_align_labels(
        self, 
        tokens: List[str], 
        labels: Optional[List[str]] = None,
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize tokens and align labels with subword tokens.
        
        Args:
            tokens: List of tokens
            labels: List of labels (optional)
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized inputs and aligned labels
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        
        # Align labels if provided
        if labels is not None:
            aligned_labels = self._align_labels(encoding.word_ids(), labels)
            result['labels'] = torch.tensor(aligned_labels, dtype=torch.long)
        
        return result
    
    def _align_labels(self, word_ids: List[Optional[int]], labels: List[str]) -> List[int]:
        """Align labels with subword tokens."""
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the label
                if word_idx < len(labels):
                    label = labels[word_idx]
                    label_id = self.label2id.get(label, self.label2id.get('O', 0))
                    aligned_labels.append(label_id)
                else:
                    aligned_labels.append(self.label2id.get('O', 0))
            else:
                # Subsequent subwords get -100 (ignored in loss)
                aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        
        return aligned_labels
    
    def predict_tokens(self, tokens: List[str]) -> List[str]:
        """
        Predict labels for a list of tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of predicted labels
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Tokenize
        tokenized = self.tokenize_and_align_labels(tokens)
        
        # Add batch dimension
        for key in tokenized:
            tokenized[key] = tokenized[key].unsqueeze(0)
        
        # Predict
        predictions = self.predict(tokenized)
        
        # Align predictions with original tokens
        word_ids = self.tokenizer(
            tokens, 
            is_split_into_words=True, 
            return_tensors='pt'
        ).word_ids()
        
        aligned_predictions = []
        previous_word_idx = None
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                # First subword of a word
                if i < len(predictions[0]):
                    pred_id = predictions[0][i].item()
                    label = self.id2label.get(pred_id, 'O')
                    aligned_predictions.append(label)
            previous_word_idx = word_idx
        
        # Ensure we have the right number of predictions
        while len(aligned_predictions) < len(tokens):
            aligned_predictions.append('O')
        
        return aligned_predictions[:len(tokens)] 