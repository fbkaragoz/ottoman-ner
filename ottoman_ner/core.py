"""
Core NER prediction functionality for Ottoman Turkish texts.
"""

import torch
import json
import logging
from typing import List, Dict, Union, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from .model_config import get_model_path, MODEL_INFO

logger = logging.getLogger(__name__)

class NERPredictor:
    """
    Ottoman Turkish Named Entity Recognition predictor.
    
    Supports both Arabic and Latin script Ottoman Turkish texts.
    """
    
    def __init__(
        self, 
        model_name_or_path: str, 
        aggregation_strategy: str = "simple",
        use_local: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize the NER predictor.
        
        Args:
            model_name_or_path: Model name (latin/arabic/unified) or HuggingFace model path
            aggregation_strategy: Token aggregation strategy for NER pipeline
            use_local: Whether to use local model paths
            device: Device to run the model on (auto-detected if None)
        """
        self.model_name_or_path = model_name_or_path
        self.aggregation_strategy = aggregation_strategy
        self.use_local = use_local
        
        # Auto-detect device
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device
            
        # Resolve model path
        self._resolve_model_path()
        
        # Load model and tokenizer
        self._load_model()
        
        # Create pipeline
        self._create_pipeline()
        
        logger.info(f"NERPredictor initialized with model: {self.resolved_model_path}")
    
    def _resolve_model_path(self):
        """Resolve the actual model path from model name or path."""
        # Check if it's a predefined model name
        if self.model_name_or_path in ["latin", "arabic", "unified"]:
            self.resolved_model_path = get_model_path(self.model_name_or_path, self.use_local)
            self.model_type = self.model_name_or_path
        else:
            # Assume it's a direct path or HuggingFace model
            self.resolved_model_path = self.model_name_or_path
            self.model_type = "custom"
    
    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.resolved_model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.resolved_model_path)
            
            # Load label mappings if available
            self._load_label_mappings()
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.resolved_model_path}: {e}")
            raise
    
    def _load_label_mappings(self):
        """Load label mappings from model config or separate file."""
        # Try to load from separate JSON file first
        if self.use_local:
            label_file = Path(self.resolved_model_path) / "label_mappings.json"
            if label_file.exists():
                with open(label_file, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                    self.label2id = mappings.get("label2id", {})
                    self.id2label = mappings.get("id2label", {})
                    return
        
        # Fallback to model config
        if hasattr(self.model.config, 'label2id'):
            self.label2id = self.model.config.label2id
            self.id2label = self.model.config.id2label
        else:
            logger.warning("No label mappings found")
            self.label2id = {}
            self.id2label = {}
    
    def _create_pipeline(self):
        """Create the NER pipeline."""
        self.pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=self.aggregation_strategy,
            device=self.device
        )
    
    def predict(self, text: str, return_confidence: bool = True) -> List[Dict]:
        """
        Predict named entities in the given text.
        
        Args:
            text: Input text to analyze
            return_confidence: Whether to include confidence scores
            
        Returns:
            List of detected entities with labels and positions
        """
        if not text or not text.strip():
            return []
        
        try:
            results = self.pipeline(text)
            
            # Format results
            entities = []
            for result in results:
                entity = {
                    "text": result["word"],
                    "label": result["entity_group"],
                    "start": result.get("start", 0),
                    "end": result.get("end", 0),
                }
                if return_confidence:
                    entity["confidence"] = result["score"]
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_file(self, input_path: str, encoding: str = "utf-8") -> List[Dict]:
        """
        Predict named entities from a text file.
        
        Args:
            input_path: Path to the input text file
            encoding: File encoding
            
        Returns:
            List of detected entities
        """
        try:
            with open(input_path, 'r', encoding=encoding) as f:
                text = f.read()
            return self.predict(text)
        except Exception as e:
            logger.error(f"Failed to process file {input_path}: {e}")
            raise
    
    def predict_batch(self, texts: List[str], return_confidence: bool = True) -> List[List[Dict]]:
        """
        Predict named entities for multiple texts.
        
        Args:
            texts: List of input texts
            return_confidence: Whether to include confidence scores
            
        Returns:
            List of entity lists for each input text
        """
        results = []
        for text in texts:
            entities = self.predict(text, return_confidence)
            results.append(entities)
        return results
    
    def predict_sentences(self, text: str, return_confidence: bool = True) -> Dict:
        """
        Predict entities and return sentence-level analysis.
        
        Args:
            text: Input text
            return_confidence: Whether to include confidence scores
            
        Returns:
            Dictionary with sentences and their entities
        """
        # Simple sentence splitting (can be improved)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        result = {
            "text": text,
            "sentences": []
        }
        
        for i, sentence in enumerate(sentences):
            entities = self.predict(sentence, return_confidence)
            result["sentences"].append({
                "sentence_id": i,
                "text": sentence,
                "entities": entities
            })
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        info = {
            "model_path": self.resolved_model_path,
            "model_type": self.model_type,
            "aggregation_strategy": self.aggregation_strategy,
            "device": self.device,
            "labels": list(self.label2id.keys()) if self.label2id else []
        }
        
        if self.model_type in MODEL_INFO:
            info.update(MODEL_INFO[self.model_type])
        
        return info
    
    def save_predictions(self, predictions: Union[List[Dict], Dict], output_path: str):
        """
        Save predictions to a JSON file.
        
        Args:
            predictions: Prediction results
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            logger.info(f"Predictions saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            raise
