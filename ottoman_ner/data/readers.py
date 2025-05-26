"""
Concrete dataset readers for Ottoman NER
"""

import datasets
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

from .io_utils import BaseDatasetLoader, extract_labels_from_data, create_label_mappings, validate_bio_sequence
from ..configs import DataConfig

logger = logging.getLogger(__name__)


class CoNLLDatasetLoader(BaseDatasetLoader):
    """Concrete implementation for loading CONLL format datasets."""
    
    def __init__(self, encoding: str = 'utf-8', validate_bio: bool = True):
        super().__init__(encoding)
        self.validate_bio = validate_bio
    
    def load(self, data_config: DataConfig) -> datasets.DatasetDict:
        """
        Load CONLL dataset according to configuration.
        
        Args:
            data_config: Data configuration object
            
        Returns:
            DatasetDict with train/dev/test splits
        """
        # Validate configuration
        self.validate_data(data_config)
        
        # Extract labels from data if not provided
        if data_config.label_list is None:
            file_paths = [f for f in [data_config.train_file, data_config.dev_file, data_config.test_file] if f]
            data_config.label_list = extract_labels_from_data(file_paths, self.encoding)
            data_config.label2id, data_config.id2label = create_label_mappings(data_config.label_list)
            logger.info(f"Extracted {len(data_config.label_list)} labels from data: {data_config.label_list}")
        
        # Load each split
        dataset_dict = {}
        
        if data_config.train_file:
            dataset_dict['train'] = self._load_conll_file(data_config.train_file, data_config)
            logger.info(f"Loaded {len(dataset_dict['train'])} training examples")
        
        if data_config.dev_file:
            dataset_dict['validation'] = self._load_conll_file(data_config.dev_file, data_config)
            logger.info(f"Loaded {len(dataset_dict['validation'])} validation examples")
        
        if data_config.test_file:
            dataset_dict['test'] = self._load_conll_file(data_config.test_file, data_config)
            logger.info(f"Loaded {len(dataset_dict['test'])} test examples")
        
        return datasets.DatasetDict(dataset_dict)
    
    def validate_data(self, data_config: DataConfig) -> bool:
        """Validate that the data configuration is correct."""
        if not data_config.train_file:
            raise ValueError("train_file must be provided")
        
        # Check that files exist
        for split, file_path in data_config.get_file_paths().items():
            if file_path and not Path(file_path).exists():
                raise FileNotFoundError(f"{split} file not found: {file_path}")
        
        return True
    
    def _load_conll_file(self, file_path: str, data_config: DataConfig) -> datasets.Dataset:
        """Load a single CONLL file."""
        sentences = self._parse_conll_file(file_path)
        
        # Convert to HuggingFace dataset format
        tokens_list = []
        ner_tags_list = []
        
        for tokens, labels in sentences:
            if self.validate_bio:
                labels = validate_bio_sequence(labels)
            
            # Convert labels to IDs
            label_ids = [data_config.label2id.get(label, data_config.label2id['O']) for label in labels]
            
            tokens_list.append(tokens)
            ner_tags_list.append(label_ids)
        
        # Create dataset
        dataset_dict = {
            data_config.text_column: tokens_list,
            data_config.label_column: ner_tags_list
        }
        
        return datasets.Dataset.from_dict(dataset_dict)
    
    def _parse_conll_file(self, file_path: str) -> List[Tuple[List[str], List[str]]]:
        """Parse CONLL file into sentences."""
        sentences = []
        current_tokens = []
        current_labels = []
        
        with open(file_path, 'r', encoding=self.encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line:
                    if current_tokens:
                        sentences.append((current_tokens.copy(), current_labels.copy()))
                        current_tokens.clear()
                        current_labels.clear()
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        token, label = parts[0], parts[1]
                        current_tokens.append(token)
                        current_labels.append(label)
                    else:
                        logger.warning(f"Invalid line format at {file_path}:{line_num}: {line}")
        
        # Add last sentence if exists
        if current_tokens:
            sentences.append((current_tokens, current_labels))
        
        return sentences


class JSONDatasetLoader(BaseDatasetLoader):
    """Loader for JSON format datasets (e.g., from Label Studio)."""
    
    def load(self, data_config: DataConfig) -> datasets.DatasetDict:
        """Load JSON dataset according to configuration."""
        # This would be implemented for JSON format support
        raise NotImplementedError("JSON dataset loading not yet implemented")
    
    def validate_data(self, data_config: DataConfig) -> bool:
        """Validate JSON data configuration."""
        raise NotImplementedError("JSON dataset validation not yet implemented")


class HuggingFaceDatasetLoader(BaseDatasetLoader):
    """Loader for datasets from HuggingFace Hub."""
    
    def __init__(self, dataset_name: str, encoding: str = 'utf-8'):
        super().__init__(encoding)
        self.dataset_name = dataset_name
    
    def load(self, data_config: DataConfig) -> datasets.DatasetDict:
        """Load dataset from HuggingFace Hub."""
        # This would load from HF Hub
        raise NotImplementedError("HuggingFace Hub dataset loading not yet implemented")
    
    def validate_data(self, data_config: DataConfig) -> bool:
        """Validate HuggingFace dataset configuration."""
        raise NotImplementedError("HuggingFace dataset validation not yet implemented")


def get_dataset_loader(loader_type: str = "conll", **kwargs) -> BaseDatasetLoader:
    """
    Factory function to get appropriate dataset loader.
    
    Args:
        loader_type: Type of loader ('conll', 'json', 'huggingface')
        **kwargs: Additional arguments for the loader
        
    Returns:
        Appropriate dataset loader instance
    """
    loaders = {
        'conll': CoNLLDatasetLoader,
        'json': JSONDatasetLoader,
        'huggingface': HuggingFaceDatasetLoader
    }
    
    if loader_type not in loaders:
        raise ValueError(f"Unknown loader type: {loader_type}. Available: {list(loaders.keys())}")
    
    return loaders[loader_type](**kwargs) 