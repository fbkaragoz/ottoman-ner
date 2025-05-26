"""
Data configuration for Ottoman NER
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Data paths
    train_file: Optional[str] = None
    dev_file: Optional[str] = None
    test_file: Optional[str] = None
    data_dir: Optional[str] = None
    
    # Tokenizer configuration
    tokenizer_name: str = "dbmdz/bert-base-turkish-cased"
    max_length: int = 512
    
    # Label configuration
    label_list: Optional[List[str]] = None
    label2id: Optional[Dict[str, int]] = None
    id2label: Optional[Dict[int, str]] = None
    
    # Data processing options
    encoding: str = 'utf-8'
    text_column: str = 'tokens'
    label_column: str = 'ner_tags'
    
    # Preprocessing options
    clean_text: bool = True
    detect_compounds: bool = False
    normalize_text: bool = True
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set default label list if not provided
        if self.label_list is None:
            self.label_list = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 
                              'B-ORG', 'I-ORG', 'B-TIT', 'I-TIT', 'B-WORK', 'I-WORK']
        
        # Create label mappings if not provided
        if self.label2id is None:
            self.label2id = {label: i for i, label in enumerate(self.label_list)}
        
        if self.id2label is None:
            self.id2label = {i: label for label, i in self.label2id.items()}
        
        # Convert paths to Path objects if they're strings
        if self.train_file:
            self.train_file = str(Path(self.train_file))
        if self.dev_file:
            self.dev_file = str(Path(self.dev_file))
        if self.test_file:
            self.test_file = str(Path(self.test_file))
        if self.data_dir:
            self.data_dir = str(Path(self.data_dir))
    
    @property
    def num_labels(self) -> int:
        """Get the number of labels."""
        return len(self.label_list)
    
    def get_file_paths(self) -> Dict[str, Optional[str]]:
        """Get all file paths as a dictionary."""
        return {
            'train': self.train_file,
            'dev': self.dev_file,
            'test': self.test_file
        }
    
    def validate(self) -> bool:
        """Validate the configuration."""
        # Check that at least train file is provided
        if not self.train_file:
            raise ValueError("train_file must be provided")
        
        # Check that files exist
        for split, file_path in self.get_file_paths().items():
            if file_path and not Path(file_path).exists():
                raise FileNotFoundError(f"{split} file not found: {file_path}")
        
        # Check label consistency
        if len(self.label2id) != len(self.id2label):
            raise ValueError("label2id and id2label must have the same length")
        
        return True 