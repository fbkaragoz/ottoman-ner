"""
NER Dataset Classes for Ottoman Turkish NER
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class NERDataset(Dataset):
    """
    Dataset class for NER training with Ottoman Turkish texts.
    
    Handles CONLL format files and provides proper tokenization and label alignment.
    """
    
    def __init__(
        self, 
        file_path: str, 
        tokenizer, 
        label2id: Dict[str, int], 
        max_length: int = 512,
        encoding: str = 'utf-8'
    ):
        """
        Initialize NER Dataset.
        
        Args:
            file_path: Path to CONLL format file
            tokenizer: HuggingFace tokenizer
            label2id: Mapping from labels to IDs
            max_length: Maximum sequence length
            encoding: File encoding
        """
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        self.sentences = self._load_sentences(encoding)
        logger.info(f"Loaded {len(self.sentences)} sentences from {file_path}")
    
    def _load_sentences(self, encoding: str) -> List[Tuple[List[str], List[str]]]:
        """Load sentences from CONLL file."""
        sentences = []
        current_tokens = []
        current_labels = []
        
        with open(self.file_path, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line:
                    if current_tokens:
                        sentences.append((current_tokens.copy(), current_labels.copy()))
                        current_tokens.clear()
                        current_labels.clear()
                else:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        token, label = parts
                        current_tokens.append(token)
                        current_labels.append(label)
                    else:
                        logger.warning(f"Invalid line format at {self.file_path}:{line_num}: {line}")
        
        # Add last sentence if exists
        if current_tokens:
            sentences.append((current_tokens, current_labels))
        
        return sentences
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        tokens, labels = self.sentences[idx]
        
        # Tokenize with word-level alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align labels with subword tokens
        aligned_labels = self._align_labels(encoding.word_ids(), labels)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }
    
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
                    aligned_labels.append(self.label2id.get(labels[word_idx], self.label2id['O']))
                else:
                    aligned_labels.append(self.label2id['O'])
            else:
                # Subsequent subwords get -100 (ignored in loss)
                aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        
        return aligned_labels
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        label_counts = {}
        for _, labels in self.sentences:
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts 