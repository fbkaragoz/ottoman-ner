"""
Data preprocessing utilities for Ottoman NER
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess Ottoman Turkish texts for NER training."""
    
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Ottoman Turkish text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize punctuation
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        return text.strip()
    
    def tokenize_ottoman_text(self, text: str) -> List[str]:
        """
        Tokenize Ottoman Turkish text with special handling for compounds.
        
        This is a basic tokenizer that can be improved based on linguistic rules.
        """
        # Split on whitespace first
        tokens = text.split()
        
        # Further split on punctuation but keep it
        final_tokens = []
        for token in tokens:
            # Split on punctuation but keep it
            parts = re.split(r'([.,;:!?()"\'-])', token)
            for part in parts:
                if part and part.strip():
                    final_tokens.append(part.strip())
        
        return final_tokens
    
    def detect_compound_entities(self, tokens: List[str]) -> List[Tuple[int, int, str]]:
        """
        Detect potential compound entities in tokenized text.
        
        Returns:
            List of (start_idx, end_idx, entity_type) tuples
        """
        compounds = []
        
        # Common Ottoman compound patterns
        compound_patterns = {
            'ORG': [
                r'.*-i\s+.*',  # X-i Y pattern (e.g., "Cemiyet-i Coğrafyası")
                r'.*\s+Komisyonu',  # X Komisyonu
                r'.*\s+Mahkemesi',  # X Mahkemesi
                r'.*\s+Şirketi',    # X Şirketi
            ],
            'WORK': [
                r'Tarih-i\s+.*',    # Tarih-i X
                r'.*\s+Divanı',     # X Divanı
                r'Risale-i\s+.*',   # Risale-i X
            ],
            'LOC': [
                r'Vilayet-i\s+.*',  # Vilayet-i X
            ]
        }
        
        text = ' '.join(tokens)
        
        for entity_type, patterns in compound_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    start_char = match.start()
                    end_char = match.end()
                    
                    # Convert character positions to token positions
                    start_token = self._char_to_token_pos(tokens, start_char)
                    end_token = self._char_to_token_pos(tokens, end_char - 1) + 1
                    
                    if start_token is not None and end_token is not None:
                        compounds.append((start_token, end_token, entity_type))
        
        return compounds
    
    def _char_to_token_pos(self, tokens: List[str], char_pos: int) -> Optional[int]:
        """Convert character position to token position."""
        current_pos = 0
        for i, token in enumerate(tokens):
            if current_pos <= char_pos < current_pos + len(token):
                return i
            current_pos += len(token) + 1  # +1 for space
        return None
    
    def suggest_entity_keywords(self) -> Dict[str, List[str]]:
        """Return keyword patterns for different entity types."""
        return {
            'WORK': [
                'kitap', 'eser', 'roman', 'tercüme', 'makale', 'gazete', 
                'mecmua', 'divan', 'risale', 'tarih', 'tefsir', 'mühimme',
                'tahrir', 'şiir', 'opera', 'tiyatro', 'hikaye', 'mektup'
            ],
            'ORG': [
                'cemiyet', 'şirket', 'komisyon', 'mahkeme', 'darülfünun',
                'mekteb', 'encümen', 'ordu', 'hükümet', 'devlet', 'müze',
                'matbaa', 'sefarethane', 'donanma'
            ],
            'PER': [
                'bey', 'paşa', 'efendi', 'hanım', 'sultan', 'şah', 'mir',
                'ağa', 'çelebi', 'hazretleri'
            ],
            'LOC': [
                'vilayet', 'sancak', 'kaza', 'nahiye', 'köy', 'mahalle',
                'sokak', 'meydan', 'köprü', 'cami', 'saray', 'kale'
            ],
            'TIT': [
                'sultan', 'paşa', 'bey', 'efendi', 'ağa', 'çelebi',
                'müdür', 'nazır', 'vali', 'kaymakam'
            ]
        }
    
    def preprocess_conll_file(
        self, 
        input_file: str, 
        output_file: str,
        clean_text: bool = True,
        detect_compounds: bool = False
    ) -> int:
        """
        Preprocess a CONLL file.
        
        Args:
            input_file: Input CONLL file path
            output_file: Output CONLL file path
            clean_text: Whether to clean the text
            detect_compounds: Whether to detect compound entities
            
        Returns:
            Number of sentences processed
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        sentences = self._load_conll_sentences(input_path)
        processed_sentences = []
        
        for tokens, labels in sentences:
            if clean_text:
                # Clean tokens
                tokens = [self.clean_text(token) for token in tokens]
                # Remove empty tokens
                filtered = [(t, l) for t, l in zip(tokens, labels) if t.strip()]
                if filtered:
                    tokens, labels = zip(*filtered)
                    tokens, labels = list(tokens), list(labels)
            
            if detect_compounds:
                # This is a placeholder - compound detection would need more sophisticated logic
                pass
            
            if tokens:  # Only add non-empty sentences
                processed_sentences.append((tokens, labels))
        
        # Write processed sentences
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding=self.encoding) as f:
            for tokens, labels in processed_sentences:
                for token, label in zip(tokens, labels):
                    f.write(f"{token}\t{label}\n")
                f.write("\n")
        
        logger.info(f"Preprocessed {len(processed_sentences)} sentences from {input_file} to {output_file}")
        return len(processed_sentences)
    
    def _load_conll_sentences(self, file_path: Path) -> List[Tuple[List[str], List[str]]]:
        """Load sentences from CONLL file."""
        sentences = []
        current_tokens = []
        current_labels = []
        
        with open(file_path, 'r', encoding=self.encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_tokens:
                        sentences.append((current_tokens.copy(), current_labels.copy()))
                        current_tokens.clear()
                        current_labels.clear()
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_labels.append(parts[1])
        
        if current_tokens:
            sentences.append((current_tokens, current_labels))
        
        return sentences