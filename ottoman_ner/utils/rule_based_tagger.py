"""
Rule-based tagger for Ottoman titles and other specific entities.

This module provides functions to automatically tag Ottoman titles
and other entities that can be identified through predefined rules.
"""

from typing import List, Tuple, Set, Dict
import re
import logging

logger = logging.getLogger(__name__)

# Comprehensive list of Ottoman titles
OTTOMAN_TITLES = {
    # High-ranking titles
    "Sultan", "Padişah", "Hünkar", "Şah", "Han", "Halife",
    
    # Administrative titles
    "Paşa", "Vezir", "Sadrazam", "Kaptan", "Defterdar", "Nişancı",
    "Beylerbeyi", "Sancakbeyi", "Mutasarrıf", "Kaymakam", "Müdür",
    
    # Military titles
    "Ağa", "Bey", "Çavuş", "Onbaşı", "Yüzbaşı", "Binbaşı", "Albay",
    "Miralai", "Ferik", "Müşir", "Serdar", "Sipahsalar",
    
    # Religious titles
    "Şeyh", "Molla", "Efendi", "Hoca", "İmam", "Hatip", "Müezzin",
    "Kadı", "Nakib", "Şeyhülislam", "Kazasker", "Müderris",
    "Hafız", "Kari", "Vaiz",
    
    # Court titles
    "Hazretleri", "Devletlü", "Saadetlü", "İzzettlü", "Rif'atlü",
    "Faziletlü", "Muhterem", "Mükerrem",
    
    # Gender-specific titles
    "Hanım", "Hatun", "Kadın", "Sultan", "Valide",
    
    # Professional titles
    "Usta", "Kalfa", "Çırak", "Reis", "Kethüda", "Yiğitbaşı",
    
    # Regional/Tribal titles
    "Voyvoda", "Knez", "Despot", "Hospodar", "Mirza", "Khan"
}

# Additional context-based title patterns
TITLE_PATTERNS = [
    r"\b\w+\s+Paşa\b",  # [Name] Paşa
    r"\b\w+\s+Bey\b",   # [Name] Bey
    r"\b\w+\s+Efendi\b", # [Name] Efendi
    r"\b\w+\s+Ağa\b",   # [Name] Ağa
]

# Common Ottoman works/documents that should be tagged as WORK
OTTOMAN_WORKS = {
    # Historical chronicles
    "Tarih", "Vekayiname", "Tevarih", "Menakıbname", "Gazavatname",
    
    # Religious works
    "Tefsir", "Hadis", "Fıkıh", "Kelam", "Tasavvuf", "Siyer",
    
    # Literary works
    "Divan", "Mesnevî", "Gazel", "Kaside", "Rubai", "Tuyug",
    
    # Administrative documents
    "Kanunname", "Tahrir", "Mühimme", "Sicil", "Temettuat",
    "Vakfiye", "Berat", "Ferman", "Hatt", "Buyruldu",
    
    # Scientific works
    "Risale", "Tercüme", "Şerh", "Haşiye", "Telhis"
}

class RuleBasedTagger:
    def __init__(self, 
                 titles: Set[str] = None,
                 works: Set[str] = None,
                 case_sensitive: bool = False):
        """
        Initialize the rule-based tagger.
        
        Args:
            titles: Set of title words to recognize
            works: Set of work/document names to recognize
            case_sensitive: Whether matching should be case-sensitive
        """
        self.titles = titles or OTTOMAN_TITLES
        self.works = works or OTTOMAN_WORKS
        self.case_sensitive = case_sensitive
        
        # Convert to appropriate case for matching
        if not case_sensitive:
            self.titles = {title.lower() for title in self.titles}
            self.works = {work.lower() for work in self.works}
        
        logger.info(f"Initialized with {len(self.titles)} titles and {len(self.works)} works")
    
    def _normalize_token(self, token: str) -> str:
        """Normalize token for matching."""
        if not self.case_sensitive:
            return token.lower()
        return token
    
    def _is_title(self, token: str) -> bool:
        """Check if a token is a title."""
        normalized = self._normalize_token(token)
        return normalized in self.titles
    
    def _is_work(self, token: str) -> bool:
        """Check if a token is a work/document name."""
        normalized = self._normalize_token(token)
        return normalized in self.works
    
    def tag_titles(self, 
                   tokens_and_tags: List[Tuple[str, str]], 
                   overwrite_existing: bool = False) -> List[Tuple[str, str]]:
        """
        Tag titles in the token sequence.
        
        Args:
            tokens_and_tags: List of (token, tag) tuples
            overwrite_existing: Whether to overwrite existing non-O tags
            
        Returns:
            Updated list of (token, tag) tuples
        """
        result = []
        
        for i, (token, current_tag) in enumerate(tokens_and_tags):
            new_tag = current_tag
            
            # Only tag if current tag is 'O' or we're allowed to overwrite
            if current_tag == 'O' or overwrite_existing:
                if self._is_title(token):
                    new_tag = 'B-TIT'
                    logger.debug(f"Tagged '{token}' as B-TIT")
            
            result.append((token, new_tag))
        
        return result
    
    def tag_works(self, 
                  tokens_and_tags: List[Tuple[str, str]], 
                  overwrite_existing: bool = False) -> List[Tuple[str, str]]:
        """
        Tag works/documents in the token sequence.
        
        Args:
            tokens_and_tags: List of (token, tag) tuples
            overwrite_existing: Whether to overwrite existing non-O tags
            
        Returns:
            Updated list of (token, tag) tuples
        """
        result = []
        
        for i, (token, current_tag) in enumerate(tokens_and_tags):
            new_tag = current_tag
            
            # Only tag if current tag is 'O' or we're allowed to overwrite
            if current_tag == 'O' or overwrite_existing:
                if self._is_work(token):
                    new_tag = 'B-WORK'
                    logger.debug(f"Tagged '{token}' as B-WORK")
            
            result.append((token, new_tag))
        
        return result
    
    def tag_contextual_titles(self, 
                             tokens_and_tags: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Tag titles based on contextual patterns.
        
        This method looks for patterns like "[Name] Paşa" and tags both
        the name and title appropriately.
        """
        result = list(tokens_and_tags)
        tokens = [token for token, _ in tokens_and_tags]
        
        # Look for [Name] + [Title] patterns
        for i in range(len(tokens) - 1):
            current_token = tokens[i]
            next_token = tokens[i + 1]
            
            # Check if next token is a title
            if self._is_title(next_token):
                current_tag = result[i][1]
                next_tag = result[i + 1][1]
                
                # If current token is untagged and looks like a name
                if (current_tag == 'O' and 
                    self._looks_like_name(current_token)):
                    result[i] = (current_token, 'B-PER')
                    logger.debug(f"Tagged '{current_token}' as B-PER (before title)")
                
                # Tag the title if not already tagged
                if next_tag == 'O':
                    result[i + 1] = (next_token, 'B-TIT')
                    logger.debug(f"Tagged '{next_token}' as B-TIT")
        
        return result
    
    def _looks_like_name(self, token: str) -> bool:
        """
        Heuristic to determine if a token looks like a personal name.
        
        This is a simple heuristic - in practice, you might want to use
        more sophisticated methods.
        """
        # Simple heuristics for Ottoman names
        if len(token) < 2:
            return False
        
        # Starts with capital letter
        if not token[0].isupper():
            return False
        
        # Common Ottoman name patterns
        ottoman_name_endings = ['oğlu', 'zade', 'paşa', 'bey', 'ağa']
        for ending in ottoman_name_endings:
            if token.lower().endswith(ending):
                return True
        
        # If it's not a common word and starts with capital, likely a name
        common_words = {'ve', 'ile', 'için', 'olan', 'olan', 'bu', 'şu', 'o'}
        if token.lower() not in common_words:
            return True
        
        return False
    
    def apply_all_rules(self, 
                       tokens_and_tags: List[Tuple[str, str]], 
                       overwrite_existing: bool = False) -> List[Tuple[str, str]]:
        """
        Apply all rule-based tagging methods.
        
        Args:
            tokens_and_tags: List of (token, tag) tuples
            overwrite_existing: Whether to overwrite existing non-O tags
            
        Returns:
            Updated list of (token, tag) tuples with rule-based tags applied
        """
        result = tokens_and_tags
        
        # Apply title tagging
        result = self.tag_titles(result, overwrite_existing)
        
        # Apply work tagging
        result = self.tag_works(result, overwrite_existing)
        
        # Apply contextual title tagging
        result = self.tag_contextual_titles(result)
        
        return result

def apply_rule_based_tagging(data: List[List[Tuple[str, str]]], 
                           overwrite_existing: bool = False) -> List[List[Tuple[str, str]]]:
    """
    Apply rule-based tagging to a dataset.
    
    Args:
        data: List of sentences, each sentence is a list of (token, tag) tuples
        overwrite_existing: Whether to overwrite existing non-O tags
        
    Returns:
        Updated dataset with rule-based tags applied
    """
    tagger = RuleBasedTagger()
    result = []
    
    for sentence in data:
        tagged_sentence = tagger.apply_all_rules(sentence, overwrite_existing)
        result.append(tagged_sentence)
    
    return result

# Utility function for integration with pre-annotation pipeline
def enhance_with_rules(input_file: str, output_file: str, overwrite_existing: bool = False):
    """
    Enhance a CONLL file with rule-based tagging.
    
    Args:
        input_file: Path to input CONLL file
        output_file: Path to output CONLL file
        overwrite_existing: Whether to overwrite existing non-O tags
    """
    from ottoman_ner.io.conll import load_conll_data, write_conll_data
    
    logger.info(f"Applying rule-based tagging: {input_file} -> {output_file}")
    
    # Load data
    data = load_conll_data(input_file)
    
    # Apply rule-based tagging
    enhanced_data = apply_rule_based_tagging(data, overwrite_existing)
    
    # Save enhanced data
    write_conll_data(enhanced_data, output_file)
    
    logger.info(f"Rule-based tagging completed. Saved to {output_file}")

if __name__ == "__main__":
    # Test the rule-based tagger
    test_sentence = [
        ("Mehmet", "O"),
        ("Paşa", "O"),
        ("İstanbul'a", "B-LOC"),
        ("gitti", "O"),
        ("ve", "O"),
        ("Tarih", "O"),
        ("kitabını", "O"),
        ("okudu", "O")
    ]
    
    tagger = RuleBasedTagger()
    result = tagger.apply_all_rules(test_sentence)
    
    print("Original:", test_sentence)
    print("Tagged:  ", result) 