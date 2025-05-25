"""
Alignment utilities for converting model predictions to CONLL format.

This module handles the complex task of aligning model predictions
(which may use subword tokenization) with pre-tokenized input tokens
to create token-level IOB2 tags compatible with CONLL format.
"""

import logging
import re
from typing import List, Tuple, Dict, Any, Optional
from ottoman_ner.core import NERPredictor

logger = logging.getLogger(__name__)


def get_predictions_in_conll_format(
    predictor: NERPredictor, 
    sentences_tokens: List[List[str]]
) -> List[List[Tuple[str, str]]]:
    """
    Get model predictions in CONLL format for pre-tokenized sentences.
    
    Args:
        predictor: An instance of NERPredictor
        sentences_tokens: List of sentences, where each sentence is a list of pre-tokenized strings
        
    Returns:
        List of sentences, where each sentence is a list of (token, predicted_tag) tuples
        
    Example:
        >>> predictor = NERPredictor("latin")
        >>> sentences = [["Emin", "Bey", "geldi"], ["İstanbul'da", "yaşıyor"]]
        >>> result = get_predictions_in_conll_format(predictor, sentences)
        >>> print(result[0])
        [('Emin', 'B-PER'), ('Bey', 'I-PER'), ('geldi', 'O')]
    """
    if not sentences_tokens:
        return []
    
    result = []
    
    for sentence_idx, tokens in enumerate(sentences_tokens):
        if not tokens:
            logger.warning(f"Empty sentence at index {sentence_idx}")
            result.append([])
            continue
        
        try:
            # Reconstruct sentence text
            sentence_text = " ".join(tokens)
            
            # Get model predictions
            predictions = predictor.predict(sentence_text, return_confidence=True)
            
            # Align predictions with original tokens
            aligned_tags = align_predictions_with_tokens(tokens, predictions, sentence_text)
            
            # Create (token, tag) tuples
            sentence_conll = [(token, tag) for token, tag in zip(tokens, aligned_tags)]
            result.append(sentence_conll)
            
        except Exception as e:
            logger.error(f"Failed to process sentence {sentence_idx}: {e}")
            # Return all 'O' tags as fallback
            fallback_tags = ['O'] * len(tokens)
            sentence_conll = [(token, tag) for token, tag in zip(tokens, fallback_tags)]
            result.append(sentence_conll)
    
    return result


def align_predictions_with_tokens(
    original_tokens: List[str], 
    predictions: List[Dict[str, Any]], 
    sentence_text: str
) -> List[str]:
    """
    Align model predictions with original tokens to create IOB2 tags.
    
    Args:
        original_tokens: List of original tokens
        predictions: List of prediction dictionaries from NERPredictor
        sentence_text: The reconstructed sentence text
        
    Returns:
        List of IOB2 tags for each original token
        
    Note:
        This function handles the complex alignment between model predictions
        (which may use character offsets or different tokenization) and the
        original tokenization used in CONLL data.
    """
    if not original_tokens:
        return []
    
    # Initialize all tags as 'O'
    tags = ['O'] * len(original_tokens)
    
    if not predictions:
        return tags
    
    # Create character-to-token mapping
    char_to_token = create_char_to_token_mapping(original_tokens, sentence_text)
    
    # Process each prediction
    for pred in predictions:
        try:
            entity_text = pred.get('text', '').strip()
            entity_label = pred.get('label', '').strip()
            start_char = pred.get('start', 0)
            end_char = pred.get('end', 0)
            
            if not entity_text or not entity_label:
                continue
            
            # Find which tokens this entity spans
            entity_tokens = find_entity_tokens(
                start_char, end_char, char_to_token, original_tokens, entity_text, sentence_text
            )
            
            if entity_tokens:
                # Apply IOB2 tagging
                apply_iob2_tags(tags, entity_tokens, entity_label)
                
        except Exception as e:
            logger.warning(f"Failed to process prediction {pred}: {e}")
            continue
    
    return tags


def create_char_to_token_mapping(tokens: List[str], sentence_text: str) -> List[int]:
    """
    Create a mapping from character positions to token indices.
    
    Args:
        tokens: List of original tokens
        sentence_text: Reconstructed sentence text
        
    Returns:
        List where index is character position and value is token index
    """
    char_to_token = [-1] * len(sentence_text)
    current_pos = 0
    
    for token_idx, token in enumerate(tokens):
        # Find the token in the sentence text starting from current position
        token_start = sentence_text.find(token, current_pos)
        
        if token_start == -1:
            # Token not found, try fuzzy matching
            token_start = find_fuzzy_token_position(token, sentence_text, current_pos)
        
        if token_start != -1:
            token_end = token_start + len(token)
            # Map all characters in this token to the token index
            for char_pos in range(token_start, min(token_end, len(sentence_text))):
                char_to_token[char_pos] = token_idx
            current_pos = token_end
        else:
            logger.warning(f"Could not find token '{token}' in sentence '{sentence_text}'")
    
    return char_to_token


def find_fuzzy_token_position(token: str, sentence_text: str, start_pos: int) -> int:
    """
    Find token position using fuzzy matching for cases where exact match fails.
    
    This handles cases where tokenization differences exist between the original
    tokens and the reconstructed sentence.
    """
    # Try to find the token with some flexibility
    search_text = sentence_text[start_pos:]
    
    # Remove punctuation and try again
    clean_token = re.sub(r'[^\w]', '', token.lower())
    
    for i, char in enumerate(search_text):
        if i + len(clean_token) <= len(search_text):
            window = search_text[i:i+len(clean_token)]
            clean_window = re.sub(r'[^\w]', '', window.lower())
            if clean_window == clean_token:
                return start_pos + i
    
    return -1


def find_entity_tokens(
    start_char: int, 
    end_char: int, 
    char_to_token: List[int], 
    original_tokens: List[str],
    entity_text: str,
    sentence_text: str
) -> List[int]:
    """
    Find which original tokens correspond to an entity prediction.
    
    Args:
        start_char: Start character position of entity
        end_char: End character position of entity
        char_to_token: Character to token mapping
        original_tokens: List of original tokens
        entity_text: The predicted entity text
        sentence_text: Full sentence text
        
    Returns:
        List of token indices that correspond to this entity
    """
    entity_tokens = set()
    
    # Method 1: Use character positions
    for char_pos in range(start_char, min(end_char, len(char_to_token))):
        token_idx = char_to_token[char_pos]
        if token_idx != -1:
            entity_tokens.add(token_idx)
    
    # Method 2: If no tokens found, try text matching
    if not entity_tokens:
        entity_tokens = find_tokens_by_text_matching(entity_text, original_tokens)
    
    return sorted(list(entity_tokens))


def find_tokens_by_text_matching(entity_text: str, original_tokens: List[str]) -> set:
    """
    Find tokens that match the entity text using text similarity.
    
    This is a fallback method when character-based alignment fails.
    """
    entity_tokens = set()
    entity_words = entity_text.split()
    
    if not entity_words:
        return entity_tokens
    
    # Try to find consecutive tokens that match the entity words
    for start_idx in range(len(original_tokens)):
        if start_idx + len(entity_words) <= len(original_tokens):
            candidate_tokens = original_tokens[start_idx:start_idx + len(entity_words)]
            
            # Check if tokens match (with some flexibility)
            if tokens_match_entity(candidate_tokens, entity_words):
                for i in range(start_idx, start_idx + len(entity_words)):
                    entity_tokens.add(i)
                break
    
    return entity_tokens


def tokens_match_entity(tokens: List[str], entity_words: List[str]) -> bool:
    """
    Check if tokens match entity words with some flexibility.
    """
    if len(tokens) != len(entity_words):
        return False
    
    for token, word in zip(tokens, entity_words):
        # Clean both strings for comparison
        clean_token = re.sub(r'[^\w]', '', token.lower())
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        # Check for exact match or substring match
        if clean_token != clean_word and clean_token not in clean_word and clean_word not in clean_token:
            return False
    
    return True


def apply_iob2_tags(tags: List[str], entity_tokens: List[int], entity_label: str) -> None:
    """
    Apply IOB2 tags to the specified token positions.
    
    Args:
        tags: List of current tags (modified in place)
        entity_tokens: List of token indices for this entity
        entity_label: Entity label (e.g., 'PER', 'LOC')
    """
    if not entity_tokens:
        return
    
    # Sort token indices
    entity_tokens = sorted(entity_tokens)
    
    # Apply B- tag to first token
    if 0 <= entity_tokens[0] < len(tags):
        tags[entity_tokens[0]] = f'B-{entity_label}'
    
    # Apply I- tags to remaining tokens
    for token_idx in entity_tokens[1:]:
        if 0 <= token_idx < len(tags):
            tags[token_idx] = f'I-{entity_label}'


def validate_alignment_quality(
    original_tokens: List[str], 
    predictions: List[Dict[str, Any]], 
    aligned_tags: List[str]
) -> Dict[str, Any]:
    """
    Validate the quality of prediction alignment.
    
    Args:
        original_tokens: Original tokens
        predictions: Model predictions
        aligned_tags: Aligned IOB2 tags
        
    Returns:
        Dictionary with alignment quality metrics
    """
    if len(original_tokens) != len(aligned_tags):
        return {"error": "Length mismatch between tokens and tags"}
    
    # Count entities in predictions vs aligned tags
    pred_entities = len([p for p in predictions if p.get('label')])
    aligned_entities = len([tag for tag in aligned_tags if tag.startswith('B-')])
    
    # Count tokens with non-O tags
    non_o_tokens = len([tag for tag in aligned_tags if tag != 'O'])
    
    # Check IOB2 consistency
    iob2_valid = validate_iob2_sequence(aligned_tags)
    
    return {
        "prediction_entities": pred_entities,
        "aligned_entities": aligned_entities,
        "non_o_tokens": non_o_tokens,
        "entity_alignment_ratio": aligned_entities / max(pred_entities, 1),
        "iob2_valid": iob2_valid,
        "total_tokens": len(original_tokens)
    }


def validate_iob2_sequence(tags: List[str]) -> bool:
    """
    Validate that a sequence of IOB2 tags is consistent.
    
    Args:
        tags: List of IOB2 tags
        
    Returns:
        True if sequence is valid, False otherwise
    """
    prev_tag = None
    
    for tag in tags:
        if tag.startswith('I-'):
            entity_type = tag[2:]
            if prev_tag != f'B-{entity_type}' and prev_tag != f'I-{entity_type}':
                return False
        prev_tag = tag
    
    return True 