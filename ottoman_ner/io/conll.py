"""
CONLL format data utilities for Ottoman NER.

This module provides functions to read and write CONLL-formatted data
for named entity recognition tasks.
"""

import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def load_conll_data(file_path: str) -> List[List[Tuple[str, str]]]:
    """
    Load CONLL-formatted data from a file.
    
    Args:
        file_path: Path to the CONLL file
        
    Returns:
        List of sentences, where each sentence is a list of (token, tag) tuples
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
        
    Example:
        >>> data = load_conll_data("data/train.conll")
        >>> print(data[0])  # First sentence
        [('Emin', 'B-PER'), ('Bey', 'I-PER'), ('geldi', 'O')]
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"CONLL file not found: {file_path}")
    
    sentences = []
    current_sentence = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Empty line indicates sentence boundary
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue
                
                # Skip comment lines
                if line.startswith('#'):
                    continue
                
                # Parse token and tag
                parts = line.split()
                if len(parts) < 2:
                    logger.warning(f"Line {line_num}: Invalid format, skipping: {line}")
                    continue
                elif len(parts) == 2:
                    token, tag = parts
                else:
                    # Handle cases with more than 2 columns (take first two)
                    token, tag = parts[0], parts[1]
                    logger.debug(f"Line {line_num}: Using first two columns: {token}, {tag}")
                
                current_sentence.append((token, tag))
        
        # Add the last sentence if it doesn't end with an empty line
        if current_sentence:
            sentences.append(current_sentence)
    
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode file {file_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading CONLL file {file_path}: {e}")
    
    logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
    return sentences


def write_conll_data(data: List[List[Tuple[str, str]]], file_path: str) -> None:
    """
    Write data in CONLL format to a file.
    
    Args:
        data: List of sentences, where each sentence is a list of (token, tag) tuples
        file_path: Path where to save the CONLL file
        
    Raises:
        ValueError: If data format is invalid
        IOError: If file cannot be written
        
    Example:
        >>> sentences = [[('Emin', 'B-PER'), ('Bey', 'I-PER')]]
        >>> write_conll_data(sentences, "output.conll")
    """
    file_path = Path(file_path)
    
    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not data:
        logger.warning("No data to write")
        return
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for sentence_idx, sentence in enumerate(data):
                if not sentence:
                    logger.warning(f"Empty sentence at index {sentence_idx}, skipping")
                    continue
                
                for token, tag in sentence:
                    if not isinstance(token, str) or not isinstance(tag, str):
                        raise ValueError(f"Invalid token-tag pair: ({token}, {tag}). Both must be strings.")
                    
                    # Write token and tag separated by tab
                    f.write(f"{token}\t{tag}\n")
                
                # Add empty line between sentences (except after the last sentence)
                if sentence_idx < len(data) - 1:
                    f.write("\n")
    
    except Exception as e:
        raise IOError(f"Failed to write CONLL file {file_path}: {e}")
    
    logger.info(f"Wrote {len(data)} sentences to {file_path}")


def validate_conll_data(data: List[List[Tuple[str, str]]]) -> bool:
    """
    Validate CONLL data format and consistency.
    
    Args:
        data: CONLL data to validate
        
    Returns:
        True if data is valid, False otherwise
        
    Raises:
        ValueError: If data contains critical errors
    """
    if not data:
        logger.warning("Empty data")
        return False
    
    valid_tag_prefixes = {'B-', 'I-', 'O'}
    entity_types = set()
    issues = []
    
    for sent_idx, sentence in enumerate(data):
        if not sentence:
            issues.append(f"Empty sentence at index {sent_idx}")
            continue
        
        prev_tag = None
        for token_idx, (token, tag) in enumerate(sentence):
            # Check token and tag types
            if not isinstance(token, str) or not isinstance(tag, str):
                issues.append(f"Sentence {sent_idx}, token {token_idx}: Invalid types")
                continue
            
            # Check tag format
            if tag == 'O':
                prev_tag = tag
                continue
            
            if not any(tag.startswith(prefix) for prefix in valid_tag_prefixes):
                issues.append(f"Sentence {sent_idx}, token {token_idx}: Invalid tag format '{tag}'")
                continue
            
            # Extract entity type
            if '-' in tag:
                prefix, entity_type = tag.split('-', 1)
                entity_types.add(entity_type)
                
                # Check IOB2 consistency
                if prefix == 'I-' and prev_tag != f'B-{entity_type}' and prev_tag != f'I-{entity_type}':
                    issues.append(f"Sentence {sent_idx}, token {token_idx}: I-tag without preceding B-tag")
            
            prev_tag = tag
    
    if issues:
        logger.warning(f"Found {len(issues)} validation issues:")
        for issue in issues[:10]:  # Show first 10 issues
            logger.warning(f"  - {issue}")
        if len(issues) > 10:
            logger.warning(f"  ... and {len(issues) - 10} more issues")
    
    logger.info(f"Found entity types: {sorted(entity_types)}")
    return len(issues) == 0


def get_conll_statistics(data: List[List[Tuple[str, str]]]) -> dict:
    """
    Get statistics about CONLL data.
    
    Args:
        data: CONLL data
        
    Returns:
        Dictionary with statistics
    """
    if not data:
        return {"sentences": 0, "tokens": 0, "entities": 0, "entity_types": {}}
    
    total_tokens = 0
    total_entities = 0
    entity_types = {}
    
    for sentence in data:
        total_tokens += len(sentence)
        
        current_entity = None
        for token, tag in sentence:
            if tag.startswith('B-'):
                # Start of new entity
                entity_type = tag[2:]
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                total_entities += 1
                current_entity = entity_type
            elif tag.startswith('I-'):
                # Continuation of entity
                entity_type = tag[2:]
                if current_entity != entity_type:
                    # This is a validation issue, but we'll count it
                    logger.warning(f"Inconsistent I-tag: {tag} after {current_entity}")
            else:  # 'O' tag
                current_entity = None
    
    return {
        "sentences": len(data),
        "tokens": total_tokens,
        "entities": total_entities,
        "entity_types": entity_types,
        "avg_tokens_per_sentence": total_tokens / len(data) if data else 0,
        "avg_entities_per_sentence": total_entities / len(data) if data else 0
    }


def convert_to_bio_scheme(data: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
    """
    Convert CONLL data to BIO tagging scheme (if not already).
    
    Args:
        data: CONLL data in any IOB scheme
        
    Returns:
        CONLL data in BIO scheme
    """
    converted_data = []
    
    for sentence in data:
        converted_sentence = []
        prev_entity_type = None
        
        for token, tag in sentence:
            if tag == 'O':
                converted_sentence.append((token, tag))
                prev_entity_type = None
            elif tag.startswith('B-'):
                entity_type = tag[2:]
                converted_sentence.append((token, tag))
                prev_entity_type = entity_type
            elif tag.startswith('I-'):
                entity_type = tag[2:]
                if prev_entity_type == entity_type:
                    converted_sentence.append((token, tag))
                else:
                    # Convert to B- tag if entity type changed
                    converted_sentence.append((token, f'B-{entity_type}'))
                    prev_entity_type = entity_type
            else:
                # Unknown tag, keep as is
                converted_sentence.append((token, tag))
                prev_entity_type = None
        
        converted_data.append(converted_sentence)
    
    return converted_data 