"""
Utility functions for Ottoman NER package.
"""

import os
import re
import json
from typing import List, Dict, Union, Optional
from pathlib import Path

def read_text(path: str, encoding: str = "utf-8") -> str:
    """
    Read text from a file.
    
    Args:
        path: File path
        encoding: File encoding
        
    Returns:
        Text content
    """
    with open(path, "r", encoding=encoding) as f:
        return f.read()

def write_text(text: str, path: str, encoding: str = "utf-8") -> None:
    """
    Write text to a file.
    
    Args:
        text: Text content
        path: Output file path
        encoding: File encoding
    """
    with open(path, "w", encoding=encoding) as f:
        f.write(text)

def read_json(path: str, encoding: str = "utf-8") -> Union[Dict, List]:
    """
    Read JSON data from a file.
    
    Args:
        path: JSON file path
        encoding: File encoding
        
    Returns:
        Parsed JSON data
    """
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)

def write_json(data: Union[Dict, List], path: str, encoding: str = "utf-8", indent: int = 2) -> None:
    """
    Write JSON data to a file.
    
    Args:
        data: Data to write
        path: Output file path
        encoding: File encoding
        indent: JSON indentation
    """
    with open(path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def clean_text(text: str) -> str:
    """
    Clean and normalize text for NER processing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def split_sentences(text: str, method: str = "simple") -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        method: Splitting method ("simple" or "regex")
        
    Returns:
        List of sentences
    """
    if method == "simple":
        # Simple splitting by period
        sentences = [s.strip() for s in text.split('.') if s.strip()]
    elif method == "regex":
        # More sophisticated regex-based splitting
        sentence_pattern = r'[.!?]+\s+'
        sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
    else:
        raise ValueError(f"Unknown splitting method: {method}")
    
    return sentences

def extract_entities_by_type(entities: List[Dict], entity_type: str) -> List[Dict]:
    """
    Extract entities of a specific type.
    
    Args:
        entities: List of entity dictionaries
        entity_type: Entity type to extract (e.g., "PER", "LOC")
        
    Returns:
        Filtered list of entities
    """
    return [entity for entity in entities if entity.get("label") == entity_type]

def group_entities_by_type(entities: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group entities by their type.
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        Dictionary with entity types as keys and entity lists as values
    """
    grouped = {}
    for entity in entities:
        entity_type = entity.get("label", "UNKNOWN")
        if entity_type not in grouped:
            grouped[entity_type] = []
        grouped[entity_type].append(entity)
    
    return grouped

def get_unique_entities(entities: List[Dict]) -> List[Dict]:
    """
    Get unique entities (remove duplicates based on text and label).
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        List of unique entities
    """
    seen = set()
    unique_entities = []
    
    for entity in entities:
        key = (entity.get("text", ""), entity.get("label", ""))
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
    
    return unique_entities

def calculate_entity_statistics(entities: List[Dict]) -> Dict:
    """
    Calculate statistics about detected entities.
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        Dictionary with statistics
    """
    if not entities:
        return {
            "total_entities": 0,
            "unique_entities": 0,
            "entity_types": {},
            "average_confidence": 0.0
        }
    
    # Group by type
    grouped = group_entities_by_type(entities)
    
    # Calculate statistics
    total_entities = len(entities)
    unique_entities = len(get_unique_entities(entities))
    
    entity_types = {}
    for entity_type, entity_list in grouped.items():
        entity_types[entity_type] = {
            "count": len(entity_list),
            "percentage": len(entity_list) / total_entities * 100
        }
    
    # Calculate average confidence if available
    confidences = [e.get("confidence", 0) for e in entities if "confidence" in e]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        "total_entities": total_entities,
        "unique_entities": unique_entities,
        "entity_types": entity_types,
        "average_confidence": average_confidence
    }

def format_entities_for_display(entities: List[Dict], include_confidence: bool = True) -> str:
    """
    Format entities for human-readable display.
    
    Args:
        entities: List of entity dictionaries
        include_confidence: Whether to include confidence scores
        
    Returns:
        Formatted string
    """
    if not entities:
        return "No entities found."
    
    lines = []
    grouped = group_entities_by_type(entities)
    
    for entity_type, entity_list in grouped.items():
        lines.append(f"\n{entity_type} entities:")
        lines.append("-" * 20)
        
        for entity in entity_list:
            text = entity.get("text", "")
            if include_confidence and "confidence" in entity:
                conf = entity["confidence"]
                lines.append(f"  {text} (confidence: {conf:.3f})")
            else:
                lines.append(f"  {text}")
    
    return "\n".join(lines)

def validate_file_path(path: str, must_exist: bool = True) -> bool:
    """
    Validate a file path.
    
    Args:
        path: File path to validate
        must_exist: Whether the file must exist
        
    Returns:
        True if valid, False otherwise
    """
    path_obj = Path(path)
    
    if must_exist:
        return path_obj.exists() and path_obj.is_file()
    else:
        # Check if parent directory exists
        return path_obj.parent.exists()

def ensure_directory(path: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def get_file_extension(path: str) -> str:
    """
    Get file extension from path.
    
    Args:
        path: File path
        
    Returns:
        File extension (without dot)
    """
    return Path(path).suffix.lstrip('.')

def is_text_file(path: str) -> bool:
    """
    Check if a file is a text file based on extension.
    
    Args:
        path: File path
        
    Returns:
        True if it's a text file
    """
    text_extensions = {'txt', 'text', 'md', 'markdown', 'rst'}
    return get_file_extension(path).lower() in text_extensions

def is_json_file(path: str) -> bool:
    """
    Check if a file is a JSON file based on extension.
    
    Args:
        path: File path
        
    Returns:
        True if it's a JSON file
    """
    return get_file_extension(path).lower() == 'json'
