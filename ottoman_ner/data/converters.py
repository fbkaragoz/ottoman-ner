"""
Data format converters for Ottoman NER
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConllToLabelStudio:
    """Convert CONLL format to Label Studio format."""
    
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
    
    def convert_file(
        self, 
        conll_file: str, 
        output_file: str,
        task_prefix: str = "ottoman_ner"
    ) -> int:
        """
        Convert CONLL file to Label Studio JSON format.
        
        Args:
            conll_file: Path to input CONLL file
            output_file: Path to output JSON file
            task_prefix: Prefix for task IDs
            
        Returns:
            Number of tasks created
        """
        conll_path = Path(conll_file)
        output_path = Path(output_file)
        
        if not conll_path.exists():
            raise FileNotFoundError(f"CONLL file not found: {conll_file}")
        
        # Parse CONLL file
        sentences = self._parse_conll(conll_path)
        
        # Convert to Label Studio format
        tasks = []
        for i, (tokens, labels) in enumerate(sentences):
            task = self._create_task(tokens, labels, f"{task_prefix}_{i}")
            tasks.append(task)
        
        # Save to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding=self.encoding) as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Converted {len(tasks)} tasks from {conll_file} to {output_file}")
        return len(tasks)
    
    def _parse_conll(self, conll_path: Path) -> List[tuple]:
        """Parse CONLL file into sentences."""
        sentences = []
        current_tokens = []
        current_labels = []
        
        with open(conll_path, 'r', encoding=self.encoding) as f:
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
                        logger.warning(f"Invalid line format at {conll_path}:{line_num}: {line}")
        
        # Add last sentence
        if current_tokens:
            sentences.append((current_tokens, current_labels))
        
        return sentences
    
    def _create_task(self, tokens: List[str], labels: List[str], task_id: str) -> Dict[str, Any]:
        """Create a Label Studio task from tokens and labels."""
        text = " ".join(tokens)
        
        # Create annotations
        annotations = []
        if any(label != 'O' for label in labels):
            entities = self._extract_entities(tokens, labels)
            if entities:
                annotation = {
                    "id": f"{task_id}_annotation",
                    "result": entities
                }
                annotations.append(annotation)
        
        task = {
            "id": task_id,
            "data": {"text": text},
            "annotations": annotations
        }
        
        return task
    
    def _extract_entities(self, tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
        """Extract entities from BIO-tagged tokens."""
        entities = []
        current_entity = None
        char_offset = 0
        
        for token, label in zip(tokens, labels):
            token_start = char_offset
            token_end = char_offset + len(token)
            
            if label.startswith('B-'):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    "value": {
                        "start": token_start,
                        "end": token_end,
                        "text": token,
                        "labels": [entity_type]
                    },
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels"
                }
            
            elif label.startswith('I-') and current_entity:
                # Continue current entity
                current_entity["value"]["end"] = token_end
                current_entity["value"]["text"] += " " + token
            
            elif label == 'O':
                # End current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            
            # Update character offset (add space)
            char_offset = token_end + 1
        
        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities


class LabelStudioToConll:
    """Convert Label Studio format to CONLL format."""
    
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
    
    def convert_file(
        self, 
        json_file: str, 
        output_file: str,
        use_latest_annotation: bool = True
    ) -> int:
        """
        Convert Label Studio JSON to CONLL format.
        
        Args:
            json_file: Path to Label Studio JSON file
            output_file: Path to output CONLL file
            use_latest_annotation: Whether to use the latest annotation
            
        Returns:
            Number of sentences converted
        """
        json_path = Path(json_file)
        output_path = Path(output_file)
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        # Load Label Studio data
        with open(json_path, 'r', encoding=self.encoding) as f:
            tasks = json.load(f)
        
        # Convert to CONLL
        sentences = []
        for task in tasks:
            sentence = self._convert_task(task, use_latest_annotation)
            if sentence:
                sentences.append(sentence)
        
        # Write CONLL file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding=self.encoding) as f:
            for tokens, labels in sentences:
                for token, label in zip(tokens, labels):
                    f.write(f"{token}\t{label}\n")
                f.write("\n")
        
        logger.info(f"Converted {len(sentences)} sentences from {json_file} to {output_file}")
        return len(sentences)
    
    def _convert_task(self, task: Dict[str, Any], use_latest: bool = True) -> Optional[tuple]:
        """Convert a single Label Studio task to CONLL format."""
        text = task.get("data", {}).get("text", "")
        if not text:
            return None
        
        annotations = task.get("annotations", [])
        if not annotations:
            # No annotations, return all O labels
            tokens = text.split()
            labels = ['O'] * len(tokens)
            return tokens, labels
        
        # Use latest or first annotation
        annotation = annotations[-1] if use_latest else annotations[0]
        entities = annotation.get("result", [])
        
        # Tokenize and create labels
        tokens = text.split()
        labels = ['O'] * len(tokens)
        
        # Apply entity labels
        for entity in entities:
            if entity.get("type") == "labels":
                self._apply_entity_labels(tokens, labels, entity, text)
        
        return tokens, labels
    
    def _apply_entity_labels(
        self, 
        tokens: List[str], 
        labels: List[str], 
        entity: Dict[str, Any], 
        text: str
    ):
        """Apply entity labels to tokens."""
        value = entity.get("value", {})
        start_char = value.get("start", 0)
        end_char = value.get("end", 0)
        entity_labels = value.get("labels", [])
        
        if not entity_labels:
            return
        
        entity_type = entity_labels[0]
        
        # Find token indices that overlap with entity span
        char_pos = 0
        entity_tokens = []
        
        for i, token in enumerate(tokens):
            token_start = char_pos
            token_end = char_pos + len(token)
            
            # Check if token overlaps with entity
            if (token_start < end_char and token_end > start_char):
                entity_tokens.append(i)
            
            char_pos = token_end + 1  # +1 for space
        
        # Apply BIO labels
        for i, token_idx in enumerate(entity_tokens):
            if i == 0:
                labels[token_idx] = f"B-{entity_type}"
            else:
                labels[token_idx] = f"I-{entity_type}" 