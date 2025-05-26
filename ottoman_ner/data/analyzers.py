"""
Entity analysis tools for Ottoman NER
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class EntityAnalyzer:
    """Analyze entity distributions and patterns in NER datasets."""
    
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
    
    def analyze_conll_file(self, file_path: str) -> Dict[str, any]:
        """
        Analyze a CONLL file and return comprehensive statistics.
        
        Args:
            file_path: Path to CONLL file
            
        Returns:
            Dictionary with analysis results
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sentences = self._load_sentences(file_path)
        
        analysis = {
            'file_info': {
                'path': str(file_path),
                'total_sentences': len(sentences),
                'total_tokens': sum(len(tokens) for tokens, _ in sentences)
            },
            'label_distribution': self._analyze_labels(sentences),
            'entity_distribution': self._analyze_entities(sentences),
            'entity_examples': self._extract_entity_examples(sentences),
            'sentence_stats': self._analyze_sentences(sentences),
            'potential_issues': self._detect_issues(sentences)
        }
        
        return analysis
    
    def compare_datasets(self, *file_paths: str) -> Dict[str, any]:
        """
        Compare multiple datasets.
        
        Args:
            *file_paths: Paths to CONLL files to compare
            
        Returns:
            Comparison analysis
        """
        analyses = {}
        for path in file_paths:
            name = Path(path).stem
            analyses[name] = self.analyze_conll_file(path)
        
        comparison = {
            'individual_analyses': analyses,
            'summary_comparison': self._create_comparison_summary(analyses)
        }
        
        return comparison
    
    def _load_sentences(self, file_path: Path) -> List[Tuple[List[str], List[str]]]:
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
    
    def _analyze_labels(self, sentences: List[Tuple[List[str], List[str]]]) -> Dict[str, any]:
        """Analyze label distribution."""
        label_counts = Counter()
        total_tokens = 0
        
        for _, labels in sentences:
            for label in labels:
                label_counts[label] += 1
                total_tokens += 1
        
        # Calculate percentages
        label_percentages = {
            label: (count / total_tokens) * 100 
            for label, count in label_counts.items()
        }
        
        return {
            'counts': dict(label_counts),
            'percentages': label_percentages,
            'total_tokens': total_tokens,
            'unique_labels': len(label_counts)
        }
    
    def _analyze_entities(self, sentences: List[Tuple[List[str], List[str]]]) -> Dict[str, any]:
        """Analyze entity-level statistics."""
        entity_counts = Counter()
        entity_lengths = defaultdict(list)
        entity_examples = defaultdict(list)
        
        for tokens, labels in sentences:
            entities = self._extract_entities_from_bio(tokens, labels)
            
            for entity_type, entity_text, start, end in entities:
                entity_counts[entity_type] += 1
                entity_lengths[entity_type].append(end - start)
                
                if len(entity_examples[entity_type]) < 10:  # Keep first 10 examples
                    entity_examples[entity_type].append(entity_text)
        
        # Calculate average lengths
        avg_lengths = {
            entity_type: sum(lengths) / len(lengths) if lengths else 0
            for entity_type, lengths in entity_lengths.items()
        }
        
        return {
            'counts': dict(entity_counts),
            'average_lengths': avg_lengths,
            'examples': dict(entity_examples),
            'total_entities': sum(entity_counts.values())
        }
    
    def _extract_entities_from_bio(self, tokens: List[str], labels: List[str]) -> List[Tuple[str, str, int, int]]:
        """Extract entities from BIO-tagged sequence."""
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                # End previous entity
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity
                entity_type = label[2:]
                current_entity = (entity_type, token, i, i + 1)
            
            elif label.startswith('I-') and current_entity:
                # Continue current entity
                entity_type, entity_text, start, _ = current_entity
                current_entity = (entity_type, entity_text + " " + token, start, i + 1)
            
            elif label == 'O':
                # End current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add final entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _extract_entity_examples(self, sentences: List[Tuple[List[str], List[str]]]) -> Dict[str, List[str]]:
        """Extract example entities for each type."""
        examples = defaultdict(set)
        
        for tokens, labels in sentences:
            entities = self._extract_entities_from_bio(tokens, labels)
            for entity_type, entity_text, _, _ in entities:
                examples[entity_type].add(entity_text)
        
        # Convert to lists and limit
        return {
            entity_type: list(entity_set)[:20]  # First 20 unique examples
            for entity_type, entity_set in examples.items()
        }
    
    def _analyze_sentences(self, sentences: List[Tuple[List[str], List[str]]]) -> Dict[str, any]:
        """Analyze sentence-level statistics."""
        sentence_lengths = [len(tokens) for tokens, _ in sentences]
        entities_per_sentence = []
        
        for tokens, labels in sentences:
            entities = self._extract_entities_from_bio(tokens, labels)
            entities_per_sentence.append(len(entities))
        
        return {
            'avg_length': sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0,
            'min_length': min(sentence_lengths) if sentence_lengths else 0,
            'max_length': max(sentence_lengths) if sentence_lengths else 0,
            'avg_entities_per_sentence': sum(entities_per_sentence) / len(entities_per_sentence) if entities_per_sentence else 0,
            'sentences_with_entities': sum(1 for count in entities_per_sentence if count > 0),
            'sentences_without_entities': sum(1 for count in entities_per_sentence if count == 0)
        }
    
    def _detect_issues(self, sentences: List[Tuple[List[str], List[str]]]) -> List[str]:
        """Detect potential issues in the dataset."""
        issues = []
        
        # Check for invalid BIO sequences
        for i, (tokens, labels) in enumerate(sentences):
            for j, label in enumerate(labels):
                if label.startswith('I-'):
                    if j == 0 or not labels[j-1].endswith(label[2:]):
                        issues.append(f"Invalid I- tag without B- at sentence {i}, token {j}: {label}")
        
        # Check for very short entities
        short_entities = 0
        for tokens, labels in sentences:
            entities = self._extract_entities_from_bio(tokens, labels)
            for _, entity_text, _, _ in entities:
                if len(entity_text.split()) == 1 and len(entity_text) <= 2:
                    short_entities += 1
        
        if short_entities > 0:
            issues.append(f"Found {short_entities} very short entities (≤2 characters)")
        
        # Check for label consistency
        all_labels = set()
        for _, labels in sentences:
            all_labels.update(labels)
        
        unexpected_labels = all_labels - {'O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-TIT', 'I-TIT', 'B-WORK', 'I-WORK'}
        if unexpected_labels:
            issues.append(f"Unexpected labels found: {unexpected_labels}")
        
        return issues
    
    def _create_comparison_summary(self, analyses: Dict[str, Dict]) -> Dict[str, any]:
        """Create a summary comparison of multiple datasets."""
        summary = {
            'dataset_sizes': {},
            'entity_type_coverage': {},
            'label_distribution_comparison': {}
        }
        
        for name, analysis in analyses.items():
            summary['dataset_sizes'][name] = {
                'sentences': analysis['file_info']['total_sentences'],
                'tokens': analysis['file_info']['total_tokens'],
                'entities': analysis['entity_distribution']['total_entities']
            }
            
            summary['entity_type_coverage'][name] = list(analysis['entity_distribution']['counts'].keys())
        
        return summary
    
    def print_analysis(self, analysis: Dict[str, any], detailed: bool = True):
        """Print analysis results in a readable format."""
        print(f"\n{'='*60}")
        print(f"DATASET ANALYSIS: {analysis['file_info']['path']}")
        print(f"{'='*60}")
        
        # File info
        info = analysis['file_info']
        print(f"\n📊 BASIC STATISTICS:")
        print(f"  Sentences: {info['total_sentences']:,}")
        print(f"  Tokens: {info['total_tokens']:,}")
        
        # Entity distribution
        entities = analysis['entity_distribution']
        print(f"\n🏷️  ENTITY DISTRIBUTION:")
        print(f"  Total entities: {entities['total_entities']:,}")
        for entity_type, count in sorted(entities['counts'].items()):
            percentage = (count / entities['total_entities']) * 100 if entities['total_entities'] > 0 else 0
            avg_len = entities['average_lengths'].get(entity_type, 0)
            print(f"  {entity_type:>6}: {count:>4} ({percentage:5.1f}%) - Avg length: {avg_len:.1f}")
        
        # Label distribution
        if detailed:
            labels = analysis['label_distribution']
            print(f"\n🔖 LABEL DISTRIBUTION:")
            for label, count in sorted(labels['counts'].items()):
                percentage = labels['percentages'][label]
                print(f"  {label:>8}: {count:>6} ({percentage:5.1f}%)")
        
        # Issues
        issues = analysis['potential_issues']
        if issues:
            print(f"\n⚠️  POTENTIAL ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✅ No issues detected!")
        
        print(f"\n{'='*60}")


def analyze_dataset_files(*file_paths: str, detailed: bool = True) -> Dict[str, any]:
    """
    Convenience function to analyze multiple dataset files.
    
    Args:
        *file_paths: Paths to CONLL files
        detailed: Whether to print detailed analysis
        
    Returns:
        Analysis results
    """
    analyzer = EntityAnalyzer()
    
    if len(file_paths) == 1:
        analysis = analyzer.analyze_conll_file(file_paths[0])
        if detailed:
            analyzer.print_analysis(analysis)
        return analysis
    else:
        comparison = analyzer.compare_datasets(*file_paths)
        if detailed:
            for name, analysis in comparison['individual_analyses'].items():
                analyzer.print_analysis(analysis, detailed=False)
        return comparison 