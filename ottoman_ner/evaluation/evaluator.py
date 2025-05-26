"""
Ottoman NER Model Evaluator
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np

logger = logging.getLogger(__name__)


class OttomanNEREvaluator:
    """
    Evaluator for Ottoman Turkish NER models.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model
            device: Device to use for evaluation (auto-detected if None)
        """
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mappings
        self.label_mappings = self._load_label_mappings()
        self.id2label = self.label_mappings.get('id2label', {})
        self.label2id = self.label_mappings.get('label2id', {})
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Labels: {list(self.label2id.keys())}")
    
    def _load_label_mappings(self) -> Dict[str, Any]:
        """Load label mappings from the model directory."""
        mappings_file = self.model_path / "label_mappings.json"
        if mappings_file.exists():
            with open(mappings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Fallback to model config
            return {
                'id2label': self.model.config.id2label,
                'label2id': self.model.config.label2id
            }
    
    def predict_tokens(self, tokens: List[str]) -> List[str]:
        """
        Predict labels for a list of tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of predicted labels
        """
        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Align predictions with original tokens
        word_ids = encoding.word_ids()
        aligned_predictions = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                # First subword of a word
                pred_id = predictions[0][len(aligned_predictions)].item()
                label = self.id2label.get(str(pred_id), 'O')
                aligned_predictions.append(label)
            previous_word_idx = word_idx
        
        # Ensure we have the right number of predictions
        while len(aligned_predictions) < len(tokens):
            aligned_predictions.append('O')
        
        return aligned_predictions[:len(tokens)]
    
    def evaluate_file(
        self, 
        test_file: str, 
        output_dir: Optional[str] = None,
        save_predictions: bool = True,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a test file.
        
        Args:
            test_file: Path to test CONLL file
            output_dir: Directory to save results
            save_predictions: Whether to save detailed predictions
            batch_size: Batch size for evaluation
            
        Returns:
            Evaluation results
        """
        test_path = Path(test_file)
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        # Load test data
        sentences = self._load_conll_file(test_path)
        
        # Predict
        all_predictions = []
        all_gold_labels = []
        detailed_predictions = []
        
        logger.info(f"Evaluating on {len(sentences)} sentences...")
        
        for i, (tokens, gold_labels) in enumerate(sentences):
            if i % 100 == 0:
                logger.info(f"Processing sentence {i+1}/{len(sentences)}")
            
            # Predict
            pred_labels = self.predict_tokens(tokens)
            
            # Store results
            all_predictions.append(pred_labels)
            all_gold_labels.append(gold_labels)
            
            if save_predictions:
                detailed_predictions.append({
                    'sentence_id': i,
                    'tokens': tokens,
                    'gold_tags': gold_labels,
                    'predicted_tags': pred_labels
                })
        
        # Calculate metrics
        results = self._calculate_metrics(all_gold_labels, all_predictions)
        
        # Save results
        if output_dir:
            self._save_results(results, detailed_predictions, output_dir, save_predictions)
        
        return results
    
    def _load_conll_file(self, file_path: Path) -> List[Tuple[List[str], List[str]]]:
        """Load sentences from CONLL file."""
        sentences = []
        current_tokens = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
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
    
    def _calculate_metrics(
        self, 
        gold_labels: List[List[str]], 
        pred_labels: List[List[str]]
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        
        # Overall metrics
        overall_f1 = f1_score(gold_labels, pred_labels)
        overall_precision = precision_score(gold_labels, pred_labels)
        overall_recall = recall_score(gold_labels, pred_labels)
        
        # Detailed classification report
        report = classification_report(gold_labels, pred_labels, output_dict=True)
        
        results = {
            'overall': {
                'f1': overall_f1,
                'precision': overall_precision,
                'recall': overall_recall
            },
            'per_entity': {},
            'classification_report': report
        }
        
        # Extract per-entity metrics
        for label, metrics in report.items():
            if isinstance(metrics, dict) and label not in ['micro avg', 'macro avg', 'weighted avg']:
                results['per_entity'][label] = {
                    'f1': metrics.get('f1-score', 0.0),
                    'precision': metrics.get('precision', 0.0),
                    'recall': metrics.get('recall', 0.0),
                    'support': metrics.get('support', 0)
                }
        
        return results
    
    def _save_results(
        self, 
        results: Dict[str, Any], 
        detailed_predictions: List[Dict[str, Any]], 
        output_dir: str,
        save_predictions: bool
    ):
        """Save evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_path / "evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save detailed predictions
        if save_predictions and detailed_predictions:
            with open(output_path / "detailed_predictions.json", 'w', encoding='utf-8') as f:
                json.dump(detailed_predictions, f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        with open(output_path / "evaluation_report.txt", 'w', encoding='utf-8') as f:
            f.write("OTTOMAN NER EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write(f"F1 Score:  {results['overall']['f1']:.3f}\n")
            f.write(f"Precision: {results['overall']['precision']:.3f}\n")
            f.write(f"Recall:    {results['overall']['recall']:.3f}\n\n")
            
            f.write("PER-ENTITY METRICS:\n")
            for entity, metrics in results['per_entity'].items():
                f.write(f"{entity:>8}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, Support={metrics['support']}\n")
        
        logger.info(f"Results saved to {output_path}")
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a readable format."""
        print(f"\n{'='*60}")
        print("OTTOMAN NER EVALUATION RESULTS")
        print(f"{'='*60}")
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"  F1 Score:  {results['overall']['f1']:.3f}")
        print(f"  Precision: {results['overall']['precision']:.3f}")
        print(f"  Recall:    {results['overall']['recall']:.3f}")
        
        print(f"\n🏷️  PER-ENTITY METRICS:")
        for entity, metrics in sorted(results['per_entity'].items()):
            print(f"  {entity:>8}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, Support={metrics['support']}")
        
        print(f"\n{'='*60}")


def evaluate_model(
    model_path: str, 
    test_file: str, 
    output_dir: Optional[str] = None,
    save_predictions: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model.
    
    Args:
        model_path: Path to the trained model
        test_file: Path to test CONLL file
        output_dir: Directory to save results
        save_predictions: Whether to save detailed predictions
        
    Returns:
        Evaluation results
    """
    evaluator = OttomanNEREvaluator(model_path)
    results = evaluator.evaluate_file(test_file, output_dir, save_predictions)
    evaluator.print_results(results)
    return results 