"""
Ottoman NER Core Module

Simple and unified interface for Ottoman Turkish Named Entity Recognition.
"""

import torch
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class OttomanNER:
    """
    Simple Ottoman Turkish Named Entity Recognition interface.
    
    Provides unified access to training, evaluation, and prediction functionality.
    """
    
    def __init__(self):
        """Initialize the Ottoman NER system."""
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Default label mappings (BIO format)
        self.default_labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
        self.label2id = {label: i for i, label in enumerate(self.default_labels)}
        self.id2label = {i: label for i, label in enumerate(self.default_labels)}
    
    def load_model(self, model_path: str):
        """Load a trained model from path."""
        model_path = Path(model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mappings if available
        label_file = model_path / "label_mappings.json"
        if label_file.exists():
            with open(label_file, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
                # Convert string keys to integers for id2label
                self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
                self.label2id = mappings['label2id']
        
        logger.info(f"✅ Model loaded from {model_path}")
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Make predictions on a single text."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize
        tokens = text.split()
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Convert predictions to labels
        # Re-tokenize to get word_ids
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        word_ids = tokenized.word_ids()
        predicted_labels = []
        
        for i, word_id in enumerate(word_ids):
            if word_id is not None and i < predictions.shape[1]:
                label_id = predictions[0][i].item()
                label = self.id2label.get(label_id, "O")
                predicted_labels.append(label)
        
        # Extract entities
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, predicted_labels)):
            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'label': label[2:],  # Remove B- prefix
                    'start': i,
                    'end': i + 1,
                    'confidence': 0.9  # Placeholder
                }
            elif label.startswith('I-') and current_entity:
                # Continue current entity
                current_entity['text'] += ' ' + token
                current_entity['end'] = i + 1
            else:
                # End current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def load_conll_data(self, file_path: str) -> Dataset:
        """Load CoNLL format data."""
        sentences = []
        labels = []
        
        current_tokens = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_tokens:
                        sentences.append(current_tokens)
                        labels.append(current_labels)
                        current_tokens = []
                        current_labels = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[-1]  # Last column is label
                        current_tokens.append(token)
                        current_labels.append(label)
        
        # Add final sentence if exists
        if current_tokens:
            sentences.append(current_tokens)
            labels.append(current_labels)
        
        return Dataset.from_dict({
            'tokens': sentences,
            'ner_tags': labels
        })
    
    def tokenize_and_align_labels(self, examples):
        """Tokenize and align labels for training."""
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=True
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id.get(label[word_idx], 0))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
        
        return results
    
    def train_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a model from configuration."""
        logger.info("🚂 Starting training...")
        
        # Extract configuration
        model_config = config.get("model", {})
        data_config = config.get("data", {})
        training_config = config.get("training", {})
        
        # Initialize tokenizer and model
        model_name = model_config.get("model_name_or_path", "dbmdz/bert-base-turkish-cased")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Update label mappings if specified
        num_labels = model_config.get("num_labels", len(self.default_labels))
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Load datasets
        train_dataset = self.load_conll_data(data_config["train_file"])
        eval_dataset = self.load_conll_data(data_config["dev_file"])
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_and_align_labels, batched=True)
        eval_dataset = eval_dataset.map(self.tokenize_and_align_labels, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=training_config.get("output_dir", "models/ottoman-ner"),
            num_train_epochs=training_config.get("num_train_epochs", 3),
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 8),
            learning_rate=training_config.get("learning_rate", 2e-5),
            weight_decay=training_config.get("weight_decay", 0.01),
            warmup_ratio=training_config.get("warmup_ratio", 0.1),
            logging_steps=training_config.get("logging_steps", 50),
            eval_steps=training_config.get("eval_steps", 100),
            save_steps=training_config.get("save_steps", 100),
            eval_strategy=training_config.get("eval_strategy", "steps"),
            save_strategy=training_config.get("save_strategy", "steps"),
            load_best_model_at_end=training_config.get("load_best_model_at_end", True),
            metric_for_best_model=training_config.get("metric_for_best_model", "eval_f1"),
            greater_is_better=training_config.get("greater_is_better", True),
            save_total_limit=training_config.get("save_total_limit", 3),
            report_to=training_config.get("report_to", ["tensorboard"]),
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        
        # Save label mappings
        output_dir = Path(training_args.output_dir)
        with open(output_dir / "label_mappings.json", 'w', encoding='utf-8') as f:
            json.dump({
                'id2label': self.id2label,
                'label2id': self.label2id
            }, f, ensure_ascii=False, indent=2)
        
        # Get final evaluation results
        eval_results = trainer.evaluate()
        
        logger.info("✅ Training completed!")
        return eval_results
    
    def evaluate(self, model_path: str, test_file: str, output_dir: Optional[str] = None) -> Dict[str, float]:
        """Evaluate a trained model."""
        logger.info("📊 Starting evaluation...")
        
        # Load model
        self.load_model(model_path)
        
        # Load test data
        test_dataset = self.load_conll_data(test_file)
        
        # Make predictions
        all_predictions = []
        all_labels = []
        
        for example in test_dataset:
            tokens = example['tokens']
            true_labels = example['ner_tags']
            
            # Get predictions
            entities = self.predict(' '.join(tokens))
            
            # Convert back to BIO format
            predicted_labels = ['O'] * len(tokens)
            for entity in entities:
                start_idx = entity['start']
                end_idx = entity['end']
                label = entity['label']
                
                if start_idx < len(predicted_labels):
                    predicted_labels[start_idx] = f"B-{label}"
                    for i in range(start_idx + 1, min(end_idx, len(predicted_labels))):
                        predicted_labels[i] = f"I-{label}"
            
            all_predictions.append(predicted_labels)
            all_labels.append(true_labels)
        
        # Calculate metrics
        results = {
            'overall_precision': precision_score(all_labels, all_predictions),
            'overall_recall': recall_score(all_labels, all_predictions),
            'overall_f1': f1_score(all_labels, all_predictions)
        }
        
        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / "evaluation_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Save detailed report
            report = classification_report(all_labels, all_predictions, output_dict=True)
            with open(output_path / "detailed_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ Evaluation completed!")
        return results
