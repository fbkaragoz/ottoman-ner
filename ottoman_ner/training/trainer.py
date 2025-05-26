"""
Refactored Ottoman NER Model Trainer using configuration-driven approach
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import logging
from transformers import (
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import DatasetDict
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

from ..configs import DataConfig, ModelConfig, TrainingConfig
from ..models import BaseNerModel, get_model
from ..data import get_dataset_loader

logger = logging.getLogger(__name__)


class HuggingFaceModelTrainer:
    """
    Configuration-driven trainer for Ottoman Turkish NER models.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig
    ):
        """
        Initialize the trainer with configurations.
        
        Args:
            model_config: Model configuration
            data_config: Data configuration  
            training_config: Training configuration
        """
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        
        # Initialize components
        self.model: Optional[BaseNerModel] = None
        self.datasets: Optional[DatasetDict] = None
        self.trainer: Optional[Trainer] = None
        
        logger.info("Initialized HuggingFaceModelTrainer")
        logger.info(f"Model: {model_config.model_name_or_path}")
        logger.info(f"Output: {training_config.output_dir}")
    
    def setup(self):
        """Setup model and datasets."""
        # Validate configurations
        self.data_config.validate()
        self.model_config.validate()
        self.training_config.validate()
        
        # Load datasets
        self._load_datasets()
        
        # Setup model
        self._setup_model()
        
        logger.info("Setup completed successfully")
    
    def _load_datasets(self):
        """Load datasets using the data configuration."""
        logger.info("Loading datasets...")
        
        # Get dataset loader
        loader = get_dataset_loader('conll', encoding=self.data_config.encoding)
        
        # Load datasets
        self.datasets = loader.load(self.data_config)
        
        # Update data config with extracted labels if needed
        if not self.data_config.label_list:
            # This should have been set by the loader
            pass
        
        logger.info(f"Loaded datasets: {list(self.datasets.keys())}")
        for split, dataset in self.datasets.items():
            logger.info(f"  {split}: {len(dataset)} examples")
    
    def _setup_model(self):
        """Setup the model using configurations."""
        logger.info("Setting up model...")
        
        # Update model config with data info
        self.model_config.num_labels = self.data_config.num_labels
        self.model_config.id2label = self.data_config.id2label
        self.model_config.label2id = self.data_config.label2id
        
        # Create model
        self.model = get_model('huggingface', self.model_config)
        
        # Load model
        self.model.load_model(
            num_labels=self.data_config.num_labels,
            id2label=self.data_config.id2label,
            label2id=self.data_config.label2id
        )
        
        logger.info("Model setup completed")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Training results and metrics
        """
        if self.model is None or self.datasets is None:
            raise RuntimeError("Must call setup() before training")
        
        logger.info("Starting training...")
        
        # Prepare datasets for training
        train_dataset = self._prepare_dataset(self.datasets['train'])
        eval_dataset = self._prepare_dataset(self.datasets.get('validation')) if 'validation' in self.datasets else None
        
        # Create training arguments
        training_args = TrainingArguments(**self.training_config.to_training_arguments())
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.model.tokenizer,
            padding=True
        )
        
        # Callbacks
        callbacks = []
        if self.training_config.early_stopping_patience:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience,
                    early_stopping_threshold=self.training_config.early_stopping_threshold
                )
            )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks
        )
        
        # Train
        train_result = self.trainer.train()
        
        # Save model
        self._save_model()
        
        # Evaluate if eval dataset exists
        eval_results = {}
        if eval_dataset is not None:
            eval_results = self.trainer.evaluate()
        
        results = {
            'train_results': train_result,
            'eval_results': eval_results,
            'model_path': self.training_config.output_dir
        }
        
        logger.info("Training completed!")
        return results
    
    def _prepare_dataset(self, dataset):
        """Prepare dataset for training by tokenizing."""
        if dataset is None:
            return None
        
        def tokenize_and_align_labels(examples):
            tokenized_inputs = []
            labels = []
            
            for tokens, ner_tags in zip(examples[self.data_config.text_column], examples[self.data_config.label_column]):
                # Convert label IDs back to strings for alignment
                label_strings = [self.data_config.id2label[tag_id] for tag_id in ner_tags]
                
                # Tokenize and align
                tokenized = self.model.tokenize_and_align_labels(
                    tokens, 
                    label_strings, 
                    max_length=self.data_config.max_length
                )
                
                tokenized_inputs.append({
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask']
                })
                labels.append(tokenized['labels'])
            
            return {
                'input_ids': [t['input_ids'] for t in tokenized_inputs],
                'attention_mask': [t['attention_mask'] for t in tokenized_inputs],
                'labels': labels
            }
        
        return dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.data_config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.data_config.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Compute metrics
        results = {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }
        
        return results
    
    def _save_model(self):
        """Save the trained model."""
        logger.info(f"Saving model to {self.training_config.output_dir}")
        
        # Save using the model's save method
        self.model.save_model(self.training_config.output_dir)
        
        # Save configurations
        config_path = Path(self.training_config.output_dir)
        
        # Save data config
        with open(config_path / "data_config.json", "w", encoding='utf-8') as f:
            json.dump(self.data_config.__dict__, f, indent=2, ensure_ascii=False)
        
        # Save training config
        with open(config_path / "training_config.json", "w", encoding='utf-8') as f:
            json.dump(self.training_config.__dict__, f, indent=2, ensure_ascii=False)
        
        logger.info("Model and configurations saved successfully")
    
    @classmethod
    def from_configs(
        cls,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig
    ) -> "HuggingFaceModelTrainer":
        """
        Create trainer from configuration objects.
        
        Args:
            model_config: Model configuration
            data_config: Data configuration
            training_config: Training configuration
            
        Returns:
            Configured trainer instance
        """
        trainer = cls(model_config, data_config, training_config)
        trainer.setup()
        return trainer
    
    @classmethod
    def quick_train(
        cls,
        train_file: str,
        eval_file: str,
        output_dir: str,
        model_name: str = "dbmdz/bert-base-turkish-cased",
        **kwargs
    ) -> "HuggingFaceModelTrainer":
        """
        Quick training method with sensible defaults.
        
        Args:
            train_file: Path to training file
            eval_file: Path to evaluation file
            output_dir: Output directory
            model_name: Model name or path
            **kwargs: Additional training arguments
            
        Returns:
            Trained model trainer
        """
        # Create configurations
        data_config = DataConfig(
            train_file=train_file,
            dev_file=eval_file,
            tokenizer_name=model_name
        )
        
        model_config = ModelConfig(
            model_name_or_path=model_name
        )
        
        training_config = TrainingConfig.quick_config(
            output_dir=output_dir,
            **kwargs
        )
        
        # Create and train
        trainer = cls.from_configs(model_config, data_config, training_config)
        trainer.train()
        
        return trainer 