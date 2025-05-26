"""
Training configuration for Ottoman NER
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training Ottoman NER models."""
    
    # Output and logging
    output_dir: str = "models/ottoman-ner"
    logging_dir: Optional[str] = None
    run_name: Optional[str] = None
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    num_train_epochs: float = 3.0
    max_steps: int = -1
    
    # Optimization
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    
    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    
    # Logging
    logging_strategy: str = "steps"
    logging_steps: int = 50
    report_to: Optional[List[str]] = None
    
    # Hardware and performance
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42
    data_seed: Optional[int] = None
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None
    
    # Custom Ottoman NER specific settings
    label_smoothing_factor: float = 0.0
    ignore_data_skip: bool = False
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert paths to strings
        self.output_dir = str(Path(self.output_dir))
        if self.logging_dir:
            self.logging_dir = str(Path(self.logging_dir))
        else:
            self.logging_dir = str(Path(self.output_dir) / "logs")
        
        # Set data_seed to seed if not provided
        if self.data_seed is None:
            self.data_seed = self.seed
        
        # Auto-detect fp16/bf16 based on hardware if not explicitly set
        if not self.fp16 and not self.bf16:
            try:
                import torch
                if torch.cuda.is_available():
                    # Use fp16 for CUDA by default
                    self.fp16 = True
            except ImportError:
                pass
    
    def to_training_arguments(self) -> Dict[str, Any]:
        """Convert to HuggingFace TrainingArguments compatible dict."""
        return {
            'output_dir': self.output_dir,
            'logging_dir': self.logging_dir,
            'run_name': self.run_name,
            'learning_rate': self.learning_rate,
            'per_device_train_batch_size': self.per_device_train_batch_size,
            'per_device_eval_batch_size': self.per_device_eval_batch_size,
            'num_train_epochs': self.num_train_epochs,
            'max_steps': self.max_steps,
            'weight_decay': self.weight_decay,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_epsilon': self.adam_epsilon,
            'max_grad_norm': self.max_grad_norm,
            'lr_scheduler_type': self.lr_scheduler_type,
            'warmup_ratio': self.warmup_ratio,
            'warmup_steps': self.warmup_steps,
            'eval_strategy': self.eval_strategy,
            'eval_steps': self.eval_steps,
            'save_strategy': self.save_strategy,
            'save_steps': self.save_steps,
            'save_total_limit': self.save_total_limit,
            'load_best_model_at_end': self.load_best_model_at_end,
            'metric_for_best_model': self.metric_for_best_model,
            'greater_is_better': self.greater_is_better,
            'logging_strategy': self.logging_strategy,
            'logging_steps': self.logging_steps,
            'report_to': self.report_to,
            'fp16': self.fp16,
            'bf16': self.bf16,
            'dataloader_num_workers': self.dataloader_num_workers,
            'dataloader_pin_memory': self.dataloader_pin_memory,
            'seed': self.seed,
            'data_seed': self.data_seed,
            'label_smoothing_factor': self.label_smoothing_factor,
            'ignore_data_skip': self.ignore_data_skip
        }
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        
        if self.num_train_epochs <= 0 and self.max_steps <= 0:
            raise ValueError("Either num_train_epochs or max_steps must be positive")
        
        if self.eval_strategy not in ["no", "steps", "epoch"]:
            raise ValueError("eval_strategy must be 'no', 'steps', or 'epoch'")
        
        if self.save_strategy not in ["no", "steps", "epoch"]:
            raise ValueError("save_strategy must be 'no', 'steps', or 'epoch'")
        
        return True
    
    @classmethod
    def quick_config(cls, output_dir: str, **kwargs) -> "TrainingConfig":
        """Create a quick training configuration with sensible defaults."""
        return cls(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
            save_steps=500,
            eval_steps=500,
            logging_steps=50,
            **kwargs
        )
    
    @classmethod
    def advanced_config(cls, output_dir: str, **kwargs) -> "TrainingConfig":
        """Create an advanced training configuration with more sophisticated settings."""
        return cls(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            save_steps=250,
            eval_steps=250,
            logging_steps=25,
            early_stopping_patience=3,
            **kwargs
        ) 