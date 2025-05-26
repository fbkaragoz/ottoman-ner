"""
Evaluation configuration for Ottoman NER
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class EvaluationConfig:
    """Configuration for evaluating Ottoman NER models."""
    
    # Input/Output paths
    model_path: str
    test_file: str
    output_dir: Optional[str] = None
    predictions_file: Optional[str] = None
    
    # Evaluation settings
    batch_size: int = 8
    max_length: int = 512
    
    # Metrics configuration
    metrics: List[str] = None
    average: str = "weighted"  # for sklearn metrics
    labels: Optional[List[str]] = None
    
    # Output options
    save_predictions: bool = True
    save_detailed_report: bool = True
    save_confusion_matrix: bool = True
    save_per_class_metrics: bool = True
    
    # Analysis options
    analyze_errors: bool = True
    error_analysis_top_k: int = 10
    
    # Device settings
    device: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set default metrics
        if self.metrics is None:
            self.metrics = ["precision", "recall", "f1", "accuracy"]
        
        # Convert paths to strings
        self.model_path = str(Path(self.model_path))
        self.test_file = str(Path(self.test_file))
        
        if self.output_dir is None:
            model_name = Path(self.model_path).name
            self.output_dir = f"results/eval_{model_name}"
        self.output_dir = str(Path(self.output_dir))
        
        if self.predictions_file is None:
            self.predictions_file = str(Path(self.output_dir) / "predictions.json")
        else:
            self.predictions_file = str(Path(self.predictions_file))
    
    def validate(self) -> bool:
        """Validate the configuration."""
        # Check that model path exists
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        # Check that test file exists
        if not Path(self.test_file).exists():
            raise FileNotFoundError(f"Test file not found: {self.test_file}")
        
        # Check batch size
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # Check max_length
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        
        # Check metrics
        valid_metrics = {"precision", "recall", "f1", "accuracy", "support"}
        invalid_metrics = set(self.metrics) - valid_metrics
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. Valid metrics: {valid_metrics}")
        
        # Check average type
        if self.average not in ["micro", "macro", "weighted", "binary"]:
            raise ValueError("average must be 'micro', 'macro', 'weighted', or 'binary'")
        
        return True
    
    def get_output_paths(self) -> Dict[str, str]:
        """Get all output file paths."""
        output_dir = Path(self.output_dir)
        return {
            'results': str(output_dir / "evaluation_results.json"),
            'predictions': self.predictions_file,
            'report': str(output_dir / "evaluation_report.txt"),
            'confusion_matrix': str(output_dir / "confusion_matrix.png"),
            'per_class_metrics': str(output_dir / "per_class_metrics.json"),
            'error_analysis': str(output_dir / "error_analysis.json")
        }
    
    @classmethod
    def quick_eval(cls, model_path: str, test_file: str, **kwargs) -> "EvaluationConfig":
        """Create a quick evaluation configuration."""
        return cls(
            model_path=model_path,
            test_file=test_file,
            batch_size=8,
            save_predictions=True,
            analyze_errors=False,
            **kwargs
        )
    
    @classmethod
    def detailed_eval(cls, model_path: str, test_file: str, **kwargs) -> "EvaluationConfig":
        """Create a detailed evaluation configuration with all analysis enabled."""
        return cls(
            model_path=model_path,
            test_file=test_file,
            batch_size=4,  # Smaller batch for detailed analysis
            save_predictions=True,
            save_detailed_report=True,
            save_confusion_matrix=True,
            save_per_class_metrics=True,
            analyze_errors=True,
            error_analysis_top_k=20,
            **kwargs
        ) 