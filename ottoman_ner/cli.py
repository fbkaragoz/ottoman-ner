#!/usr/bin/env python3
"""
Ottoman NER Command Line Interface
"""

import click
import logging
from pathlib import Path

from .training import OttomanNERTrainer
from .data import ConllToLabelStudio, LabelStudioToConll, EntityAnalyzer
from .evaluation import OttomanNEREvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Ottoman Turkish NER toolkit."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.group()
def train():
    """Training commands."""
    pass


@cli.group()
def data():
    """Data processing commands."""
    pass


@cli.group()
def evaluate():
    """Evaluation commands."""
    pass


# Training commands
@train.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('eval_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--model-name', default='dbmdz/bert-base-turkish-cased', help='HuggingFace model name')
@click.option('--learning-rate', default=2e-5, type=float, help='Learning rate')
@click.option('--batch-size', default=4, type=int, help='Batch size')
@click.option('--epochs', default=3, type=int, help='Number of epochs')
@click.option('--max-length', default=512, type=int, help='Maximum sequence length')
def quick(train_file, eval_file, output_dir, model_name, learning_rate, batch_size, epochs, max_length):
    """Quick training with default settings."""
    click.echo(f"🚀 Starting quick training...")
    click.echo(f"Train: {train_file}")
    click.echo(f"Eval: {eval_file}")
    click.echo(f"Output: {output_dir}")
    
    trainer = OttomanNERTrainer.quick_train(
        train_file=train_file,
        eval_file=eval_file,
        output_dir=output_dir,
        model_name=model_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=epochs,
        max_length=max_length
    )
    
    click.echo("✅ Training completed!")


@train.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('eval_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--model-name', default='dbmdz/bert-base-turkish-cased', help='HuggingFace model name')
@click.option('--learning-rate', default=2e-5, type=float, help='Learning rate')
@click.option('--batch-size', default=4, type=int, help='Batch size')
@click.option('--epochs', default=3, type=int, help='Number of epochs')
@click.option('--weight-decay', default=0.01, type=float, help='Weight decay')
@click.option('--save-steps', default=500, type=int, help='Save every N steps')
@click.option('--eval-steps', default=500, type=int, help='Evaluate every N steps')
@click.option('--logging-steps', default=50, type=int, help='Log every N steps')
@click.option('--seed', default=42, type=int, help='Random seed')
@click.option('--max-length', default=512, type=int, help='Maximum sequence length')
def advanced(train_file, eval_file, output_dir, model_name, learning_rate, batch_size, 
             epochs, weight_decay, save_steps, eval_steps, logging_steps, seed, max_length):
    """Advanced training with custom parameters."""
    click.echo(f"🚀 Starting advanced training...")
    
    trainer = OttomanNERTrainer(model_name=model_name, output_dir=output_dir)
    trainer.setup_model()
    trainer.prepare_datasets(train_file, eval_file, max_length)
    
    trainer.train(
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=epochs,
        weight_decay=weight_decay,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        seed=seed
    )
    
    click.echo("✅ Training completed!")


# Data processing commands
@data.command()
@click.argument('conll_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--task-prefix', default='ottoman_ner', help='Task ID prefix')
def conll_to_labelstudio(conll_file, output_file, task_prefix):
    """Convert CONLL format to Label Studio JSON."""
    converter = ConllToLabelStudio()
    num_tasks = converter.convert_file(conll_file, output_file, task_prefix)
    click.echo(f"✅ Converted {num_tasks} tasks from {conll_file} to {output_file}")


@data.command()
@click.argument('json_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--use-latest/--use-first', default=True, help='Use latest or first annotation')
def labelstudio_to_conll(json_file, output_file, use_latest):
    """Convert Label Studio JSON to CONLL format."""
    converter = LabelStudioToConll()
    num_sentences = converter.convert_file(json_file, output_file, use_latest)
    click.echo(f"✅ Converted {num_sentences} sentences from {json_file} to {output_file}")


@data.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--detailed/--summary', default=True, help='Show detailed analysis')
def analyze(files, detailed):
    """Analyze CONLL dataset files."""
    analyzer = EntityAnalyzer()
    
    if len(files) == 1:
        analysis = analyzer.analyze_conll_file(files[0])
        analyzer.print_analysis(analysis, detailed)
    else:
        comparison = analyzer.compare_datasets(*files)
        for name, analysis in comparison['individual_analyses'].items():
            analyzer.print_analysis(analysis, detailed=False)
        
        # Print comparison summary
        summary = comparison['summary_comparison']
        click.echo(f"\n{'='*60}")
        click.echo("DATASET COMPARISON SUMMARY")
        click.echo(f"{'='*60}")
        
        for name, sizes in summary['dataset_sizes'].items():
            click.echo(f"{name:>15}: {sizes['sentences']:>6} sentences, {sizes['tokens']:>8} tokens, {sizes['entities']:>6} entities")


@data.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--train-ratio', default=0.7, type=float, help='Training set ratio')
@click.option('--dev-ratio', default=0.15, type=float, help='Development set ratio')
@click.option('--test-ratio', default=0.15, type=float, help='Test set ratio')
@click.option('--seed', default=42, type=int, help='Random seed')
def split(input_dir, output_dir, train_ratio, dev_ratio, test_ratio, seed):
    """Split dataset into train/dev/test sets."""
    # This would need to be implemented
    click.echo("Dataset splitting functionality to be implemented...")


# Evaluation commands
@evaluate.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--output-dir', type=click.Path(), help='Output directory for results')
@click.option('--save-predictions/--no-save-predictions', default=True, help='Save detailed predictions')
@click.option('--batch-size', default=8, type=int, help='Evaluation batch size')
def model(model_path, test_file, output_dir, save_predictions, batch_size):
    """Evaluate a trained model."""
    if not output_dir:
        output_dir = f"results/eval_{Path(model_path).name}"
    
    click.echo(f"🔍 Evaluating model: {model_path}")
    click.echo(f"Test file: {test_file}")
    click.echo(f"Output: {output_dir}")
    
    evaluator = OttomanNEREvaluator(model_path)
    results = evaluator.evaluate_file(
        test_file, 
        output_dir=output_dir,
        save_predictions=save_predictions,
        batch_size=batch_size
    )
    
    click.echo("✅ Evaluation completed!")
    click.echo(f"Overall F1: {results['overall']['f1']:.3f}")


@evaluate.command()
@click.argument('predictions_file', type=click.Path(exists=True))
def analyze_failures(predictions_file):
    """Analyze prediction failures from detailed predictions JSON."""
    # This would use the analyze_failures.py logic
    click.echo("Failure analysis functionality to be implemented...")


# Legacy compatibility commands
@cli.command()
@click.argument('text')
def predict(text):
    """Predict entities in text (requires trained model)."""
    click.echo("Prediction functionality to be implemented...")


if __name__ == '__main__':
    cli()
