"""
Command Line Interface for Ottoman NER.
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Optional

from ottoman_ner import NERPredictor
from ottoman_ner.model_config import list_available_models

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def setup_parser() -> argparse.ArgumentParser:
    """Setup the argument parser."""
    parser = argparse.ArgumentParser(
        description="Ottoman Turkish Named Entity Recognition CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ottoman-ner --text "Emin Bey'in kuklaları Tepebaşı'nda oynuyor"
  ottoman-ner --input text.txt --output results.json --script arabic
  ottoman-ner --input text.txt --script latin --confidence
  ottoman-ner --list-models
  ottoman-ner --model-info latin
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--text", 
        help="Input text to analyze"
    )
    input_group.add_argument(
        "--input", 
        help="Path to input text file"
    )
    
    # Output options
    parser.add_argument(
        "--output", 
        help="Path to save output JSON file (default: stdout)"
    )
    
    # Model options
    parser.add_argument(
        "--script", 
        choices=["arabic", "latin", "unified"], 
        default="latin",
        help="Script type for Ottoman text (default: latin)"
    )
    parser.add_argument(
        "--model", 
        help="Custom model path or HuggingFace model name"
    )
    parser.add_argument(
        "--use-local", 
        action="store_true",
        help="Use local model instead of HuggingFace Hub"
    )
    
    # Processing options
    parser.add_argument(
        "--confidence", 
        action="store_true",
        help="Include confidence scores in output"
    )
    parser.add_argument(
        "--sentences", 
        action="store_true",
        help="Analyze text sentence by sentence"
    )
    parser.add_argument(
        "--aggregation", 
        choices=["simple", "first", "average", "max"],
        default="simple",
        help="Token aggregation strategy (default: simple)"
    )
    
    # Information options
    parser.add_argument(
        "--list-models", 
        action="store_true",
        help="List all available models"
    )
    parser.add_argument(
        "--model-info", 
        metavar="MODEL",
        help="Show information about a specific model"
    )
    
    # Other options
    parser.add_argument(
        "--encoding", 
        default="utf-8",
        help="Input file encoding (default: utf-8)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--version", 
        action="version",
        version="%(prog)s 0.2.0"
    )
    
    return parser

def list_models():
    """List all available models."""
    models_info = list_available_models()
    
    print("Available Ottoman NER Models:")
    print("=" * 40)
    
    print("\nRemote Models (HuggingFace Hub):")
    for name, path in models_info["remote_models"].items():
        info = models_info["model_info"].get(name, {})
        print(f"  {name:10} -> {path}")
        if info:
            print(f"             Description: {info.get('description', 'N/A')}")
            print(f"             Entities: {', '.join(info.get('entities', []))}")
    
    print("\nLocal Models:")
    for name, path in models_info["local_models"].items():
        print(f"  {name:10} -> {path}")

def show_model_info(model_name: str):
    """Show detailed information about a model."""
    models_info = list_available_models()
    
    if model_name not in models_info["model_info"]:
        print(f"Model '{model_name}' not found.")
        print(f"Available models: {', '.join(models_info['model_info'].keys())}")
        return
    
    info = models_info["model_info"][model_name]
    remote_path = models_info["remote_models"].get(model_name, "N/A")
    local_path = models_info["local_models"].get(model_name, "N/A")
    
    print(f"Model Information: {model_name}")
    print("=" * 40)
    print(f"Description: {info.get('description', 'N/A')}")
    print(f"Languages: {', '.join(info.get('languages', []))}")
    print(f"Entities: {', '.join(info.get('entities', []))}")
    print(f"Base Model: {info.get('base_model', 'N/A')}")
    print(f"Remote Path: {remote_path}")
    print(f"Local Path: {local_path}")

def format_output(entities, include_confidence=True, format_type="json"):
    """Format the output entities."""
    if format_type == "json":
        return json.dumps(entities, ensure_ascii=False, indent=2)
    elif format_type == "text":
        if not entities:
            return "No entities found."
        
        output = []
        if isinstance(entities, list):
            # Simple entity list
            for entity in entities:
                conf_str = f" (confidence: {entity.get('confidence', 0):.3f})" if include_confidence and 'confidence' in entity else ""
                output.append(f"{entity['text']} -> {entity['label']}{conf_str}")
        elif isinstance(entities, dict) and "sentences" in entities:
            # Sentence-level analysis
            for sent in entities["sentences"]:
                output.append(f"Sentence {sent['sentence_id']}: {sent['text']}")
                if sent["entities"]:
                    for entity in sent["entities"]:
                        conf_str = f" (confidence: {entity.get('confidence', 0):.3f})" if include_confidence and 'confidence' in entity else ""
                        output.append(f"  {entity['text']} -> {entity['label']}{conf_str}")
                else:
                    output.append("  No entities found.")
                output.append("")
        
        return "\n".join(output)

def main():
    """Main CLI function."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle information commands
    if args.list_models:
        list_models()
        return
    
    if args.model_info:
        show_model_info(args.model_info)
        return
    
    # Validate input
    if not args.text and not args.input:
        parser.error("You must provide either --text or --input")
    
    # Determine model to use
    if args.model:
        model_name_or_path = args.model
    else:
        model_name_or_path = args.script
    
    try:
        # Initialize predictor
        logger.info(f"Loading model: {model_name_or_path}")
        predictor = NERPredictor(
            model_name_or_path=model_name_or_path,
            aggregation_strategy=args.aggregation,
            use_local=args.use_local
        )
        
        # Get input text
        if args.text:
            input_text = args.text
        else:
            try:
                with open(args.input, 'r', encoding=args.encoding) as f:
                    input_text = f.read()
            except FileNotFoundError:
                logger.error(f"Input file not found: {args.input}")
                sys.exit(1)
            except UnicodeDecodeError:
                logger.error(f"Failed to decode file with encoding: {args.encoding}")
                sys.exit(1)
        
        # Perform prediction
        logger.info("Performing NER prediction...")
        if args.sentences:
            result = predictor.predict_sentences(input_text, return_confidence=args.confidence)
        else:
            result = predictor.predict(input_text, return_confidence=args.confidence)
        
        # Format and output results
        if args.output:
            # Save to file
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {args.output}")
        else:
            # Print to stdout
            if args.verbose:
                # Pretty text format for verbose mode
                print(format_output(result, args.confidence, "text"))
            else:
                # JSON format for normal mode
                print(format_output(result, args.confidence, "json"))
    
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
