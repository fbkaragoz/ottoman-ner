import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OttomanNERInference:
    def __init__(self, model_path: str):
        """Initialize the Ottoman NER inference pipeline."""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Load label mappings
        try:
            with open(f"{model_path}/label_mappings.json", "r") as f:
                mappings = json.load(f)
                self.label2id = mappings["label2id"]
                self.id2label = mappings["id2label"]
        except FileNotFoundError:
            logger.warning("Label mappings not found, using model config")
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
        
        # Create pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Available labels: {list(self.label2id.keys())}")
    
    def predict(self, text: str, return_confidence: bool = True):
        """Predict named entities in the given text."""
        if isinstance(text, str):
            # Split text into tokens (simple whitespace splitting)
            tokens = text.split()
        else:
            tokens = text
        
        # Use pipeline for prediction
        results = self.ner_pipeline(tokens)
        
        # Format results
        entities = []
        for result in results:
            entity = {
                "text": result["word"],
                "label": result["entity_group"],
                "start": result.get("start", 0),
                "end": result.get("end", 0),
            }
            if return_confidence:
                entity["confidence"] = result["score"]
            entities.append(entity)
        
        return entities
    
    def predict_sentence(self, sentence: str):
        """Predict entities in a sentence and return formatted output."""
        tokens = sentence.split()
        entities = self.predict(tokens)
        
        # Create token-level predictions
        token_predictions = []
        entity_map = {ent["text"]: ent["label"] for ent in entities}
        
        for token in tokens:
            label = entity_map.get(token, "O")
            token_predictions.append((token, label))
        
        return token_predictions, entities
    
    def batch_predict(self, texts: list):
        """Predict entities for multiple texts."""
        results = []
        for text in texts:
            entities = self.predict(text)
            results.append(entities)
        return results
    
    def print_predictions(self, sentence: str):
        """Print predictions in a nice format."""
        token_preds, entities = self.predict_sentence(sentence)
        
        print(f"\nInput: {sentence}")
        print("-" * 50)
        print("Token-level predictions:")
        for token, label in token_preds:
            print(f"  {token:<15} -> {label}")
        
        print("\nDetected entities:")
        if entities:
            for ent in entities:
                conf_str = f" (confidence: {ent.get('confidence', 0):.3f})" if 'confidence' in ent else ""
                print(f"  {ent['text']:<15} -> {ent['label']}{conf_str}")
        else:
            print("  No entities detected")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Ottoman NER Inference")
    parser.add_argument("--model_path", type=str, default="./models/ottoman_ner_latin",
                       help="Path to the trained model")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--file", type=str, help="File containing texts to analyze")
    
    args = parser.parse_args()
    
    # Initialize inference
    try:
        ner_model = OttomanNERInference(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    if args.interactive:
        # Interactive mode
        print("Ottoman NER Interactive Mode")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            text = input("\nEnter text: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text:
                ner_model.print_predictions(text)
    
    elif args.text:
        # Single text prediction
        ner_model.print_predictions(args.text)
    
    elif args.file:
        # File processing
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line:
                    print(f"\n--- Line {i} ---")
                    ner_model.print_predictions(line)
        except FileNotFoundError:
            logger.error(f"File not found: {args.file}")
    
    else:
        # Default examples
        examples = [
            "Emin Bey'in kuklaları bir haftadır Tepebaşı'nda oynuyor.",
            "Viktor Ügo'ya hasrettiği bapta ben demiyorum ki Natüralizm ebediyen baki kalacaktır.",
            "Paris halkının kısm-ı azimi hab-ı istirahate çekilir.",
            "Babıali Caddesinde Kütüphane-i İslamda bulunur."
        ]
        
        print("Running with example sentences:")
        for example in examples:
            ner_model.print_predictions(example)

if __name__ == "__main__":
    main() 