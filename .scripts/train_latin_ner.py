import os
import logging
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
import evaluate
import numpy as np
from sklearn.metrics import classification_report
from parse_conll_to_dataset import load_dataset_from_conll, create_label_mappings, encode_labels

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OttomanNERTrainer:
    def __init__(self, data_dir: str, model_checkpoint: str = "dbmdz/bert-base-turkish-cased"):
        self.data_dir = data_dir
        self.model_checkpoint = model_checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load and prepare dataset
        self.dataset = load_dataset_from_conll(data_dir)
        self.label2id, self.id2label = create_label_mappings(self.dataset, data_dir)
        self.num_labels = len(self.label2id)
        
        logger.info(f"Dataset loaded: {self.dataset}")
        logger.info(f"Number of labels: {self.num_labels}")
        logger.info(f"Labels: {list(self.label2id.keys())}")
        
        # Encode labels
        self.dataset = self.dataset.map(lambda x: encode_labels(x, self.label2id))
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
        # Tokenize dataset and remove unnecessary columns
        self.tokenized_datasets = self.dataset.map(
            self.tokenize_and_align_labels, 
            batched=False,
            desc="Tokenizing",
            remove_columns=self.dataset["train"].column_names  # Remove all original columns
        )
        
        # Load metrics
        self.metric = evaluate.load("seqeval")
        
    def tokenize_and_align_labels(self, example):
        """Tokenize and align labels with subword tokens."""
        tokenized_inputs = self.tokenizer(
            example["tokens"], 
            truncation=True, 
            is_split_into_words=True,
            max_length=512,
            padding=False  # Don't pad here, let DataCollator handle it
        )
        
        labels = []
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 label
                labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the label
                try:
                    labels.append(example["labels"][word_idx])
                except IndexError:
                    # Handle edge case where word_ids might be out of bounds
                    labels.append(-100)
            else:
                # Other subwords get -100 (ignore in loss calculation)
                labels.append(-100)
            previous_word_idx = word_idx
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[str(p)] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[str(l)] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Compute seqeval metrics
        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def create_model(self):
        """Create and configure the model."""
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        return model.to(self.device)
    
    def get_training_args(self, output_dir: str = "./models/ottoman_ner_latin"):
        """Get training arguments with advanced configuration."""
        return TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",  # Updated parameter name
            save_strategy="epoch",
            logging_dir="./logs",
            learning_rate=2e-5,
            per_device_train_batch_size=8,  # Reduced batch size for stability
            per_device_eval_batch_size=8,
            num_train_epochs=5,  # Reduced epochs for faster testing
            weight_decay=0.01,
            logging_steps=50,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            gradient_accumulation_steps=1,
            dataloader_num_workers=0,  # Disable multiprocessing to avoid tokenizer warnings
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
            seed=42,
            data_seed=42,
        )
    
    def train(self, output_dir: str = "./models/ottoman_ner_latin"):
        """Train the model with advanced configuration."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        
        # Create model
        model = self.create_model()
        
        # Get training arguments
        training_args = self.get_training_args(output_dir)
        
        # Create data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=512,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
        
        # Create trainer with early stopping
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["dev"],
            processing_class=self.tokenizer,  # Use processing_class instead of tokenizer
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        trainer.save_state()
        
        # Log training results
        logger.info(f"Training completed!")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=self.tokenized_datasets["test"])
        logger.info(f"Test results: {test_results}")
        
        # Save label mappings
        import json
        with open(f"{output_dir}/label_mappings.json", "w") as f:
            json.dump({
                "label2id": self.label2id,
                "id2label": self.id2label
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model and mappings saved to {output_dir}")
        return trainer, test_results

def main():
    """Main training function."""
    # Set environment variable to avoid tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    trainer_obj = OttomanNERTrainer("data/raw")
    trainer, results = trainer_obj.train()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Final Test Results:")
    for key, value in results.items():
        if key.startswith("eval_"):
            print(f"  {key.replace('eval_', '').upper()}: {value:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()
