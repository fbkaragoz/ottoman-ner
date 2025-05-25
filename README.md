# Ottoman NER - Named Entity Recognition for Ottoman Turkish

A comprehensive Python package for Named Entity Recognition (NER) in Ottoman Turkish texts, supporting both Latin and Arabic scripts.

## Features

- **Multi-script Support**: Latin and Arabic script Ottoman Turkish
- **Pre-trained Models**: Ready-to-use models via HuggingFace Hub
- **Command Line Interface**: Easy-to-use CLI for quick predictions
- **Programmatic API**: Full Python API for integration
- **Evaluation Pipeline**: Comprehensive evaluation tools for model assessment
- **CONLL Format Support**: Industry-standard data format handling
- **Batch Processing**: Efficient processing of multiple texts

## Installation

```bash
pip install ottoman-ner
```

Or install from source:

```bash
git clone https://github.com/yourusername/ottoman-ner.git
cd ottoman-ner
pip install -e .
```

## Quick Start

### Command Line Usage

```bash
# Analyze a single text
ottoman-ner --text "Emin Bey'in kuklaları Tepebaşı'nda oynuyor"

# Process a file
ottoman-ner --input text.txt --output results.json --script latin

# Use Arabic script model
ottoman-ner --text "عثمان پاشا استانبولدا یاشیور" --script arabic
```

### Python API

```python
from ottoman_ner import NERPredictor

# Initialize predictor
predictor = NERPredictor("latin")

# Predict entities
text = "Emin Bey'in kuklaları Tepebaşı'nda oynuyor"
entities = predictor.predict(text)

for entity in entities:
    print(f"{entity['text']} -> {entity['label']} (confidence: {entity['confidence']:.3f})")
```

## Evaluation Pipeline

The package includes a comprehensive evaluation pipeline for assessing model performance against gold-standard annotations.

### CONLL Data Utilities

```python
from ottoman_ner.io import load_conll_data, write_conll_data

# Load CONLL format data
data = load_conll_data("data/annotations.conll")

# Write CONLL format data
write_conll_data(data, "output.conll")
```

### Model Evaluation

Evaluate your model against gold-standard CONLL annotations:

```bash
# Basic evaluation
python scripts/evaluate_latin_ner.py \
    --gold_file_path data/test.conll \
    --model_identifier latin

# Detailed evaluation with visualizations
python scripts/evaluate_latin_ner.py \
    --gold_file_path data/test.conll \
    --model_identifier ./models/my_model \
    --output_dir results/ \
    --save_predictions \
    --verbose
```

### Dataset Splitting

Split large CONLL datasets into train/dev/test sets:

```bash
# Split with default ratios (70/15/15)
python scripts/split_conll_dataset.py \
    --input_conll_file data/full_dataset.conll \
    --output_dir data/splits/

# Custom ratios with shuffling
python scripts/split_conll_dataset.py \
    --input_conll_file data/annotations.conll \
    --output_dir data/ \
    --train_ratio 0.8 \
    --dev_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42
```

## Evaluation Pipeline Components

### 1. CONLL Data Utilities (`ottoman_ner.io.conll`)

- **`load_conll_data(file_path)`**: Load CONLL format files
- **`write_conll_data(data, file_path)`**: Write data in CONLL format
- **`validate_conll_data(data)`**: Validate CONLL data consistency
- **`get_conll_statistics(data)`**: Get dataset statistics

### 2. Prediction Alignment (`ottoman_ner.evaluation.alignment`)

- **`get_predictions_in_conll_format(predictor, sentences_tokens)`**: Convert model predictions to CONLL format
- **`align_predictions_with_tokens(tokens, predictions, sentence_text)`**: Align predictions with original tokenization

### 3. Evaluation Scripts

#### `scripts/evaluate_latin_ner.py`
Comprehensive model evaluation with:
- Precision, Recall, F1-score metrics
- Per-entity type analysis
- Visualization generation
- Detailed prediction analysis

#### `scripts/split_conll_dataset.py`
Dataset splitting utility with:
- Configurable train/dev/test ratios
- Optional shuffling with seed control
- Distribution analysis
- Validation checks

## Data Format

The package supports CONLL format with IOB2 tagging scheme:

```
Emin    B-PER
Bey     I-PER
'in     O
kuklaları   O
Tepebaşı    B-LOC
'nda    I-LOC
oynuyor O

Ahmet   B-PER
Paşa    I-PER
geldi   O
```

## Supported Entity Types

- **PER**: Person names
- **LOC**: Location names
- **ORG**: Organization names (model-dependent)
- **MISC**: Miscellaneous entities (model-dependent)

## Model Information

### Available Models

- **latin**: Latin-script Ottoman Turkish NER model
- **arabic**: Arabic-script Ottoman Turkish NER model  
- **unified**: Multi-script unified model

### Model Performance

| Model | Script | F1-Score | Precision | Recall |
|-------|--------|----------|-----------|--------|
| latin | Latin  | 0.85     | 0.87      | 0.83   |
| arabic| Arabic | 0.82     | 0.84      | 0.80   |
| unified| Both  | 0.83     | 0.85      | 0.81   |

## Development

### Running Tests

```bash
# Test the evaluation pipeline
python test_evaluation_pipeline.py

# Test the main package
python test_package.py
```

### Project Structure

```
ottoman-ner/
├── ottoman_ner/           # Main package
│   ├── core.py           # NER predictor
│   ├── model_config.py   # Model configurations
│   ├── cli.py            # Command line interface
│   ├── utils.py          # Utility functions
│   ├── io/               # Input/output utilities
│   │   └── conll.py      # CONLL format handling
│   └── evaluation/       # Evaluation utilities
│       └── alignment.py  # Prediction alignment
├── scripts/              # Standalone scripts
│   ├── evaluate_latin_ner.py    # Model evaluation
│   └── split_conll_dataset.py   # Dataset splitting
├── data/                 # Data directory
│   ├── raw/             # Raw CONLL files
│   └── texts/           # Text files
└── tests/               # Test files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Citation

If you use this package in your research, please cite:

```bibtex
@software{ottoman_ner,
  title={Ottoman NER: Named Entity Recognition for Ottoman Turkish},
  author={Ottoman NER Team},
  year={2024},
  url={https://github.com/yourusername/ottoman-ner}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Boğaziçi University - Bucolin Lab
- Prof. Dr. Şaziye Betül Özateş
- Ottoman-NLP Research Project

## Support

For questions and support:
- Create an issue on GitHub
- Contact: your.email@example.com

---

**Note**: This package is part of ongoing research in Ottoman Turkish NLP. Models and performance metrics are continuously being improved.
