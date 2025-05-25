## Ottoman Turkish Named Entity Recognition (Ottoman-NER)

Ottoman-NER is a Python library and command-line tool for identifying named entities (such as persons, locations, organizations) in Ottoman Turkish texts. It supports both texts written in the original Perso-Arabic script and texts transliterated into the Latin alphabet.

This project is developed by Fatih Burak Karagoz and has been worked on in collaboration with Boğaziçi University - Bucolin Lab under the supervision of Prof. Dr. Şaziye Betül Özateş. It is part of the broader research initiative Ottoman-NLP.

### Features

- **Dual Script Support**: Accurately processes Ottoman Turkish in both Perso-Arabic script (e.g., استانبول) and Latin-based transliterations (e.g., İstanbul).
- State-of-the-art NER models for Ottoman Turkish.
- Easy-to-use Python API.
- Convenient Command Line Interface (CLI) for quick processing.
- Integration with Hugging Face Hub for model sharing and usage.
- Built upon robust libraries like Transformers and PyTorch.

### Installation

You can install ottoman-ner directly from PyPI:

```bash
pip install ottoman-ner
```

To install from source for development:

```bash
git clone https://github.com/fatihburak/ottoman-ner.git
cd ottoman-ner
pip install -e .[dev]
```

### Usage

#### Command Line Interface (CLI)

```bash
# For Perso-Arabic script
ottoman-ner --input path/to/your/arabic_script_document.txt --output results_arabic.json --script arabic
ottoman-ner --text "بورسه ده اسکی بر جامعك قربنده اوتورور." --script arabic

# For Latin script
ottoman-ner --input path/to/your/latin_script_document.txt --output results_latin.json --script latin
ottoman-ner --text "Bursa'da eski bir cami'in kurbunda oturur." --script latin
```

See all options:
```bash
ottoman-ner --help
```

#### Python API

```python
from ottoman_ner import NERPredictor

predictor_arabic = NERPredictor(model_name_or_path="fatihburakkaragoz/ottoman-ner-arabic")
predictor_latin = NERPredictor(model_name_or_path="fatihburakkaragoz/ottoman-ner-latin")

text_arabic = "بورسه شهری تاریخی و زیبا است."
entities_arabic = predictor_arabic.predict(text_arabic)

text_latin = "Bursa tarihi ve güzel bir şehirdir."
entities_latin = predictor_latin.predict(text_latin)
```

### 🤗 Hugging Face Models

Models hosted on Hugging Face:

- `fatihburakkaragoz/ottoman-ner-arabic`
- `fatihburakkaragoz/ottoman-ner-latin`
- `fatihburakkaragoz/ottoman-ner-unified`

Example usage:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("fatihburakkaragoz/ottoman-ner-latin")
model = AutoModelForTokenClassification.from_pretrained("fatihburakkaragoz/ottoman-ner-latin")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
```

### Datasets

Dataset documentation and download links will be available on the Hugging Face model page.

### Contributing

Contributions are highly welcome! Please read our CONTRIBUTING.md for details.

### License

This project is licensed under the MIT License.

### Citation

```bibtex
@software{Karagoz_Ottoman_NER_2025,
  author = {Karagoz, Fatih Burak and Özateş, Şaziye Betül},
  title = {{Ottoman-NER: A Tool for Named Entity Recognition in Ottoman Turkish Texts}},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fbkaragoz/ottoman-ner}},
  note = {Version 0.1.0}
}
```

### Acknowledgements

- Prof. Dr. Şaziye Betül Özateş
- Boğaziçi University - Bucolin Lab
- Ottoman-NLP community
- All contributors and users

### Contact

Fatih Burak Karagoz  
Email: fatihburak@pm.me  
GitHub: https://github.com/fbkaragoz/ottoman-ner
