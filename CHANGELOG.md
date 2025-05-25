# Changelog

## [0.2.0] - 2025-05-25

### Added
- Initial Latin script NER model training script (`train_latin_ner.py`)
- CONLL parsing script (`parse_conll_to_dataset.py`)
- New folder structure for scripts and model artifacts
- Hugging Face model upload logic

### Changed
- Project cleaned for modern modular structure
- All legacy code moved to `.legacy_codes/` directory

### Deprecated
- Original flat script implementations are deprecated but retained under `.legacy_codes`

### Upcoming
- Arabic script model integration planned for 0.3.0
