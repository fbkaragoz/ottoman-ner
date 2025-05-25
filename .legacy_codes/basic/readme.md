# Ruznamce Named Entity Recognition (NER) System

## Overview

This project builds a Named Entity Recognition (NER) system for Ottoman Turkish, using annotated sentences from various issues of the Servet-i Funun journal (1896-1901) and optionally raw texts from judicial Ruznamçe registers.

## Requirements

- Python 3.6+ ----> Python 3.9 env is used
- 
- `spacy` library
- `tqdm` library
- `python-docx` library
- `scikit-learn` library
- `pandas` library
- `matplotlib` library (for graph visualization)
- `seaborn` library (for visualization)

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_directory>
```
Required Packages to install for :
```
pip install spacy tqdm python-docx scikit-learn pandas matplotlib seaborn
```

Training and evaluating the NER model:
```
cd term/basic; python3 train_ner.py
```
Over 30 iteration provides sufficient result. Evaluation can be performed henceforth:
```
python3 evaluate_ner.py
```
For graphic representation of scores:
```
python3 visualize_p.py
```









