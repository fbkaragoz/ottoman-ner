# Contributing to Ottoman Turkish NER (Ottoman-NER)

First off, thank you for considering contributing to `ottoman-ner`! Your efforts help make this tool better for everyone. This project aims to be a collaborative effort, and we welcome contributions from an open and respectful community.

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold these standards.

---

## How Can I Contribute?

There are many ways to contribute:

- Reporting bugs 🐛
- Suggesting features ✨
- Writing code or documentation 📝
- Improving test coverage ✅

---

## Reporting Bugs

If you discover a bug, first check [GitHub Issues](https://github.com/fbkaragoz/ottoman-ner/issues) to see if it's already reported.

If not, open a new issue and include:

- ottoman-ner version (`pip show ottoman-ner`)
- Python version (`python --version`)
- OS (e.g., Ubuntu 22.04, macOS, Windows)
- Script type (Arabic or Latin)
- Steps to reproduce
- Expected vs actual behavior

---

## Suggesting Enhancements

To suggest a feature or improvement:

1. Open a GitHub issue
2. Describe the enhancement and its benefits
3. If possible, include mockups, examples, or code

---

## Your First Code Contribution

Look at the issue tracker for labels like:

- `good first issue`
- `help wanted`
- `documentation`

---

## Pull Request Process

### Create a branch

```bash
git checkout -b feat/your-feature-name
```

### Set up your development environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

### Make changes

- Support both Perso-Arabic and Latin scripts if applicable
- Follow code style (Black, Flake8)

### Run tests

```bash
pytest tests/
```

### Format & lint

```bash
black .
flake8 .
```

### Commit

```bash
git add .
git commit -m "feat: short description of feature"
```

### Push

```bash
git push origin feat/your-feature-name
```

### Open a Pull Request (PR)

- Go to the original repo → "New pull request"
- Link to related issues (e.g., `Closes #123`)
- Explain your change clearly
- Ensure all checks pass

---

## Documentation

- Update `README.md` and usage examples
- Use Google or NumPy docstring style

---

## Code Style Guide

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Format code with [Black](https://github.com/psf/black)
- Lint with [Flake8](https://flake8.pycqa.org/)
- Use type hints (PEP 484)
- Write tests using `pytest`

---

## Development Setup

```bash
git clone https://github.com/fbkaragoz/ottoman-ner.git
cd ottoman-ner
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

---

## Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=ottoman_ner --cov-report=html
```

Open `htmlcov/index.html` in your browser to view.

---

## Pre-Commit Hooks (Recommended)

Install and configure `pre-commit`:

```bash
pip install pre-commit
pre-commit install
```

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

---

## Questions?

Open an issue or [contact the maintainers](mailto:fatihburak@pm.me).

Thanks again for helping advance **Ottoman Turkish NLP**! 🇹🇷📚🧠
