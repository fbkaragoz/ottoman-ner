#! /usr/bin/env python3

'''
This is a setup.py file for the Ottoman Named Entity Recognition project. 
I have worked on the project along with Boğaziçi University - Bucolin Lab 
under the supervision of Prof. Dr. Şaziye Betül Özateş.

The project is a part of the research project "https://github.com/Ottoman-NLP"
and the code is licensed under the MIT License.

The project is developed by Fatih Burak Karagoz.

It is open source and free to use. 

If you have any questions, please contact me at fatihburak@pm.me
'''

from setuptools import setup, find_packages

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ottoman-ner",
    version="0.2.0",
    author="Ottoman NER Team",
    author_email="your.email@example.com",
    description="Named Entity Recognition for Ottoman Turkish texts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ottoman-ner",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Researchers",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ottoman-ner=ottoman_ner.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ottoman_ner": ["*.json", "*.txt"],
    },
    keywords=[
        "nlp",
        "ner",
        "named-entity-recognition", 
        "ottoman-turkish",
        "historical-texts",
        "transformers",
        "bert",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ottoman-ner/issues",
        "Source": "https://github.com/yourusername/ottoman-ner",
        "Documentation": "https://ottoman-ner.readthedocs.io/",
    },
)