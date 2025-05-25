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

setup(
    name='ottoman-ner',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'torch',
        'seqeval',
        'tqdm',
        'scikit-learn',
        'pandas',
        'numpy',
        'scipy',
        'nltk',
        'spacy',
        'flair'
    ],
    entry_points={
        'console_scripts': [
            'ottoman-ner=ottoman_ner.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    author='Fatih Burak Karagoz',
    author_email='fatihburak@pm.me',
    description='Ottoman Named Entity Recognition',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fbkaragoz/ottoman-ner',
    hf_repo_id='fatihburakkaragoz/ottoman-ner',
    project_urls={
        'Bug Tracker': 'https://github.com/fbkaragoz/ottoman-ner/issues',
    },
    keywords='named-entity-recognition',
    license='MIT',
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    test_suite='tests',
    tests_require=['pytest'],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'pytest-mock',
        ],
    },
)