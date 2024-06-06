from setuptools import setup, find_packages

setup(
    name='nerpackage',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'datasets',
        'seqeval',
        'docx',  #python-docx for document handling
        'torch',
        'pandas',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'ner_train=nerpackage.model:main',
        ],
    },
    description='Named Entity Recognition package for historical texts',
    author='fatih karagöz',
    author_email='fatih.karagoz@std.bogazici.edu.tr'
)