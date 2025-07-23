from setuptools import setup, find_packages

setup(
    name="ao1-scanner",
    version="2.0.0",
    description="AO1 BigQuery Semantic Dataset Discovery & Query Generator with Claude-Level Intelligence",
    packages=find_packages(),
    install_requires=[
        "google-cloud-bigquery>=3.0.0",
        "google-auth>=2.0.0", 
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "sentence-transformers>=2.2.0",
        "fuzzywuzzy>=0.18.0",
        "python-Levenshtein>=0.12.0",
        "duckdb>=0.8.0",
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "nltk>=3.8",
        "spacy>=3.4.0",
        "datasets>=2.0.0",
        "networkx>=2.8",
        "gensim>=4.2.0",
        "textstat>=0.7.0",
        "textblob>=0.17.0",
        "vaderSentiment>=3.3.0",
        "umap-learn>=0.5.0",
        "scipy>=1.9.0"
    ],
    entry_points={
        'console_scripts': [
            'ao1-scan=scanner.main:main',
        ],
    },
    python_requires=">=3.8",
)