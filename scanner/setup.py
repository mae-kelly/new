from setuptools import setup, find_packages

setup(
    name="ao1-scanner",
    version="1.0.0",
    description="AO1 BigQuery Semantic Dataset Discovery & Query Generator",
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
        "duckdb>=0.8.0"
    ],
    entry_points={
        'console_scripts': [
            'ao1-scan=ao1_scanner.main:main',
        ],
    },
    python_requires=">=3.8",
)