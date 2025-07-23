# AO1 BigQuery Semantic Scanner v2.0

Advanced semantic discovery of AO1 log visibility data across 100+ BigQuery datasets using  AI intelligence.

## 🚀 New in v2.0: AI Intelligence

- **Foundation Models**: BERT, RoBERTa, T5, GPT-2, Sentence Transformers
- **Multi-Step Reasoning**: 8-layer analysis with contextual validation
- **Persistent Learning**: Builds knowledge from every scan
- **Advanced NLP**: WordNet, spaCy, NLTK with morphological analysis
- **Uncertainty Quantification**: Knows when it's uncertain
- **Explanation Generation**: Human-readable reasoning

## Features

- **Semantic Content Analysis**: Uses ML to understand field content regardless of naming conventions
- **Advanced Pattern Recognition**: Handles typos, variations, and complex semantic relationships
- **AO1-Optimized Discovery**: Specifically tuned for cybersecurity log visibility requirements
- **Automated Query Generation**: Creates validated queries based on discovered relationships
- **Persistent Memory**: Learns and improves from every scan
- **Multi-Modal Analysis**: Combines text, patterns, structure, and context

## Architecture

```
scanner/
├── __init__.py              # Package initialization
├── config.py               # Configuration and AO1 concepts
├── connection.py           # BigQuery authentication & connection
├── semantic_analyzer.py    # Main semantic analysis
├── query_generator.py      # AO1-specific query generation
├── data_validator.py       # Query validation and data quality
├── scanner.py             # Main orchestration
├── main.py               # CLI interface
├── nlp/                  # NLP foundation models
│   ├── __init__.py
│   ├── foundation.py     # BERT, RoBERTa, T5, GPT-2
│   ├── specialized.py    # NER, classification, QA
│   └── linguistic.py     # NLTK, spaCy, WordNet
├── semantic/             # Semantic intelligence
│   ├── __init__.py
│   ├── types.py         # Data structures
│   ├── embeddings.py    # Multi-modal embeddings
│   ├── reasoning.py     # Multi-step reasoning engine
│   └── memory.py        # Persistent learning memory
└── requirements.txt      # Dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLP models (first time only)
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
python -c "import spacy; spacy.cli.download('en_core_web_sm')"

# Place your gcp_prod_key.json file in the project directory

# Run the scanner
python run_ao1_scan.py

# Or use the module directly
python -m scanner.main -s gcp_prod_key.json
```

## What It Discovers

### Asset Identity Data
- Hostnames, device names, asset IDs with semantic understanding
- Network identifiers (IPs, FQDNs) with pattern recognition
- CMDB correlation keys with relationship mapping

### Security Tool Coverage
- CrowdStrike agent deployment with confidence scoring
- Chronicle log ingestion with contextual analysis
- Splunk data sources with cross-field validation
- EDR/endpoint security tools with domain expertise

### Log Source Analysis  
- Log types and sources with semantic classification
- Coverage gaps by system with reasoning explanations
- Volume and frequency patterns with uncertainty estimation

### Geographic Distribution
- Regional asset distribution with location intelligence
- Country/location analysis with entity recognition
- Site and datacenter mapping with relationship inference

## Example Output

```
AO1 SEMANTIC SCAN COMPLETE
=========================
Project: prj-fisv-p-gcss-sas-dl9dd0f1df
Datasets analyzed: 127
Tables analyzed: 1,847
Execution time: 185.4s

AO1-Relevant discoveries:
  Asset identity tables: 23
  Security tool tables: 31
  Log source tables: 67

Query generation:
  Total queries: 156
  Success rate: 96.8%

High-confidence queries (≥0.8): 89

Top AO1 queries:
  1. Asset Coverage Baseline - CSIRT.all_sources (confidence: 0.95) ★★★★★
  2. Tool Coverage Analysis - CrowdStrike (confidence: 0.92) ★★★★★
  3. Log Source Analysis - Chronicle Events (confidence: 0.91) ★★★★★

Reasoning Examples:
  Field 'hostname' → asset_identity (0.94)
    Multi-step reasoning: semantic_similarity(0.89) + contextual_validation(+0.15) + 
    cross-field_support(+0.12) - uncertainty(-0.08) = 0.94
```

## Command Line Options

```bash
python -m scanner.main [options]

Options:
  -s, --service-account PATH    Path to service account JSON file
  -o, --output-dir PATH        Output directory for results
  -q, --quick-scan            Quick scan mode (fewer samples)
  -w, --max-workers N         Maximum concurrent workers (default: 4)
  -v, --verbose               Verbose logging
  -c, --min-confidence N      Minimum confidence for best queries (default: 0.8)
```

## Advanced Intelligence Features

### 🧠 Multi-Step Reasoning
- **Step 1**: Initial concept ranking with semantic similarity
- **Step 2**: Contextual validation against table themes
- **Step 3**: Cross-field relationship analysis
- **Step 4**: Pattern consistency verification
- **Step 5**: Domain expertise validation
- **Step 6**: Uncertainty estimation and calibration

### 🎯 Semantic Understanding
- **WordNet Integration**: Automatic synonym expansion
- **Morphological Analysis**: Understands word structure (`devices` → `device`)
- **Typo Intelligence**: Handles `hosname` → `hostname`
- **Contextual Inference**: Table-level semantic themes
- **Entity Recognition**: Identifies organizations, locations, technical terms

### 💡 Persistent Learning
- **Field Pattern Memory**: Remembers successful field classifications
- **Value Signature Learning**: Clusters similar data patterns
- **Cross-Table Relationships**: Maps connections between datasets
- **Confidence Calibration**: Improves accuracy estimates over time
- **Error Correction**: Learns from feedback and mistakes

### 🔮 Uncertainty & Explanation
- **Confidence Intervals**: Quantifies prediction uncertainty
- **Alternative Classifications**: Shows runner-up categories
- **Reasoning Chains**: Explains decision process step-by-step
- **Evidence Attribution**: Links conclusions to specific evidence

## Performance Optimizations

- **Batch Processing**: Analyzes multiple fields simultaneously
- **Optimized Sampling**: Uses `TABLESAMPLE SYSTEM` for speed
- **Embedding Caching**: Reuses computed embeddings
- **Parallel Execution**: Concurrent dataset processing
- **Smart Limiting**: Focuses on most promising tables/fields

## Output Files

- `ao1_scan_results_TIMESTAMP.json`: Complete analysis with reasoning
- `ao1_queries_TIMESTAMP.sql`: Generated and validated SQL queries  
- `ao1_summary_TIMESTAMP.txt`: Human-readable summary and recommendations

## Model Files (Auto-Downloaded)
- `./ao1_models/concept_embeddings.pkl`: Multi-modal concept embeddings
- `./ao1_models/semantic_memory.pkl`: Persistent learning memory
- `./ao1_models/learned_patterns.pkl`: Field and value pattern memory

The scanner now reasons through multiple steps, considering context deeply, learning continuously, and explaining its decisions with human-level intelligence.