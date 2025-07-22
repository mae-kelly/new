# AO1 BigQuery Semantic Scanner

Intelligent discovery of AO1 log visibility data across 100+ BigQuery datasets using semantic machine learning.

## Features

- **Semantic Content Analysis**: Uses ML to understand field content regardless of naming conventions
- **Fuzzy Pattern Matching**: Handles typos and atypical naming patterns
- **AO1-Optimized Discovery**: Specifically tuned for cybersecurity log visibility requirements
- **Automated Query Generation**: Creates validated queries based on discovered relationships
- **Corporate Network Compatible**: Uses your existing BigQuery authentication patterns

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Place your gcp_prod_key.json file in the project directory

# Run the scanner
python run_ao1_scan.py

# Or use the module directly
python -m ao1_scanner.main -s gcp_prod_key.json
```

## What It Discovers

### Asset Identity Data
- Hostnames, device names, asset IDs
- Network identifiers (IPs, FQDNs)
- CMDB correlation keys

### Security Tool Coverage
- CrowdStrike agent deployment
- Chronicle log ingestion
- Splunk data sources
- EDR/endpoint security tools

### Log Source Analysis  
- Log types and sources
- Coverage gaps by system
- Volume and frequency patterns

### Geographic Distribution
- Regional asset distribution
- Country/location analysis
- Site and datacenter mapping

## Example Output

```
AO1 SEMANTIC SCAN COMPLETE
=========================
Project: prj-fisv-p-gcss-sas-dl9dd0f1df
Datasets analyzed: 127
Tables analyzed: 1,847
Execution time: 285.4s

AO1-Relevant discoveries:
  Asset identity tables: 23
  Security tool tables: 31
  Log source tables: 67

Query generation:
  Total queries: 156
  Success rate: 94.2%

High-confidence queries (≥0.8): 89

Top AO1 queries:
  1. Asset Coverage Baseline - CSIRT.all_sources (confidence: 0.95)
  2. Tool Coverage Analysis - CrowdStrike (confidence: 0.92)
  3. Log Source Analysis - Chronicle Events (confidence: 0.91)
```

## Authentication

The scanner uses the exact same authentication pattern as your Flask application:

```python
file_path = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_FILE = os.path.join(file_path, "gcp_prod_key.json")
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
client = bigquery.Client(project="prj-fisv-p-gcss-sas-dl9dd0f1df", credentials=credentials)
```

This ensures compatibility with your corporate proxy setup.

## Configuration

Edit `ao1_scanner/config.py` to customize:
- AO1 concept definitions
- Semantic similarity thresholds
- Query templates
- Output formats

## Architecture

- **connection.py**: BigQuery authentication using your existing patterns
- **semantic_analyzer.py**: ML-powered content analysis and field classification
- **query_generator.py**: AO1-specific query generation based on discovered patterns
- **data_validator.py**: Query validation and data quality assessment
- **scanner.py**: Main orchestration and result generation

## Output Files

- `ao1_scan_results_TIMESTAMP.json`: Complete semantic analysis results
- `ao1_queries_TIMESTAMP.sql`: Generated and validated SQL queries  
- `ao1_summary_TIMESTAMP.txt`: Human-readable summary and recommendations

## Command Line Options

```bash
python -m ao1_scanner.main [options]

Options:
  -s, --service-account PATH    Path to service account JSON file
  -o, --output-dir PATH        Output directory for results
  -q, --quick-scan            Quick scan mode (fewer samples)
  -w, --max-workers N         Maximum concurrent workers (default: 4)
  -v, --verbose               Verbose logging
  -c, --min-confidence N      Minimum confidence for best queries (default: 0.8)
```