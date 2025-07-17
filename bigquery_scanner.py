import os
import sys
import json
import pandas as pd
from google.cloud import bigquery
from google.cloud import resourcemanager_v3
from datetime import datetime
import time
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigQueryScanner:
    def __init__(self):
        try:
            self.bq_client = bigquery.Client()
            self.resource_client = resourcemanager_v3.ProjectsClient()
            logger.info("BigQuery client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            logger.error("Set GOOGLE_APPLICATION_CREDENTIALS to your service account key file")
            sys.exit(1)
            
        try:
            from neural_engine_brilliant import BrilliantVisibilityMapper
            self.ml_mapper = BrilliantVisibilityMapper()
            logger.info("Using BrilliantVisibilityMapper")
        except ImportError:
            logger.info("Using built-in pattern matching")
            self.ml_mapper = None
        
    def classify_column(self, data_series, column_name):
        clean_data = data_series.dropna().astype(str).head(100).tolist()
        if not clean_data:
            return None
            
        patterns = {
            'host_coverage': [
                lambda x: bool(re.search(r'[a-zA-Z]+-[a-zA-Z]+-\d+', str(x))),
                lambda x: bool(re.search(r'srv|host|server|node|machine', str(x).lower())),
                lambda x: bool(re.search(r'[A-Z]{3,}-[A-Z]+-\d+', str(x))),
                lambda x: 'host' in column_name.lower() or 'hostname' in column_name.lower()
            ],
            'network_coverage': [
                lambda x: bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', str(x))),
                lambda x: bool(re.search(r'eth0|ens3|bond0|wifi', str(x).lower())),
                lambda x: ':' in str(x) and any(c.isdigit() for c in str(x)),
                lambda x: 'ip' in column_name.lower() or 'addr' in column_name.lower()
            ],
            'security_coverage': [
                lambda x: bool(re.search(r'agent|defender|security|antivirus|falcon|tanium', str(x).lower())),
                lambda x: bool(re.search(r'v\d+\.\d+', str(x))),
                lambda x: 'agent' in column_name.lower() or 'security' in column_name.lower()
            ],
            'platform_coverage': [
                lambda x: bool(re.search(r'windows|linux|ubuntu|darwin|macos|rhel|centos', str(x).lower())),
                lambda x: bool(re.search(r'build|version|os', str(x).lower())),
                lambda x: 'os' in column_name.lower() or 'platform' in column_name.lower()
            ],
            'domain_coverage': [
                lambda x: bool(re.search(r'\.[a-z]{2,4}$', str(x).lower())),
                lambda x: bool(re.search(r'\.local|\.corp|\.internal', str(x).lower())),
                lambda x: 'domain' in column_name.lower() or 'url' in column_name.lower()
            ]
        }
        
        for metric, checks in patterns.items():
            score = 0
            for check in checks:
                try:
                    matches = sum(1 for sample in clean_data[:50] if check(sample))
                    if matches > len(clean_data) * 0.1:
                        score += 1
                except:
                    pass
            
            if score >= 2:
                return {
                    'metric': metric,
                    'confidence': min(score / len(checks), 1.0),
                    'samples': clean_data[:3],
                    'pattern_type': 'builtin_pattern'
                }
        
        return None
        
    def get_current_project(self):
        try:
            return self.bq_client.project
        except:
            return None
            
    def get_datasets_for_project(self, project_id):
        try:
            client = bigquery.Client(project=project_id)
            datasets = list(client.list_datasets())
            dataset_ids = [dataset.dataset_id for dataset in datasets]
            logger.info(f"Project {project_id}: Found {len(dataset_ids)} datasets")
            return dataset_ids
        except Exception as e:
            logger.error(f"Error getting datasets for project {project_id}: {e}")
            return []
    
    def get_tables_for_dataset(self, project_id, dataset_id):
        try:
            client = bigquery.Client(project=project_id)
            tables = list(client.list_tables(dataset_id))
            table_info = []
            for table in tables:
                table_info.append({
                    'project': project_id,
                    'dataset': dataset_id,
                    'table': table.table_id,
                    'full_name': f"{project_id}.{dataset_id}.{table.table_id}"
                })
            logger.info(f"Dataset {project_id}.{dataset_id}: Found {len(table_info)} tables")
            return table_info
        except Exception as e:
            logger.error(f"Error getting tables for dataset {project_id}.{dataset_id}: {e}")
            return []
    
    def sample_table_data(self, project_id, dataset_id, table_id):
        try:
            client = bigquery.Client(project=project_id)
            
            sample_query = f"""
            SELECT * FROM `{project_id}.{dataset_id}.{table_id}`
            LIMIT 200
            """
            
            query_job = client.query(sample_query)
            df = query_job.to_dataframe()
            
            if df.empty:
                logger.info(f"No data returned for table {table_id}")
                return None
            
            logger.info(f"Sampled {len(df)} rows from {table_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error sampling table {project_id}.{dataset_id}.{table_id}: {e}")
            return None
    
    def analyze_table(self, table_info):
        try:
            project_id = table_info['project']
            dataset_id = table_info['dataset']
            table_id = table_info['table']
            
            logger.info(f"Analyzing {project_id}.{dataset_id}.{table_id}")
            
            df = self.sample_table_data(project_id, dataset_id, table_id)
            if df is None:
                return []
            
            mappings = []
            
            if self.ml_mapper:
                try:
                    tables_dict = {table_id: df}
                    ml_mappings = self.ml_mapper.discover_mappings_with_brilliance(tables_dict)
                    
                    for mapping in ml_mappings:
                        mappings.append({
                            'project': project_id,
                            'dataset': dataset_id,
                            'table': table_id,
                            'column': mapping.column_name,
                            'source': f"{project_id}.{dataset_id}.{table_id}.{mapping.column_name}",
                            'metric': mapping.target_metric,
                            'confidence': mapping.entanglement_strength,
                            'pattern_type': mapping.reasoning_graph.get('pattern_type', 'ml_detected'),
                            'detection_methods': mapping.reasoning_graph.get('detection_methods', []),
                            'sample_values': mapping.reasoning_graph.get('sample_values', []),
                            'full_table_name': table_info['full_name']
                        })
                except Exception as ml_error:
                    logger.error(f"ML analysis failed: {ml_error}")
            
            if not mappings:
                for column_name in df.columns:
                    result = self.classify_column(df[column_name], column_name)
                    if result:
                        mappings.append({
                            'project': project_id,
                            'dataset': dataset_id,
                            'table': table_id,
                            'column': column_name,
                            'source': f"{project_id}.{dataset_id}.{table_id}.{column_name}",
                            'metric': result['metric'],
                            'confidence': result['confidence'],
                            'pattern_type': result['pattern_type'],
                            'detection_methods': ['builtin_classification'],
                            'sample_values': result['samples'],
                            'full_table_name': table_info['full_name']
                        })
            
            logger.info(f"Found {len(mappings)} mappings for {table_id}")
            return mappings
            
        except Exception as e:
            logger.error(f"Error analyzing table {table_info}: {e}")
            return []
    
    def scan_bigquery_project(self, project_id=None):
        if project_id is None:
            project_id = self.get_current_project()
            if not project_id:
                logger.error("No project ID provided and couldn't detect current project")
                return []
        
        logger.info(f"Starting BigQuery scan for project: {project_id}")
        start_time = time.time()
        
        all_mappings = []
        datasets = self.get_datasets_for_project(project_id)
        
        for dataset_id in datasets:
            try:
                tables = self.get_tables_for_dataset(project_id, dataset_id)
                
                for table_info in tables:
                    mappings = self.analyze_table(table_info)
                    all_mappings.extend(mappings)
            
            except Exception as dataset_error:
                logger.error(f"Error processing dataset {dataset_id}: {dataset_error}")
                continue
        
        end_time = time.time()
        logger.info(f"Scan completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Total mappings found: {len(all_mappings)}")
        
        return all_mappings
    
    def save_results(self, mappings, project_id):
        os.makedirs('outputs', exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        high_confidence_mappings = [m for m in mappings if m['confidence'] > 0.7]
        
        report = {
            'scan_metadata': {
                'timestamp': timestamp,
                'project_id': project_id,
                'total_mappings': len(mappings),
                'high_confidence_mappings': len(high_confidence_mappings),
                'datasets_scanned': len(set(f"{m['dataset']}" for m in mappings)),
                'tables_scanned': len(set(f"{m['dataset']}.{m['table']}" for m in mappings)),
                'ml_system': 'BrilliantVisibilityMapper' if self.ml_mapper else 'BuiltinPatternMatcher'
            },
            'mappings': sorted(mappings, key=lambda x: x['confidence'], reverse=True)
        }
        
        filename = f'outputs/bigquery_scan_{project_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        
        self.print_summary(report)
    
    def print_summary(self, report):
        print("\n" + "="*80)
        print("BIGQUERY SCAN RESULTS")
        print("="*80)
        
        metadata = report['scan_metadata']
        print(f"Project: {metadata['project_id']}")
        print(f"ML System: {metadata['ml_system']}")
        print(f"Datasets scanned: {metadata['datasets_scanned']}")
        print(f"Tables scanned: {metadata['tables_scanned']}")
        print(f"Total mappings: {metadata['total_mappings']}")
        print(f"High confidence mappings: {metadata['high_confidence_mappings']}")
        
        print("\nTOP MAPPINGS:")
        for mapping in report['mappings'][:15]:
            print(f"  {mapping['source']} -> {mapping['metric']} ({mapping['confidence']:.3f})")
            print(f"    Pattern: {mapping['pattern_type']}")
            print(f"    Samples: {mapping['sample_values'][:2]}")
            print()
        
        print("="*80)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Scan BigQuery for visibility data')
    parser.add_argument('--project', help='BigQuery project ID')
    args = parser.parse_args()
    
    scanner = BigQueryScanner()
    mappings = scanner.scan_bigquery_project(args.project)
    scanner.save_results(mappings, args.project or scanner.get_current_project())

if __name__ == "__main__":
    main()
