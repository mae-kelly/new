import os
import sys
import json
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import resourcemanager_v3
from datetime import datetime
import time
import logging
from typing import List, Dict, Optional
import re
from neural_engine_brilliant import BrilliantVisibilityMapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustBigQueryScanner:
    def __init__(self):
        try:
            self.bq_client = bigquery.Client()
            self.resource_client = resourcemanager_v3.ProjectsClient()
            self.ml_mapper = BrilliantVisibilityMapper()
            self.errors = []
            self.skipped_tables = []
            self.processed_tables = 0
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            sys.exit(1)
    
    def safe_list_projects(self, max_projects=50):
        try:
            request = resourcemanager_v3.ListProjectsRequest()
            projects = list(self.resource_client.list_projects(request=request))
            active = [p for p in projects if p.state == resourcemanager_v3.Project.State.ACTIVE]
            return active[:max_projects]
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []
    
    def safe_list_datasets(self, project_id, max_datasets=20):
        try:
            client = bigquery.Client(project=project_id)
            datasets = list(client.list_datasets(project_id))
            return [d.dataset_id for d in datasets[:max_datasets]]
        except Exception as e:
            logger.warning(f"Cannot access datasets in {project_id}: {e}")
            self.errors.append(f"Dataset access failed: {project_id} - {str(e)[:100]}")
            return []
    
    def safe_list_tables(self, project_id, dataset_id, max_tables=50):
        try:
            client = bigquery.Client(project=project_id)
            tables = list(client.list_tables(f"{project_id}.{dataset_id}"))
            return [(t.table_id, t.table_type) for t in tables[:max_tables]]
        except Exception as e:
            logger.warning(f"Cannot access tables in {project_id}.{dataset_id}: {e}")
            self.errors.append(f"Table access failed: {project_id}.{dataset_id} - {str(e)[:100]}")
            return []
    
    def safe_sample_table(self, project_id, dataset_id, table_id, sample_size=200):
        try:
            client = bigquery.Client(project=project_id)
            
            query = f"""
            SELECT * FROM `{project_id}.{dataset_id}.{table_id}`
            LIMIT {sample_size}
            """
            
            job_config = bigquery.QueryJobConfig()
            job_config.maximum_bytes_billed = 10 * 1024 * 1024  # 10MB limit
            
            query_job = client.query(query, job_config=job_config)
            df = query_job.to_dataframe()
            
            if df.empty:
                logger.info(f"Empty table: {project_id}.{dataset_id}.{table_id}")
                return None
            
            # Handle edge cases in data
            df = self.clean_dataframe(df)
            return df
            
        except Exception as e:
            logger.warning(f"Cannot sample {project_id}.{dataset_id}.{table_id}: {e}")
            self.errors.append(f"Sampling failed: {project_id}.{dataset_id}.{table_id} - {str(e)[:100]}")
            return None
    
    def clean_dataframe(self, df):
        try:
            # Handle common edge cases
            df = df.copy()
            
            # Drop columns that are all null
            df = df.dropna(axis=1, how='all')
            
            # Limit string column length to prevent memory issues
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str[:1000]
            
            # Handle datetime columns
            for col in df.columns:
                if 'timestamp' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                    except:
                        pass
            
            # Limit rows if too large
            if len(df) > 1000:
                df = df.sample(1000)
            
            return df
            
        except Exception as e:
            logger.warning(f"DataFrame cleaning failed: {e}")
            return df
    
    def safe_ml_analysis(self, tables_dict, table_info):
        try:
            mappings = self.ml_mapper.discover_mappings_with_brilliance(tables_dict)
            enhanced_mappings = []
            
            for mapping in mappings:
                # Cap confidence scores
                confidence = min(mapping.entanglement_strength, 1.0)
                
                enhanced_mapping = {
                    'project': table_info['project'],
                    'dataset': table_info['dataset'], 
                    'table': table_info['table'],
                    'column': mapping.column_name,
                    'source': f"{table_info['project']}.{table_info['dataset']}.{table_info['table']}.{mapping.column_name}",
                    'metric': mapping.target_metric,
                    'confidence': confidence,
                    'pattern_type': mapping.reasoning_graph.get('pattern_type', 'unknown'),
                    'detection_methods': mapping.reasoning_graph.get('detection_methods', []),
                    'sample_values': mapping.reasoning_graph.get('sample_values', [])[:3],
                    'full_table_name': f"{table_info['project']}.{table_info['dataset']}.{table_info['table']}"
                }
                enhanced_mappings.append(enhanced_mapping)
            
            return enhanced_mappings
            
        except Exception as e:
            logger.warning(f"ML analysis failed for {table_info}: {e}")
            self.errors.append(f"ML analysis failed: {table_info} - {str(e)[:100]}")
            return []
    
    def scan_with_edge_cases(self):
        logger.info("Starting robust BigQuery scan...")
        start_time = time.time()
        all_mappings = []
        
        projects = self.safe_list_projects()
        logger.info(f"Found {len(projects)} projects to scan")
        
        for project in projects:
            project_id = project.project_id
            logger.info(f"Scanning project: {project_id}")
            
            datasets = self.safe_list_datasets(project_id)
            
            for dataset_id in datasets:
                logger.info(f"Scanning dataset: {project_id}.{dataset_id}")
                
                tables = self.safe_list_tables(project_id, dataset_id)
                
                for table_id, table_type in tables:
                    # Skip views and external tables that might cause issues
                    if table_type not in ['TABLE']:
                        self.skipped_tables.append(f"{project_id}.{dataset_id}.{table_id} (type: {table_type})")
                        continue
                    
                    table_info = {
                        'project': project_id,
                        'dataset': dataset_id,
                        'table': table_id
                    }
                    
                    df = self.safe_sample_table(project_id, dataset_id, table_id)
                    
                    if df is not None and len(df) > 0:
                        tables_dict = {table_id: df}
                        mappings = self.safe_ml_analysis(tables_dict, table_info)
                        all_mappings.extend(mappings)
                        self.processed_tables += 1
                    
                    # Progress indicator
                    if self.processed_tables % 10 == 0:
                        logger.info(f"Processed {self.processed_tables} tables...")
        
        end_time = time.time()
        
        logger.info(f"Scan completed in {(end_time - start_time)/60:.1f} minutes")
        logger.info(f"Processed {self.processed_tables} tables")
        logger.info(f"Found {len(all_mappings)} mappings")
        logger.info(f"Encountered {len(self.errors)} errors")
        logger.info(f"Skipped {len(self.skipped_tables)} tables")
        
        return all_mappings
    
    def save_robust_results(self, mappings):
        os.makedirs('outputs', exist_ok=True)
        
        report = {
            'scan_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_mappings': len(mappings),
                'high_confidence_mappings': len([m for m in mappings if m['confidence'] > 0.7]),
                'projects_scanned': len(set(m['project'] for m in mappings)),
                'tables_processed': self.processed_tables,
                'errors_encountered': len(self.errors),
                'tables_skipped': len(self.skipped_tables),
                'scanner_version': 'robust_v1.0'
            },
            'mappings': sorted(mappings, key=lambda x: x['confidence'], reverse=True),
            'errors': self.errors,
            'skipped_tables': self.skipped_tables,
            'summary_by_metric': self.summarize_by_metric(mappings)
        }
        
        with open('outputs/robust_bigquery_scan.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Results saved to outputs/robust_bigquery_scan.json")
        self.print_summary(report)
    
    def summarize_by_metric(self, mappings):
        metrics = {}
        for mapping in mappings:
            metric = mapping['metric']
            if metric not in metrics:
                metrics[metric] = []
            metrics[metric].append(mapping)
        
        return {metric: {
            'count': len(mappings),
            'avg_confidence': sum(m['confidence'] for m in mappings) / len(mappings),
            'projects': list(set(m['project'] for m in mappings)),
            'best_sources': sorted(mappings, key=lambda x: x['confidence'], reverse=True)[:3]
        } for metric, mappings in metrics.items()}
    
    def print_summary(self, report):
        print(f"\n{'='*80}")
        print("ROBUST BIGQUERY SCAN RESULTS")
        print(f"{'='*80}")
        
        meta = report['scan_metadata']
        print(f"Projects scanned: {meta['projects_scanned']}")
        print(f"Tables processed: {meta['tables_processed']}")
        print(f"Total mappings: {meta['total_mappings']}")
        print(f"High confidence: {meta['high_confidence_mappings']}")
        print(f"Errors handled: {meta['errors_encountered']}")
        print(f"Tables skipped: {meta['tables_skipped']}")
        
        print(f"\nVISIBILITY METRICS FOUND:")
        for metric, data in report['summary_by_metric'].items():
            print(f"  {metric}: {data['count']} sources (avg: {data['avg_confidence']:.3f})")
        
        print(f"\nTOP MAPPINGS:")
        for mapping in report['mappings'][:10]:
            print(f"  {mapping['source']} -> {mapping['metric']} ({mapping['confidence']:.3f})")
        
        if report['errors']:
            print(f"\nERRORS ENCOUNTERED:")
            for error in report['errors'][:5]:
                print(f"  {error}")
            if len(report['errors']) > 5:
                print(f"  ... and {len(report['errors'])-5} more")

def main():
    scanner = RobustBigQueryScanner()
    mappings = scanner.scan_with_edge_cases()
    scanner.save_robust_results(mappings)

if __name__ == "__main__":
    main()
