import os
import sys
import json
import pandas as pd
from google.cloud import bigquery
from google.cloud import resourcemanager_v3
from datetime import datetime
import time
import logging
from neural_engine_brilliant import BrilliantVisibilityMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigQueryWithExistingML:
    def __init__(self):
        self.bq_client = bigquery.Client()
        self.resource_client = resourcemanager_v3.ProjectsClient()
        self.ml_mapper = BrilliantVisibilityMapper()
        
    def get_all_projects(self):
        try:
            projects = []
            request = resourcemanager_v3.ListProjectsRequest()
            page_result = self.resource_client.list_projects(request=request)
            
            for project in page_result:
                if project.state == resourcemanager_v3.Project.State.ACTIVE:
                    projects.append(project.project_id)
            
            logger.info(f"Found {len(projects)} active projects")
            return projects
        except Exception as e:
            logger.error(f"Error getting projects: {e}")
            return []
    
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
            LIMIT 500
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
    
    def analyze_table_with_existing_ml(self, table_info):
        try:
            project_id = table_info['project']
            dataset_id = table_info['dataset']
            table_id = table_info['table']
            
            logger.info(f"Analyzing {project_id}.{dataset_id}.{table_id}")
            
            df = self.sample_table_data(project_id, dataset_id, table_id)
            if df is None:
                return []
            
            # Convert to the format expected by existing ML
            tables_dict = {table_id: df}
            
            # Use existing ML mapper
            mappings = self.ml_mapper.discover_mappings_with_brilliance(tables_dict)
            
            # Enhance mappings with BigQuery metadata
            enhanced_mappings = []
            for mapping in mappings:
                enhanced_mapping = {
                    'project': project_id,
                    'dataset': dataset_id,
                    'table': table_id,
                    'column': mapping.column_name,
                    'source': f"{project_id}.{dataset_id}.{table_id}.{mapping.column_name}",
                    'metric': mapping.target_metric,
                    'confidence': mapping.entanglement_strength,
                    'pattern_type': mapping.reasoning_graph.get('pattern_type', 'unknown'),
                    'detection_methods': mapping.reasoning_graph.get('detection_methods', []),
                    'sample_values': mapping.reasoning_graph.get('sample_values', []),
                    'full_table_name': table_info['full_name']
                }
                enhanced_mappings.append(enhanced_mapping)
            
            logger.info(f"Found {len(enhanced_mappings)} mappings for {table_id}")
            return enhanced_mappings
            
        except Exception as e:
            logger.error(f"Error analyzing table {table_info}: {e}")
            return []
    
    def scan_all_bigquery_with_existing_ml(self):
        logger.info("Starting BigQuery scan with existing ML system")
        start_time = time.time()
        
        projects = self.get_all_projects()
        if not projects:
            logger.error("No projects found")
            return []
        
        all_mappings = []
        
        for project_id in projects:
            try:
                logger.info(f"Scanning project: {project_id}")
                datasets = self.get_datasets_for_project(project_id)
                
                for dataset_id in datasets:
                    try:
                        tables = self.get_tables_for_dataset(project_id, dataset_id)
                        
                        for table_info in tables:
                            mappings = self.analyze_table_with_existing_ml(table_info)
                            all_mappings.extend(mappings)
                    
                    except Exception as dataset_error:
                        logger.error(f"Error processing dataset {dataset_id}: {dataset_error}")
                        continue
                        
            except Exception as project_error:
                logger.error(f"Error processing project {project_id}: {project_error}")
                continue
        
        end_time = time.time()
        logger.info(f"Scan completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Total mappings found: {len(all_mappings)}")
        
        return all_mappings
    
    def save_results(self, mappings):
        os.makedirs('outputs', exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        high_confidence_mappings = [m for m in mappings if m['confidence'] > 0.7]
        
        # Group by metric for better analysis
        metrics_summary = {}
        for mapping in mappings:
            metric = mapping['metric']
            if metric not in metrics_summary:
                metrics_summary[metric] = []
            metrics_summary[metric].append(mapping)
        
        # Group by project
        projects_summary = {}
        for mapping in mappings:
            project = mapping['project']
            if project not in projects_summary:
                projects_summary[project] = []
            projects_summary[project].append(mapping)
        
        report = {
            'scan_metadata': {
                'timestamp': timestamp,
                'total_mappings': len(mappings),
                'high_confidence_mappings': len(high_confidence_mappings),
                'projects_scanned': len(set(m['project'] for m in mappings)),
                'datasets_scanned': len(set(f"{m['project']}.{m['dataset']}" for m in mappings)),
                'tables_scanned': len(set(f"{m['project']}.{m['dataset']}.{m['table']}" for m in mappings)),
                'ml_system': 'BrilliantVisibilityMapper from neural_engine_brilliant.py'
            },
            'mappings': sorted(mappings, key=lambda x: x['confidence'], reverse=True),
            'metrics_summary': {
                metric: {
                    'count': len(mappings),
                    'avg_confidence': sum(m['confidence'] for m in mappings) / len(mappings),
                    'projects': list(set(m['project'] for m in mappings)),
                    'best_sources': sorted(mappings, key=lambda x: x['confidence'], reverse=True)[:5]
                }
                for metric, mappings in metrics_summary.items()
            },
            'projects_summary': {
                project: {
                    'total_mappings': len(mappings),
                    'high_confidence': len([m for m in mappings if m['confidence'] > 0.7]),
                    'datasets': len(set(m['dataset'] for m in mappings)),
                    'tables': len(set(m['table'] for m in mappings)),
                    'metrics': list(set(m['metric'] for m in mappings))
                }
                for project, mappings in projects_summary.items()
            }
        }
        
        with open('outputs/bigquery_existing_ml_scan.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Results saved to outputs/bigquery_existing_ml_scan.json")
        
        self.print_summary(report)
    
    def print_summary(self, report):
        print("\n" + "="*80)
        print("BIGQUERY SCAN WITH EXISTING ML SYSTEM")
        print("="*80)
        
        metadata = report['scan_metadata']
        print(f"ML System: {metadata['ml_system']}")
        print(f"Scan completed at: {metadata['timestamp']}")
        print(f"Projects scanned: {metadata['projects_scanned']}")
        print(f"Datasets scanned: {metadata['datasets_scanned']}")
        print(f"Tables scanned: {metadata['tables_scanned']}")
        print(f"Total mappings: {metadata['total_mappings']}")
        print(f"High confidence mappings: {metadata['high_confidence_mappings']}")
        
        print("\nVISIBILITY METRICS DISCOVERED:")
        for metric, data in report['metrics_summary'].items():
            print(f"  {metric}: {data['count']} sources (avg confidence: {data['avg_confidence']:.3f})")
            print(f"    Found in projects: {', '.join(data['projects'][:3])}")
        
        print("\nTOP PROJECTS BY MAPPINGS:")
        sorted_projects = sorted(report['projects_summary'].items(), 
                               key=lambda x: x[1]['total_mappings'], reverse=True)
        for project, data in sorted_projects[:10]:
            print(f"  {project}: {data['total_mappings']} mappings, {data['datasets']} datasets, {data['tables']} tables")
        
        print("\nHIGHEST CONFIDENCE MAPPINGS:")
        for mapping in report['mappings'][:15]:
            print(f"  {mapping['source']} -> {mapping['metric']} ({mapping['confidence']:.3f})")
            print(f"    Pattern: {mapping['pattern_type']}")
            print(f"    Methods: {', '.join(mapping['detection_methods'])}")
            print(f"    Samples: {mapping['sample_values'][:2]}")
            print()
        
        print("="*80)

def main():
    scanner = BigQueryWithExistingML()
    mappings = scanner.scan_all_bigquery_with_existing_ml()
    scanner.save_results(mappings)

if __name__ == "__main__":
    main()
