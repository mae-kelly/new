#!/usr/bin/env python3

import os
import json
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Any
from google.cloud import bigquery
from google.oauth2 import service_account
import ssl
import certifi

class BigQueryDataDiscovery:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
        # Set SSL certificate bundle like the original script
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        
        SERVICE_ACCOUNT_FILE = os.path.join(self.project_root, "gcp_prod_key.json")
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        project = "chronicle-flow"
        clientBQ = bigquery.Client(project=project, credentials=credentials)
        self.clientBQ = clientBQ
        
        self.discovered_data = {
            "all_datasets": {},
            "dataset_count": 0,
            "table_count": 0
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def runBQQuery(self, query, params=None):
        if params:
            df = self.clientBQ.query(query, job_config=params).to_dataframe()
        else:
            df = self.clientBQ.query(query).to_dataframe()
        return df

    def discover_all_datasets(self):
        self.logger.info("Discovering all BigQuery datasets...")
        
        try:
            datasets_query = """
            SELECT 
                schema_name as dataset_id,
                creation_time,
                last_modified_time
            FROM `chronicle-flow.INFORMATION_SCHEMA.SCHEMATA`
            WHERE catalog_name = 'chronicle-flow'
                AND schema_name NOT IN ('information_schema', 'sys')
            ORDER BY schema_name
            """
            
            datasets_df = self.runBQQuery(datasets_query)
            
            for _, row in datasets_df.iterrows():
                dataset_id = row['dataset_id']
                
                dataset_info = {
                    "dataset_id": dataset_id,
                    "creation_time": str(row['creation_time']),
                    "last_modified": str(row['last_modified_time']),
                    "tables": self.discover_dataset_tables(dataset_id)
                }
                
                self.discovered_data["all_datasets"][dataset_id] = dataset_info
                self.discovered_data["dataset_count"] += 1
                
                self.logger.info(f"Discovered dataset: {dataset_id}")
            
        except Exception as e:
            self.logger.error(f"Error discovering datasets: {e}")
        
        self.save_results()
        return self.discovered_data

    def discover_dataset_tables(self, dataset_id: str) -> Dict:
        try:
            tables_query = f"""
            SELECT 
                table_name,
                table_type,
                creation_time,
                last_modified_time,
                row_count,
                size_bytes
            FROM `chronicle-flow.{dataset_id}.INFORMATION_SCHEMA.TABLES`
            ORDER BY table_name
            """
            
            tables_df = self.runBQQuery(tables_query)
            tables_dict = {}
            
            for _, row in tables_df.iterrows():
                table_name = row['table_name']
                
                table_info = {
                    "table_name": table_name,
                    "table_type": row['table_type'],
                    "creation_time": str(row['creation_time']),
                    "last_modified": str(row['last_modified_time']),
                    "row_count": int(row['row_count']) if pd.notna(row['row_count']) else 0,
                    "size_bytes": int(row['size_bytes']) if pd.notna(row['size_bytes']) else 0,
                    "fields": self.discover_table_fields(dataset_id, table_name)
                }
                
                tables_dict[table_name] = table_info
                self.discovered_data["table_count"] += 1
            
            return tables_dict
            
        except Exception as e:
            self.logger.error(f"Error discovering tables in {dataset_id}: {e}")
            return {}

    def discover_table_fields(self, dataset_id: str, table_name: str) -> List[Dict]:
        try:
            fields_query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                is_partitioning_column,
                clustering_ordinal_position,
                description
            FROM `chronicle-flow.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """
            
            fields_df = self.runBQQuery(fields_query)
            return fields_df.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error discovering fields in {dataset_id}.{table_name}: {e}")
            return []

    def save_results(self):
        output_file = self.project_root / "bigquery_discovery_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.discovered_data, f, indent=2, default=str)
        
        self.logger.info(f"Discovery results saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else "."
    
    discovery_engine = BigQueryDataDiscovery(project_root)
    results = discovery_engine.discover_all_datasets()
    
    print(f"\nBigQuery Discovery Summary:")
    print(f"- Total datasets: {results['dataset_count']}")
    print(f"- Total tables: {results['table_count']}")
    print(f"- Datasets discovered: {list(results['all_datasets'].keys())}")