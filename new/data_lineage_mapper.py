#!/usr/bin/env python3

import json
import os
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Any
from google.cloud import bigquery
from google.oauth2 import service_account
import ssl
import certifi

class DataLineageMapper:
    def __init__(self, project_root: str = ".", discovery_results_path: str = "bigquery_discovery_results.json"):
        self.project_root = Path(project_root)
        self.discovery_results_path = Path(discovery_results_path)
        
        # Set SSL certificate bundle like the original script
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        
        SERVICE_ACCOUNT_FILE = os.path.join(self.project_root, "gcp_prod_key.json")
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        project = "chronicle-flow"
        clientBQ = bigquery.Client(project=project, credentials=credentials)
        self.clientBQ = clientBQ
        
        self.lineage_data = {
            "data_sources": {},
            "data_flows": {},
            "transformation_points": {},
            "data_dependencies": {}
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def runBQQuery(self, query, params=None):
        if params:
            df = self.clientBQ.query(query, job_config=params).to_dataframe()
        else:
            df = self.clientBQ.query(query).to_dataframe()
        return df

    def map_all_lineage(self):
        if not self.discovery_results_path.exists():
            self.logger.error(f"Discovery results not found: {self.discovery_results_path}")
            return None
        
        with open(self.discovery_results_path, 'r') as f:
            discovery_data = json.load(f)
        
        self.logger.info("Starting data lineage mapping...")
        
        self.map_bigquery_lineage(discovery_data)
        self.identify_data_flows()
        self.map_transformation_points()
        self.analyze_dependencies()
        
        self.save_lineage_data()
        return self.lineage_data

    def map_bigquery_lineage(self, discovery_data: Dict):
        for dataset_id, dataset_info in discovery_data.get("all_datasets", {}).items():
            for table_name, table_info in dataset_info.get("tables", {}).items():
                table_key = f"{dataset_id}.{table_name}"
                
                source_info = {
                    "table_key": table_key,
                    "row_count": table_info.get("row_count", 0),
                    "size_bytes": table_info.get("size_bytes", 0),
                    "last_modified": table_info.get("last_modified", ""),
                    "upstream_sources": self.identify_upstream_sources(dataset_id, table_name),
                    "downstream_targets": self.identify_downstream_targets(dataset_id, table_name),
                    "ingestion_patterns": self.analyze_ingestion_patterns(dataset_id, table_name),
                    "access_patterns": self.analyze_access_patterns(dataset_id, table_name)
                }
                
                self.lineage_data["data_sources"][table_key] = source_info

    def identify_upstream_sources(self, dataset_id: str, table_name: str) -> List[str]:
        upstream_sources = []
        
        try:
            job_query = f"""
            SELECT DISTINCT
                referenced_tables.dataset_id,
                referenced_tables.table_id
            FROM `chronicle-flow.region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`,
            UNNEST(referenced_tables) AS referenced_tables
            WHERE 
                destination_table.dataset_id = '{dataset_id}'
                AND destination_table.table_id = '{table_name}'
                AND creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                AND referenced_tables.dataset_id != '{dataset_id}'
            """
            
            upstream_df = self.runBQQuery(job_query)
            
            for _, row in upstream_df.iterrows():
                if pd.notna(row['dataset_id']) and pd.notna(row['table_id']):
                    upstream_sources.append(f"{row['dataset_id']}.{row['table_id']}")
            
        except Exception as e:
            self.logger.warning(f"Could not identify upstream sources for {dataset_id}.{table_name}: {e}")
        
        return upstream_sources

    def identify_downstream_targets(self, dataset_id: str, table_name: str) -> List[str]:
        downstream_targets = []
        
        try:
            downstream_query = f"""
            SELECT DISTINCT
                destination_table.dataset_id,
                destination_table.table_id
            FROM `chronicle-flow.region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`,
            UNNEST(referenced_tables) AS referenced_tables
            WHERE 
                referenced_tables.dataset_id = '{dataset_id}'
                AND referenced_tables.table_id = '{table_name}'
                AND creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                AND destination_table.dataset_id != '{dataset_id}'
            """
            
            downstream_df = self.runBQQuery(downstream_query)
            
            for _, row in downstream_df.iterrows():
                if pd.notna(row['dataset_id']) and pd.notna(row['table_id']):
                    downstream_targets.append(f"{row['dataset_id']}.{row['table_id']}")
            
        except Exception as e:
            self.logger.warning(f"Could not identify downstream targets for {dataset_id}.{table_name}: {e}")
        
        return downstream_targets

    def analyze_ingestion_patterns(self, dataset_id: str, table_name: str) -> Dict:
        try:
            ingestion_query = f"""
            SELECT 
                DATE(creation_time) as ingestion_date,
                COUNT(*) as job_count,
                SUM(total_bytes_processed) as total_bytes,
                job_type
            FROM `chronicle-flow.region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
            WHERE 
                destination_table.dataset_id = '{dataset_id}'
                AND destination_table.table_id = '{table_name}'
                AND creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
            GROUP BY DATE(creation_time), job_type
            ORDER BY ingestion_date DESC
            LIMIT 30
            """
            
            pattern_df = self.runBQQuery(ingestion_query)
            
            if not pattern_df.empty:
                return {
                    "daily_patterns": pattern_df.to_dict('records'),
                    "avg_daily_jobs": pattern_df['job_count'].mean(),
                    "total_bytes_processed": pattern_df['total_bytes'].sum(),
                    "ingestion_frequency": "daily" if len(pattern_df) > 20 else "irregular"
                }
            
        except Exception as e:
            self.logger.warning(f"Could not analyze ingestion patterns for {dataset_id}.{table_name}: {e}")
        
        return {"ingestion_frequency": "unknown"}

    def analyze_access_patterns(self, dataset_id: str, table_name: str) -> Dict:
        try:
            access_query = f"""
            SELECT 
                DATE(creation_time) as access_date,
                COUNT(*) as query_count,
                COUNT(DISTINCT user_email) as unique_users,
                AVG(total_slot_ms) as avg_processing_time
            FROM `chronicle-flow.region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`,
            UNNEST(referenced_tables) AS referenced_tables
            WHERE 
                referenced_tables.dataset_id = '{dataset_id}'
                AND referenced_tables.table_id = '{table_name}'
                AND creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
                AND job_type = 'QUERY'
            GROUP BY DATE(creation_time)
            ORDER BY access_date DESC
            """
            
            access_df = self.runBQQuery(access_query)
            
            if not access_df.empty:
                return {
                    "weekly_access_patterns": access_df.to_dict('records'),
                    "avg_daily_queries": access_df['query_count'].mean(),
                    "unique_users_count": access_df['unique_users'].sum(),
                    "access_frequency": "high" if access_df['query_count'].mean() > 10 else "low"
                }
            
        except Exception as e:
            self.logger.warning(f"Could not analyze access patterns for {dataset_id}.{table_name}: {e}")
        
        return {"access_frequency": "unknown"}

    def identify_data_flows(self):
        for table_key, source_info in self.lineage_data["data_sources"].items():
            upstream_sources = source_info.get("upstream_sources", [])
            downstream_targets = source_info.get("downstream_targets", [])
            
            for upstream in upstream_sources:
                flow_key = f"{upstream}->{table_key}"
                self.lineage_data["data_flows"][flow_key] = {
                    "source": upstream,
                    "target": table_key,
                    "flow_type": "ingestion",
                    "frequency": source_info.get("ingestion_patterns", {}).get("ingestion_frequency", "unknown")
                }
            
            for downstream in downstream_targets:
                flow_key = f"{table_key}->{downstream}"
                self.lineage_data["data_flows"][flow_key] = {
                    "source": table_key,
                    "target": downstream,
                    "flow_type": "transformation",
                    "frequency": source_info.get("access_patterns", {}).get("access_frequency", "unknown")
                }

    def map_transformation_points(self):
        transformation_patterns = {
            "staging": ["stg_", "staging_", "raw_"],
            "intermediate": ["int_", "intermediate_", "temp_"],
            "mart": ["mart_", "dim_", "fact_", "agg_"],
            "reporting": ["rpt_", "report_", "dashboard_"]
        }
        
        for table_key, source_info in self.lineage_data["data_sources"].items():
            table_name = table_key.split('.')[-1].lower()
            
            transformation_type = "unknown"
            for pattern_type, patterns in transformation_patterns.items():
                if any(pattern in table_name for pattern in patterns):
                    transformation_type = pattern_type
                    break
            
            if transformation_type != "unknown" or source_info.get("upstream_sources", []):
                self.lineage_data["transformation_points"][table_key] = {
                    "table_key": table_key,
                    "transformation_type": transformation_type,
                    "input_count": len(source_info.get("upstream_sources", [])),
                    "output_count": len(source_info.get("downstream_targets", [])),
                    "complexity_score": self.calculate_transformation_complexity(source_info)
                }

    def calculate_transformation_complexity(self, source_info: Dict) -> str:
        input_count = len(source_info.get("upstream_sources", []))
        output_count = len(source_info.get("downstream_targets", []))
        total_connections = input_count + output_count
        
        if total_connections == 0:
            return "isolated"
        elif total_connections <= 2:
            return "simple"
        elif total_connections <= 5:
            return "moderate"
        else:
            return "complex"

    def analyze_dependencies(self):
        for table_key, source_info in self.lineage_data["data_sources"].items():
            upstream_sources = source_info.get("upstream_sources", [])
            
            dependency_info = {
                "table_key": table_key,
                "dependency_count": len(upstream_sources),
                "critical_dependencies": [],
                "dependency_health": "healthy"
            }
            
            for upstream in upstream_sources:
                if upstream in self.lineage_data["data_sources"]:
                    upstream_info = self.lineage_data["data_sources"][upstream]
                    
                    if upstream_info.get("row_count", 0) == 0:
                        dependency_info["critical_dependencies"].append({
                            "source": upstream,
                            "issue": "empty_table"
                        })
                        dependency_info["dependency_health"] = "warning"
            
            if dependency_info["critical_dependencies"]:
                dependency_info["dependency_health"] = "critical"
            
            self.lineage_data["data_dependencies"][table_key] = dependency_info

    def save_lineage_data(self):
        output_file = Path("data_lineage_mapping.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.lineage_data, f, indent=2, default=str)
        
        self.logger.info(f"Data lineage mapping saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else "."
    discovery_results = sys.argv[2] if len(sys.argv) > 2 else "bigquery_discovery_results.json"
    
    mapper = DataLineageMapper(project_root, discovery_results)
    results = mapper.map_all_lineage()
    
    print(f"\nData Lineage Mapping Summary:")
    print(f"- Data sources mapped: {len(results['data_sources'])}")
    print(f"- Data flows identified: {len(results['data_flows'])}")
    print(f"- Transformation points: {len(results['transformation_points'])}")
    print(f"- Dependencies analyzed: {len(results['data_dependencies'])}")