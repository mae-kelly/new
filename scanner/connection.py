import os
import logging
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account
from .config import PROJECT_ID, SERVICE_ACCOUNT_FILE

logger = logging.getLogger(__name__)

class BigQueryConnection:
    def __init__(self, service_account_path=None):
        self.project_id = PROJECT_ID
        self.client = None
        self._connect(service_account_path)
    
    def _connect(self, service_account_path):
        try:
            file_path = os.path.dirname(os.path.abspath(__file__))
            
            if service_account_path and os.path.exists(service_account_path):
                SERVICE_ACCOUNT_FILE_PATH = service_account_path
            else:
                SERVICE_ACCOUNT_FILE_PATH = os.path.join(file_path, SERVICE_ACCOUNT_FILE)
            
            if not os.path.exists(SERVICE_ACCOUNT_FILE_PATH):
                raise FileNotFoundError(f"Service account file not found: {SERVICE_ACCOUNT_FILE_PATH}")
            
            credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE_PATH)
            self.client = bigquery.Client(project=self.project_id, credentials=credentials)
            
            self._test_connection()
            logger.info(f"Connected to BigQuery project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"Failed to connect to BigQuery: {e}")
            raise
    
    def _test_connection(self):
        try:
            datasets = list(self.client.list_datasets())
            logger.info(f"Connection test successful. Found {len(datasets)} datasets.")
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise
    
    def get_client(self):
        return self.client
    
    def list_datasets(self):
        return list(self.client.list_datasets())
    
    def list_tables(self, dataset_id):
        dataset_ref = bigquery.DatasetReference(self.project_id, dataset_id)
        return list(self.client.list_tables(dataset_ref))
    
    def get_table_schema(self, dataset_id, table_id):
        table_ref = bigquery.TableReference(
            bigquery.DatasetReference(self.project_id, dataset_id), 
            table_id
        )
        table = self.client.get_table(table_ref)
        return table.schema
    
    def sample_table_data(self, dataset_id, table_id, limit=100, fields=None):
        if fields:
            field_list = ', '.join([f'`{field}`' for field in fields])
            query = f"""
                SELECT {field_list}
                FROM `{self.project_id}.{dataset_id}.{table_id}`
                WHERE RAND() < 0.1
                LIMIT {limit}
            """
        else:
            query = f"""
                SELECT *
                FROM `{self.project_id}.{dataset_id}.{table_id}`
                WHERE RAND() < 0.1
                LIMIT {limit}
            """
        
        try:
            job = self.client.query(query)
            return list(job.result())
        except Exception as e:
            logger.warning(f"Failed to sample data from {dataset_id}.{table_id}: {e}")
            return []
    
    def execute_query(self, query):
        try:
            job = self.client.query(query)
            return list(job.result())
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    def get_table_info(self, dataset_id, table_id):
        try:
            table_ref = bigquery.TableReference(
                bigquery.DatasetReference(self.project_id, dataset_id), 
                table_id
            )
            table = self.client.get_table(table_ref)
            
            return {
                'num_rows': table.num_rows or 0,
                'num_bytes': table.num_bytes or 0,
                'created': table.created,
                'modified': table.modified,
                'table_type': table.table_type,
                'description': table.description or ''
            }
        except Exception as e:
            logger.warning(f"Failed to get table info for {dataset_id}.{table_id}: {e}")
            return {}