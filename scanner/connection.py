import os
import logging
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class BigQueryConnection:
    def __init__(self, service_account_path=None):
        self.project_id = "chronicle-fisv"
        self.client = None
        self.config = self._load_environment_config()
        self.session = self._create_session()
        self._connect(service_account_path)
    
    def _load_environment_config(self):
        return {
            'authority': os.getenv('AUTHORITY'),
            'client_id': os.getenv('CLIENT_ID'),
            'client_secret': os.getenv('CLIENT_SECRET'),
            'redirect_uri': os.getenv('REDIRECT_URI'),
            'scope': os.getenv('SCOPE'),
            'endpoint': os.getenv('ENDPOINT'),
            'chronicle_api_key': os.getenv('CHRONICLE_API_KEY'),
            'chronicle_secret_key': os.getenv('CHRONICLE_SECRET_KEY'),
            'chronicle_endpoint': os.getenv('CHRONICLE_ENDPOINT'),
            'http_proxy': os.getenv('HTTP_PROXY'),
            'https_proxy': os.getenv('HTTPS_PROXY')
        }
    
    def _create_session(self):
        session = requests.Session()
        
        if self.config['http_proxy'] or self.config['https_proxy']:
            session.proxies = {
                'http': self.config['http_proxy'],
                'https': self.config['https_proxy']
            }
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.timeout = 30
        
        return session
    
    def _connect(self, service_account_path):
        try:
            if service_account_path and os.path.exists(service_account_path):
                SERVICE_ACCOUNT_FILE = service_account_path
            else:
                file_path = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(file_path)
                SERVICE_ACCOUNT_FILE = os.path.join(parent_dir, "gcp_prod_key.json")
            
            if not os.path.exists(SERVICE_ACCOUNT_FILE):
                raise FileNotFoundError(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
            
            credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
            
            os.environ.update({
                'GOOGLE_APPLICATION_CREDENTIALS': SERVICE_ACCOUNT_FILE,
                'GCP_PROJECT': self.project_id
            })
            
            if self.config['http_proxy']:
                os.environ.update({
                    'HTTP_PROXY': self.config['http_proxy'],
                    'HTTPS_PROXY': self.config['https_proxy'] or self.config['http_proxy']
                })
            
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
    
    def batch_sample_table_data(self, dataset_id, table_id, fields=None, limit=20):
        try:
            if fields:
                field_list = ', '.join([f'`{field}`' for field in fields])
            else:
                field_list = '*'
            
            query = f"""
                SELECT {field_list}
                FROM `{self.project_id}.{dataset_id}.{table_id}` TABLESAMPLE SYSTEM (0.1 PERCENT)
                LIMIT {limit}
            """
            
            job = self.client.query(query)
            return list(job.result())
        except Exception as e:
            logger.warning(f"Failed to sample data from {dataset_id}.{table_id}: {e}")
            try:
                query = f"""
                    SELECT {field_list}
                    FROM `{self.project_id}.{dataset_id}.{table_id}`
                    LIMIT {limit}
                """
                job = self.client.query(query)
                return list(job.result())
            except Exception as e2:
                logger.warning(f"Fallback sampling also failed: {e2}")
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
    
    def log_activity(self, action, data):
        try:
            if self.config['chronicle_endpoint'] and self.config['chronicle_api_key']:
                headers = {
                    'X-goog-api-key': self.config['chronicle_api_key'],
                    'X-webhook-Access-Key': self.config['chronicle_secret_key'],
                    'Content-Type': 'application/json'
                }
                
                log_data = {
                    'base_app': 'AO1Scanner',
                    'app': 'scanner',
                    'host': 'localhost',
                    'method': 'scan',
                    'path': f'/scanner/{action}',
                    'query_string': '',
                    'remote_addr': '127.0.0.1',
                    'root_path': '',
                    'scheme': 'https',
                    'server': 'ao1-scanner',
                    'url': f'https://ao1-scanner/action/{action}',
                    'status_code': 200,
                    'description': f'AO1 Scanner {action}: {data}'
                }
                
                response = self.session.post(
                    self.config['chronicle_endpoint'], 
                    headers=headers,
                    json=log_data,
                    timeout=30
                )
                
                if response.status_code >= 400:
                    logger.warning(f"Chronicle logging failed: {response.status_code}")
                    
        except Exception as e:
            logger.debug(f"Activity logging failed: {e}")
    
    def run_bq_query(self, query, params=None):
        try:
            job_config = bigquery.QueryJobConfig()
            if params:
                job_config.query_parameters = params
            
            query_job = self.client.query(query, job_config=job_config)
            return query_job.result()
        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
            raise
    
    def run_local_db_query(self, query, params=None):
        try:
            import duckdb
            
            conn = duckdb.connect()
            
            if params:
                result = conn.execute(query, params).fetchall()
            else:
                result = conn.execute(query).fetchall()
            
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"Local database query failed: {e}")
            raise