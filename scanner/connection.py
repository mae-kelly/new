import os
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from google.cloud import bigquery
from google.oauth2 import service_account
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class BigQueryConnection:
    def __init__(self, service_account_path=None):
        self.project_id = "chronicle-fisv"
        self.client = None
        self.config = self._load_environment_config()
        self.session = self._create_session()
        self.connection_pool = None
        self.query_cache = {}
        self.connection_stats = {
            'queries_executed': 0,
            'cache_hits': 0,
            'connection_time': None,
            'last_activity': None
        }
        self._connect(service_account_path)
    
    def _load_environment_config(self):
        config = {
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
            'https_proxy': os.getenv('HTTPS_PROXY'),
            'max_retries': int(os.getenv('MAX_RETRIES', '3')),
            'timeout': int(os.getenv('REQUEST_TIMEOUT', '30')),
            'cache_ttl': int(os.getenv('CACHE_TTL', '300'))
        }
        
        logger.info(f"Configuration loaded with proxy: {config['http_proxy'] or 'None'}")
        return config
    
    def _create_session(self):
        session = requests.Session()
        
        if self.config['http_proxy'] or self.config['https_proxy']:
            session.proxies = {
                'http': self.config['http_proxy'],
                'https': self.config['https_proxy']
            }
            logger.info(f"Session configured with proxy: {self.config['http_proxy']}")
        
        retry_strategy = Retry(
            total=self.config['max_retries'],
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.timeout = self.config['timeout']
        
        ssl_cert_file = os.getenv('SSL_CERT_FILE') or os.getenv('REQUESTS_CA_BUNDLE')
        if ssl_cert_file and os.path.exists(ssl_cert_file):
            session.verify = ssl_cert_file
            logger.info(f"SSL verification using: {ssl_cert_file}")
        
        return session
    
    def _connect(self, service_account_path):
        start_time = time.time()
        
        try:
            service_account_file = self._resolve_service_account_path(service_account_path)
            
            if not os.path.exists(service_account_file):
                raise FileNotFoundError(f"Service account file not found: {service_account_file}")
            
            credentials = service_account.Credentials.from_service_account_file(service_account_file)
            
            self._configure_environment(service_account_file)
            
            self.client = bigquery.Client(
                project=self.project_id, 
                credentials=credentials,
                default_query_job_config=bigquery.QueryJobConfig(
                    use_query_cache=True,
                    use_legacy_sql=False,
                    maximum_bytes_billed=10**12
                )
            )
            
            self._test_connection()
            
            connection_time = time.time() - start_time
            self.connection_stats['connection_time'] = connection_time
            self.connection_stats['last_activity'] = datetime.now()
            
            logger.info(f"Connected to BigQuery project: {self.project_id} in {connection_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to connect to BigQuery: {e}")
            raise
    
    def _resolve_service_account_path(self, service_account_path):
        if service_account_path and os.path.exists(service_account_path):
            return service_account_path
        
        file_path = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(file_path)
        
        possible_paths = [
            os.path.join(parent_dir, "gcp_prod_key.json"),
            os.path.join(file_path, "gcp_prod_key.json"),
            os.path.join(os.getcwd(), "gcp_prod_key.json"),
            os.path.expanduser("~/gcp_prod_key.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found service account file at: {path}")
                return path
        
        raise FileNotFoundError("Service account file not found in any expected location")
    
    def _configure_environment(self, service_account_file):
        os.environ.update({
            'GOOGLE_APPLICATION_CREDENTIALS': service_account_file,
            'GCP_PROJECT': self.project_id,
            'GOOGLE_CLOUD_PROJECT': self.project_id
        })
        
        if self.config['http_proxy']:
            os.environ.update({
                'HTTP_PROXY': self.config['http_proxy'],
                'HTTPS_PROXY': self.config['https_proxy'] or self.config['http_proxy'],
                'http_proxy': self.config['http_proxy'],
                'https_proxy': self.config['https_proxy'] or self.config['http_proxy']
            })
            logger.info("Proxy environment variables configured")
    
    def _test_connection(self):
        try:
            datasets = list(self.client.list_datasets(max_results=5))
            logger.info(f"Connection test successful. Found {len(datasets)} datasets (showing first 5).")
            
            if datasets:
                sample_dataset = datasets[0]
                tables = list(self.client.list_tables(sample_dataset.dataset_id, max_results=3))
                logger.info(f"Sample dataset '{sample_dataset.dataset_id}' has {len(tables)} tables (showing first 3)")
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise
    
    def get_client(self):
        return self.client
    
    def list_datasets(self):
        try:
            datasets = list(self.client.list_datasets())
            logger.info(f"Retrieved {len(datasets)} datasets")
            return datasets
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return []
    
    def list_tables(self, dataset_id):
        try:
            dataset_ref = bigquery.DatasetReference(self.project_id, dataset_id)
            tables = list(self.client.list_tables(dataset_ref))
            logger.debug(f"Dataset {dataset_id} has {len(tables)} tables")
            return tables
        except Exception as e:
            logger.error(f"Failed to list tables in dataset {dataset_id}: {e}")
            return []
    
    def get_table_schema(self, dataset_id, table_id):
        try:
            table_ref = bigquery.TableReference(
                bigquery.DatasetReference(self.project_id, dataset_id), 
                table_id
            )
            table = self.client.get_table(table_ref)
            logger.debug(f"Retrieved schema for {dataset_id}.{table_id} with {len(table.schema)} fields")
            return table.schema
        except Exception as e:
            logger.error(f"Failed to get schema for {dataset_id}.{table_id}: {e}")
            return []
    
    def batch_sample_table_data(self, dataset_id, table_id, fields=None, limit=20):
        cache_key = f"{dataset_id}.{table_id}:{hash(str(fields))}:{limit}"
        
        if cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.config['cache_ttl']:
                self.connection_stats['cache_hits'] += 1
                logger.debug(f"Cache hit for table sample: {dataset_id}.{table_id}")
                return cache_entry['data']
        
        try:
            if fields:
                field_list = ', '.join([f'`{field}`' for field in fields])
            else:
                field_list = '*'
            
            queries_to_try = [
                f"""
                    SELECT {field_list}
                    FROM `{self.project_id}.{dataset_id}.{table_id}` TABLESAMPLE SYSTEM (0.1 PERCENT)
                    LIMIT {limit}
                """,
                f"""
                    SELECT {field_list}
                    FROM `{self.project_id}.{dataset_id}.{table_id}`
                    LIMIT {limit}
                """,
                f"""
                    SELECT {field_list}
                    FROM `{self.project_id}.{dataset_id}.{table_id}`
                    WHERE RAND() < 0.01
                    LIMIT {limit}
                """
            ]
            
            for i, query in enumerate(queries_to_try):
                try:
                    job = self.client.query(query)
                    results = list(job.result())
                    
                    self.query_cache[cache_key] = {
                        'data': results,
                        'timestamp': time.time()
                    }
                    
                    self.connection_stats['queries_executed'] += 1
                    self.connection_stats['last_activity'] = datetime.now()
                    
                    logger.debug(f"Successfully sampled {len(results)} rows from {dataset_id}.{table_id} using method {i+1}")
                    return results
                    
                except Exception as e:
                    logger.debug(f"Sampling method {i+1} failed for {dataset_id}.{table_id}: {e}")
                    if i == len(queries_to_try) - 1:
                        raise
                    continue
            
        except Exception as e:
            logger.warning(f"All sampling methods failed for {dataset_id}.{table_id}: {e}")
            return []
    
    def execute_query(self, query, use_cache=True):
        query_hash = hash(query)
        
        if use_cache and query_hash in self.query_cache:
            cache_entry = self.query_cache[query_hash]
            if time.time() - cache_entry['timestamp'] < self.config['cache_ttl']:
                self.connection_stats['cache_hits'] += 1
                return cache_entry['data']
        
        try:
            start_time = time.time()
            job = self.client.query(query)
            results = list(job.result())
            execution_time = time.time() - start_time
            
            if use_cache:
                self.query_cache[query_hash] = {
                    'data': results,
                    'timestamp': time.time()
                }
            
            self.connection_stats['queries_executed'] += 1
            self.connection_stats['last_activity'] = datetime.now()
            
            logger.debug(f"Query executed in {execution_time:.2f}s, returned {len(results)} rows")
            return results
            
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
            
            info = {
                'num_rows': table.num_rows or 0,
                'num_bytes': table.num_bytes or 0,
                'created': table.created,
                'modified': table.modified,
                'table_type': table.table_type,
                'description': table.description or '',
                'labels': dict(table.labels) if table.labels else {},
                'location': table.location,
                'schema_fields': len(table.schema)
            }
            
            logger.debug(f"Retrieved info for {dataset_id}.{table_id}: {info['num_rows']} rows, {info['schema_fields']} fields")
            return info
            
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
                    'description': f'AO1 Scanner {action}: {data}',
                    'timestamp': datetime.now().isoformat(),
                    'connection_stats': self.connection_stats
                }
                
                response = self.session.post(
                    self.config['chronicle_endpoint'], 
                    headers=headers,
                    json=log_data,
                    timeout=30
                )
                
                if response.status_code >= 400:
                    logger.warning(f"Chronicle logging failed: {response.status_code}")
                else:
                    logger.debug(f"Activity logged to Chronicle: {action}")
                    
        except Exception as e:
            logger.debug(f"Activity logging failed: {e}")
    
    def run_bq_query(self, query, params=None):
        try:
            job_config = bigquery.QueryJobConfig()
            if params:
                job_config.query_parameters = params
            
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            self.connection_stats['queries_executed'] += 1
            self.connection_stats['last_activity'] = datetime.now()
            
            return results
            
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
    
    @contextmanager
    def get_db_cursor(self):
        conn = None
        try:
            yield self.client
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise e
        finally:
            if conn:
                conn.close()
    
    def clear_cache(self):
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_connection_stats(self):
        return self.connection_stats.copy()
    
    def health_check(self):
        try:
            test_query = f"SELECT 1 as health_check"
            result = self.execute_query(test_query, use_cache=False)
            
            if result and len(result) == 1:
                return {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'stats': self.get_connection_stats()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': 'Health check query failed',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }