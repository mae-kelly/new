import os
import asyncio
import json
import time
import logging
from google.cloud import bigquery
from google.cloud import resourcemanager_v3
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
from typing import Dict, List, Any
from quantum_engine import QuantumAO1Engine
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class HyperScaleBigQueryScanner:
    def __init__(self):
        try:
            credentials, project = default()
            self.bq_client = bigquery.Client(credentials=credentials, project=project)
            self.resource_client = resourcemanager_v3.ProjectsClient(credentials=credentials)
            self.quantum_engine = QuantumAO1Engine()
            self.scan_stats = {
                'projects_processed': 0,
                'datasets_processed': 0,
                'tables_processed': 0,
                'detections_found': 0,
                'errors_handled': 0,
                'permission_errors': 0,
                'quota_errors': 0,
                'network_errors': 0
            }
            self.failed_resources = []
            logger.info("BigQuery clients initialized successfully")
        except DefaultCredentialsError as e:
            logger.error(f"Authentication failed: {e}")
            raise Exception("Set GOOGLE_APPLICATION_CREDENTIALS and verify service account permissions")
        except Exception as e:
            logger.error(f"Client initialization failed: {e}")
            raise
    
    async def execute_enterprise_scan(self) -> Dict[str, Any]:
        start_time = time.time()
        projects = await self._discover_all_projects()
        
        all_detections = []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            project_futures = {
                executor.submit(self._scan_project_sync, project_id): project_id 
                for project_id in projects[:50]
            }
            
            for future in as_completed(project_futures):
                try:
                    project_detections = future.result()
                    all_detections.extend(project_detections)
                    self.scan_stats['projects_processed'] += 1
                except Exception as e:
                    self.scan_stats['errors_handled'] += 1
                    logger.warning(f"Project scan failed: {e}")
        
        execution_time = time.time() - start_time
        
        return {
            'ao1_assessment': self._generate_ao1_assessment(all_detections),
            'visibility_detections': all_detections,
            'scan_performance': {
                **self.scan_stats,
                'execution_time_seconds': execution_time,
                'tables_per_second': self.scan_stats['tables_processed'] / max(execution_time, 1)
            },
            'enterprise_summary': self._create_enterprise_summary(all_detections)
        }
    
    async def _discover_all_projects(self) -> List[str]:
        try:
            request = resourcemanager_v3.ListProjectsRequest()
            projects = list(self.resource_client.list_projects(request=request))
            active_projects = [p.project_id for p in projects if p.state == resourcemanager_v3.Project.State.ACTIVE]
            logger.info(f"Discovered {len(active_projects)} active projects")
            return active_projects
        except Exception as e:
            logger.error(f"Project discovery failed: {e}")
            return []
    
    def _scan_project_sync(self, project_id: str) -> List[Dict]:
        detections = []
        try:
            logger.info(f"Scanning project: {project_id}")
            datasets = list(self.bq_client.list_datasets(project_id))
            
            for dataset in datasets[:20]:
                self.scan_stats['datasets_processed'] += 1
                try:
                    dataset_detections = self._scan_dataset_resilient(project_id, dataset.dataset_id)
                    detections.extend(dataset_detections)
                except Exception as e:
                    self._handle_scan_error(f"{project_id}.{dataset.dataset_id}", e)
                    
        except Exception as e:
            self._handle_scan_error(project_id, e)
        
        logger.info(f"Project {project_id}: {len(detections)} detections found")
        return detections
    
    def _scan_dataset_resilient(self, project_id: str, dataset_id: str) -> List[Dict]:
        detections = []
        try:
            dataset_ref = f"{project_id}.{dataset_id}"
            tables = list(self.bq_client.list_tables(dataset_ref))
            
            for table in tables[:100]:
                if table.table_type == 'TABLE':
                    self.scan_stats['tables_processed'] += 1
                    try:
                        table_detections = self._scan_table_quantum_resilient(project_id, dataset_id, table.table_id)
                        detections.extend(table_detections)
                    except Exception as e:
                        self._handle_scan_error(f"{project_id}.{dataset_id}.{table.table_id}", e)
                        
        except Exception as e:
            self._handle_scan_error(f"{project_id}.{dataset_id}", e)
        
        return detections
    
    def _scan_table_quantum_resilient(self, project_id: str, dataset_id: str, table_id: str) -> List[Dict]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self._scan_table_quantum_attempt(project_id, dataset_id, table_id)
            except Exception as e:
                error_msg = str(e).lower()
                
                if 'permission' in error_msg or 'forbidden' in error_msg or '403' in error_msg:
                    self.scan_stats['permission_errors'] += 1
                    logger.debug(f"Permission denied: {project_id}.{dataset_id}.{table_id}")
                    return []
                
                elif 'quota' in error_msg or 'limit' in error_msg or '429' in error_msg:
                    self.scan_stats['quota_errors'] += 1
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + (attempt * 0.5)
                        logger.debug(f"Quota limit hit, waiting {wait_time}s before retry")
                        time.sleep(wait_time)
                        continue
                    return []
                
                elif 'network' in error_msg or 'connection' in error_msg or 'timeout' in error_msg:
                    self.scan_stats['network_errors'] += 1
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt)
                        logger.debug(f"Network error, retrying in {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    return []
                
                else:
                    if attempt == max_retries - 1:
                        self._handle_scan_error(f"{project_id}.{dataset_id}.{table_id}", e)
                    return []
        
        return []
    
    def _scan_table_quantum_attempt(self, project_id: str, dataset_id: str, table_id: str) -> List[Dict]:
        query = f"""
        SELECT * FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE RAND() < 0.001
        LIMIT 500
        """
        
        job_config = bigquery.QueryJobConfig()
        job_config.maximum_bytes_billed = 25 * 1024 * 1024
        job_config.job_timeout_ms = 120000
        job_config.use_query_cache = True
        job_config.use_legacy_sql = False
        
        query_job = self.bq_client.query(query, job_config=job_config)
        df = query_job.to_dataframe()
        
        if df.empty:
            return []
        
        detections = []
        for column in df.columns:
            try:
                source_ref = f"{project_id}.{dataset_id}.{table_id}.{column}"
                quantum_detections = self.quantum_engine.quantum_extract(df[column], source_ref)
                
                for detection in quantum_detections:
                    detections.append({
                        'project': project_id,
                        'dataset': dataset_id,
                        'table': table_id,
                        'column': column,
                        'metric': detection.metric,
                        'confidence': detection.confidence,
                        'evidence_count': len(detection.evidence),
                        'extraction_method': detection.extraction_method,
                        'sample_evidence': detection.evidence[:3]
                    })
                    self.scan_stats['detections_found'] += 1
            except Exception as e:
                logger.debug(f"Column analysis failed {project_id}.{dataset_id}.{table_id}.{column}: {e}")
                continue
        
        return detections
    
    def _handle_scan_error(self, resource: str, error: Exception):
        self.scan_stats['errors_handled'] += 1
        error_msg = str(error).lower()
        
        error_category = "unknown"
        if 'permission' in error_msg or 'forbidden' in error_msg:
            error_category = "permission_denied"
        elif 'quota' in error_msg or 'limit' in error_msg:
            error_category = "quota_exceeded"
        elif 'not found' in error_msg or '404' in error_msg:
            error_category = "not_found"
        elif 'network' in error_msg or 'timeout' in error_msg:
            error_category = "network_error"
        
        self.failed_resources.append({
            'resource': resource,
            'error_category': error_category,
            'error_message': str(error)[:100]
        })
        
        logger.debug(f"Scan error ({error_category}): {resource} - {str(error)[:100]}")
    
    def _generate_ao1_assessment(self, detections: List[Dict]) -> Dict[str, Any]:
        coverage_metrics = {
            'network_presence': 0,
            'endpoint_identity': 0,
            'identity_context': 0,
            'application_telemetry': 0,
            'cloud_infrastructure': 0
        }
        
        high_confidence_threshold = 0.7
        
        for detection in detections:
            if detection['confidence'] >= high_confidence_threshold:
                metric = detection['metric']
                if metric in coverage_metrics:
                    coverage_metrics[metric] += 1
        
        ao1_readiness = {}
        for metric, count in coverage_metrics.items():
            ao1_readiness[metric] = {
                'status': 'READY' if count >= 5 else 'PARTIAL' if count >= 2 else 'NOT_READY',
                'high_confidence_sources': count,
                'coverage_score': min(count / 10, 1.0) * 100
            }
        
        ready_areas = sum(1 for area in ao1_readiness.values() if area['status'] == 'READY')
        overall_score = (ready_areas / 5) * 100
        
        return {
            'overall_ao1_readiness': {
                'percentage': overall_score,
                'status': 'READY' if ready_areas >= 4 else 'PARTIAL' if ready_areas >= 2 else 'NOT_READY',
                'ready_areas': f"{ready_areas}/5"
            },
            'coverage_breakdown': ao1_readiness,
            'recommendation': self._get_ao1_recommendation(ready_areas, overall_score)
        }
    
    def _get_ao1_recommendation(self, ready_areas: int, overall_score: float) -> str:
        if ready_areas >= 4:
            return "AO1 implementation ready - sufficient visibility coverage detected"
        elif ready_areas >= 2:
            return "Partial AO1 readiness - recommend expanding data collection in weak areas"
        else:
            return "AO1 implementation not recommended - insufficient visibility coverage"
    
    def _create_enterprise_summary(self, detections: List[Dict]) -> Dict[str, Any]:
        projects_with_data = set(d['project'] for d in detections)
        
        return {
            'data_rich_projects': len(projects_with_data),
            'total_unique_sources': len(set(f"{d['project']}.{d['dataset']}.{d['table']}" for d in detections)),
            'scan_reliability': {
                'success_rate': ((self.scan_stats['tables_processed'] - self.scan_stats['errors_handled']) / max(self.scan_stats['tables_processed'], 1)) * 100,
                'permission_failures': self.scan_stats['permission_errors'],
                'quota_failures': self.scan_stats['quota_errors'],
                'network_failures': self.scan_stats['network_errors']
            },
            'failed_resources_sample': self.failed_resources[:10]
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str = "ao1_enterprise_scan.json"):
        os.makedirs('outputs', exist_ok=True)
        output_file = os.path.join('outputs', output_path)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self._print_executive_summary(results)
        logger.info(f"Complete results saved to {output_file}")
    
    def _print_executive_summary(self, results: Dict[str, Any]):
        print("\n" + "="*80)
        print("🚀 AO1 ENTERPRISE BIGQUERY VISIBILITY ASSESSMENT")
        print("="*80)
        
        ao1 = results['ao1_assessment']['overall_ao1_readiness']
        print(f"AO1 Readiness: {ao1['percentage']:.1f}% ({ao1['status']})")
        print(f"Coverage Areas Ready: {ao1['ready_areas']}")
        print(f"Recommendation: {results['ao1_assessment']['recommendation']}")
        
        print(f"\nScan Performance:")
        perf = results['scan_performance']
        print(f"  Projects: {perf['projects_processed']}")
        print(f"  Tables: {perf['tables_processed']}")
        print(f"  Detections: {perf['detections_found']}")
        print(f"  Speed: {perf['tables_per_second']:.1f} tables/sec")
        print(f"  Duration: {perf['execution_time_seconds']:.1f} seconds")
        
        reliability = results['enterprise_summary']['scan_reliability']
        print(f"\nScan Reliability:")
        print(f"  Success Rate: {reliability['success_rate']:.1f}%")
        print(f"  Permission Errors: {reliability['permission_failures']}")
        print(f"  Quota Errors: {reliability['quota_failures']}")
        print(f"  Network Errors: {reliability['network_failures']}")
        
        print(f"\nCoverage Breakdown:")
        for metric, data in results['ao1_assessment']['coverage_breakdown'].items():
            status_icon = "✅" if data['status'] == 'READY' else "⚠️" if data['status'] == 'PARTIAL' else "❌"
            print(f"  {status_icon} {metric}: {data['coverage_score']:.0f}% ({data['high_confidence_sources']} sources)")
        
        if results['enterprise_summary']['failed_resources_sample']:
            print(f"\nConnection Issues (Sample):")
            for failure in results['enterprise_summary']['failed_resources_sample'][:3]:
                print(f"  ⚠️ {failure['resource']} ({failure['error_category']})")
        
        print("="*80)

async def main():
    scanner = HyperScaleBigQueryScanner()
    results = await scanner.execute_enterprise_scan()
    scanner.save_results(results)

if __name__ == "__main__":
    asyncio.run(main())
