import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Any
from google.cloud import bigquery
from google.cloud import resourcemanager_v3
import logging
from secure_config import DataPrivacy, PerformanceConfig
import asyncio
import base64
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from quantum_engine import QuantumAO1Engine

logger = logging.getLogger(__name__)

class ValueContentScanner:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.bq_client = bigquery.Client()
        self.resource_client = resourcemanager_v3.ProjectsClient()
        self.quantum_engine = QuantumAO1Engine()
        
    async def scan_all_values_for_ao1(self) -> Dict:
        projects = self._get_all_projects()
        
        all_findings = {
            'network_coverage': [],
            'endpoint_coverage': [],
            'identity_coverage': [],
            'application_coverage': [],
            'cloud_coverage': [],
            'scan_metadata': {
                'total_projects_scanned': 0,
                'tables_analyzed': 0,
                'columns_analyzed': 0
            }
        }
        
        for project_id in projects[:PerformanceConfig.MAX_CONCURRENT_SCANS]:
            try:
                project_results = await self._scan_project_values(project_id)
                for coverage_type, findings in project_results.items():
                    if coverage_type in all_findings and isinstance(findings, list):
                        all_findings[coverage_type].extend(findings)
                
                all_findings['scan_metadata']['total_projects_scanned'] += 1
                
            except Exception as e:
                logger.warning(f"Project value scan failed {project_id}: {e}")
                continue
        
        all_findings['value_summary'] = self._generate_value_summary(all_findings)
        return all_findings

    async def _scan_project_values(self, project_id: str) -> Dict:
        project_findings = {
            'network_coverage': [],
            'endpoint_coverage': [],
            'identity_coverage': [],
            'application_coverage': [],
            'cloud_coverage': []
        }
        
        datasets = self._list_datasets_safe(project_id)
        
        for dataset_id in datasets:
            tables = self._list_tables_safe(project_id, dataset_id)
            
            for table_id in tables:
                try:
                    table_findings = await self._scan_table_values(project_id, dataset_id, table_id)
                    for coverage_type, findings in table_findings.items():
                        if coverage_type in project_findings:
                            project_findings[coverage_type].extend(findings)
                except Exception as e:
                    logger.debug(f"Table value scan failed {project_id}.{dataset_id}.{table_id}: {e}")
                    continue
        
        return project_findings

    async def _scan_table_values(self, project_id: str, dataset_id: str, table_id: str) -> Dict:
        findings = {
            'network_coverage': [],
            'endpoint_coverage': [],
            'identity_coverage': [],
            'application_coverage': [],
            'cloud_coverage': []
        }
        
        sample_data = await self._sample_table_raw_values(project_id, dataset_id, table_id)
        if sample_data is None or sample_data.empty:
            return findings
        
        for column in sample_data.columns:
            source_ref = f"{project_id}.{dataset_id}.{table_id}.{column}"
            quantum_detections = self.quantum_engine.quantum_extract(sample_data[column], source_ref)
            
            for detection in quantum_detections:
                coverage_type = self._map_metric_to_coverage(detection.metric)
                if coverage_type in findings:
                    finding = {
                        'project': project_id,
                        'dataset': dataset_id,
                        'table': table_id,
                        'column': column,
                        'coverage_type': coverage_type,
                        'detector': detection.extraction_method,
                        'confidence': detection.confidence,
                        'match_count': len(detection.evidence),
                        'sample_matches': DataPrivacy.sanitize_evidence(detection.evidence)
                    }
                    findings[coverage_type].append(finding)
        
        return findings

    async def _sample_table_raw_values(self, project_id: str, dataset_id: str, table_id: str) -> pd.DataFrame:
        try:
            query = f"""
            SELECT *
            FROM `{project_id}.{dataset_id}.{table_id}`
            WHERE RAND() < {PerformanceConfig.SAMPLE_RATE}
            LIMIT {PerformanceConfig.MAX_ROWS_PER_TABLE}
            """
            
            job_config = bigquery.QueryJobConfig()
            job_config.maximum_bytes_billed = PerformanceConfig.MAX_BYTES_BILLED
            job_config.job_timeout_ms = PerformanceConfig.QUERY_TIMEOUT_MS
            
            query_job = self.bq_client.query(query, job_config=job_config)
            df = query_job.to_dataframe()
            
            return df if not df.empty else None
            
        except Exception as e:
            logger.debug(f"Raw sampling failed {project_id}.{dataset_id}.{table_id}: {e}")
            return None

    def _map_metric_to_coverage(self, metric: str) -> str:
        mapping = {
            'network_presence': 'network_coverage',
            'endpoint_identity': 'endpoint_coverage',
            'identity_context': 'identity_coverage',
            'application_telemetry': 'application_coverage',
            'cloud_infrastructure': 'cloud_coverage'
        }
        return mapping.get(metric, 'application_coverage')

    def _generate_value_summary(self, findings: Dict) -> Dict:
        summary = {}
        
        for coverage_type in ['network_coverage', 'endpoint_coverage', 'identity_coverage', 'application_coverage', 'cloud_coverage']:
            coverage_findings = findings.get(coverage_type, [])
            
            if coverage_findings:
                high_confidence = [f for f in coverage_findings if f['confidence'] > 0.7]
                
                summary[coverage_type] = {
                    'ao1_ready': len(high_confidence) >= 3,
                    'total_findings': len(coverage_findings),
                    'high_confidence': len(high_confidence),
                    'unique_tables': len(set(f"{f['project']}.{f['dataset']}.{f['table']}" for f in coverage_findings))
                }
            else:
                summary[coverage_type] = {
                    'ao1_ready': False,
                    'total_findings': 0,
                    'high_confidence': 0,
                    'unique_tables': 0
                }
        
        ready_areas = sum(1 for area in summary.values() if area.get('ao1_ready', False))
        
        summary['overall_ao1_readiness'] = {
            'ready_percentage': (ready_areas / 5) * 100,
            'areas_ready': f"{ready_areas}/5",
            'status': 'READY' if ready_areas >= 4 else 'PARTIAL' if ready_areas >= 2 else 'NOT_READY'
        }
        
        return summary

    def _get_all_projects(self) -> List[str]:
        try:
            request = resourcemanager_v3.ListProjectsRequest()
            projects = list(self.resource_client.list_projects(request=request))
            active_projects = [p.project_id for p in projects if p.state == resourcemanager_v3.Project.State.ACTIVE]
            return active_projects
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []

    def _list_datasets_safe(self, project_id: str) -> List[str]:
        try:
            datasets = list(self.bq_client.list_datasets(project_id))
            return [d.dataset_id for d in datasets]
        except Exception:
            return []

    def _list_tables_safe(self, project_id: str, dataset_id: str) -> List[str]:
        try:
            tables = list(self.bq_client.list_tables(f"{project_id}.{dataset_id}"))
            return [t.table_id for t in tables if t.table_type == 'TABLE']
        except Exception:
            return []

from resilience_manager import resilience_manager
from performance_monitor import performance_monitor

class ValueContentScanner:
    async def _scan_table_values_resilient(self, project_id: str, dataset_id: str, table_id: str):
        resource_id = f"{project_id}.{dataset_id}.{table_id}"
        operation_id = f"scan_{int(time.time())}_{resource_id}"
        
        metrics = performance_monitor.start_operation(operation_id, "table_scan", resource_id)
        
        try:
            async def scan_operation():
                return await self._scan_table_values(project_id, dataset_id, table_id)
            
            result = await resilience_manager.execute_with_resilience(
                scan_operation, resource_id, "table_scan"
            )
            
            performance_monitor.update_operation(operation_id, 
                rows_scanned=len

from resilience_manager import resilience_manager
from performance_monitor import performance_monitor

class ValueContentScanner:
    async def _scan_table_values_resilient(self, project_id: str, dataset_id: str, table_id: str):
        resource_id = f"{project_id}.{dataset_id}.{table_id}"
        operation_id = f"scan_{int(time.time())}_{resource_id}"
        
        metrics = performance_monitor.start_operation(operation_id, "table_scan", resource_id)
        
        try:
            async def scan_operation():
                return await self._scan_table_values(project_id, dataset_id, table_id)
            
            result = await resilience_manager.execute_with_resilience(
                scan_operation, resource_id, "table_scan"
            )
            
            performance_monitor.update_operation(operation_id, 
                rows_scanned=len
