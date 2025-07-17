import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Any
from google.cloud import bigquery
from google.cloud import resourcemanager_v3
import logging
from secure_config import DataPrivacy
import asyncio
import base64
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)

class ValueContentScanner:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.bq_client = bigquery.Client()
        self.resource_client = resourcemanager_v3.ProjectsClient()
        
        self.ao1_value_detectors = {
            'network_ips': {
                'patterns': [
                    r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                    r'\b([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
                    r'\b([0-9a-fA-F]{1,4}:){1,7}::\b',
                    r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}\b'
                ],
                'weight': 1.0
            },
            'hostnames': {
                'patterns': [
                    r'\b[A-Z]{2,4}-[A-Z0-9]+-\d+\b',
                    r'\b[a-zA-Z]+-[a-zA-Z]+-\d+\b',
                    r'\b[a-zA-Z]+\d+\.[a-zA-Z]+\.local\b',
                    r'\bWS-[A-Z0-9]+-\d+\b',
                    r'\bSRV-[A-Z0-9]+-\d+\b',
                    r'\bPC-[A-Z0-9]+-\d+\b',
                    r'\bDESKTOP-[A-Z0-9]+\b',
                    r'\bLAPTOP-[A-Z0-9]+\b',
                    r'\b[a-zA-Z]{3,8}\d{3,6}\b'
                ],
                'weight': 1.0
            },
            'domains': {
                'patterns': [
                    r'\b[a-zA-Z0-9][a-zA-Z0-9\-\.]*\.[a-zA-Z]{2,}\b',
                    r'\bhttps?://[^\s]+\b',
                    r'\b[a-zA-Z0-9\-]+\.local\b',
                    r'\b[a-zA-Z0-9\-]+\.corp\b',
                    r'\b[a-zA-Z0-9\-]+\.internal\b'
                ],
                'weight': 0.9
            },
            'usernames': {
                'patterns': [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    r'\bdomain\\[a-zA-Z0-9_\-]+\b',
                    r'\b[a-zA-Z]+\.[a-zA-Z]+@[a-zA-Z]+\.[a-zA-Z]+\b',
                    r'\b[a-zA-Z][a-zA-Z0-9_\-\.]{2,30}\b'
                ],
                'weight': 0.8
            },
            'event_codes': {
                'patterns': [
                    r'\bevent[_\-]?id[:\s]*\d+\b',
                    r'\bevent[_\-]?code[:\s]*\d+\b',
                    r'\b(4624|4625|4648|4672|4768|4769|4776|4778|4779)\b',
                    r'\b[45]\d{2}\b',
                    r'\bstatus[_\-]?code[:\s]*\d+\b'
                ],
                'weight': 0.9
            },
            'cloud_resources': {
                'patterns': [
                    r'\bi-[0-9a-f]{8,17}\b',
                    r'\bvol-[0-9a-f]{8,17}\b',
                    r'\bsubnet-[0-9a-f]{8,17}\b',
                    r'\bvpc-[0-9a-f]{8,17}\b',
                    r'\bsg-[0-9a-f]{8,17}\b',
                    r'\bami-[0-9a-f]{8,17}\b',
                    r'\b/subscriptions/[0-9a-f\-]{36}\b',
                    r'\bus-east-1[a-z]?\b',
                    r'\bus-west-2[a-z]?\b',
                    r'\beu-west-1[a-z]?\b'
                ],
                'weight': 1.0
            }
        }

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
        
        for project_id in projects:
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
            column_findings = self._analyze_raw_values(sample_data[column], project_id, dataset_id, table_id, column)
            for finding in column_findings:
                coverage_type = finding['coverage_type']
                if coverage_type in findings:
                    findings[coverage_type].append(finding)
        
        return findings

    def _analyze_raw_values(self, series: pd.Series, project_id: str, dataset_id: str, table_id: str, column_name: str) -> List[Dict]:
        findings = []
        
        if len(series) == 0:
            return findings
        
        clean_data = series.dropna()
        if len(clean_data) == 0:
            return findings
        
        raw_values = []
        for value in clean_data.head(1000):
            try:
                str_value = str(value)
                raw_values.append(str_value)
                
                if self._looks_like_json(str_value):
                    json_content = self._extract_json_content(str_value)
                    if json_content:
                        raw_values.extend(json_content)
                
                if self._looks_like_xml(str_value):
                    xml_content = self._extract_xml_content(str_value)
                    if xml_content:
                        raw_values.extend(xml_content)
                
                if self._looks_like_base64(str_value):
                    decoded_content = self._decode_base64_safe(str_value)
                    if decoded_content:
                        raw_values.append(decoded_content)
                        
            except Exception:
                continue
        
        all_content = ' '.join(raw_values)
        
        detection_results = {}
        for detector_name, detector_config in self.ao1_value_detectors.items():
            match_count = 0
            total_matches = []
            
            for pattern in detector_config['patterns']:
                try:
                    matches = re.findall(pattern, all_content, re.IGNORECASE)
                    match_count += len(matches)
                    total_matches.extend(matches)
                except Exception:
                    continue
            
            if match_count > 0:
                confidence = min((match_count / len(raw_values)) * detector_config['weight'], 1.0)
                
                if confidence > 0.05:
                    detection_results[detector_name] = {
                        'confidence': confidence,
                        'match_count': match_count,
                        'sample_matches': total_matches[:5]
                    }
        
        for detector_name, result in detection_results.items():
            if result['confidence'] > 0.05:
                coverage_type = self._map_detector_to_coverage(detector_name)
                
                finding = {
                    'project': project_id,
                    'dataset': dataset_id,
                    'table': table_id,
                    'column': column_name,
                    'coverage_type': coverage_type,
                    'detector': detector_name,
                    'confidence': result['confidence'],
                    'match_count': result['match_count'],
                    'sample_matches': DataPrivacy.sanitize_column_sample(result['sample_matches'])
                }
                findings.append(finding)
        
        return findings

    async def _sample_table_raw_values(self, project_id: str, dataset_id: str, table_id: str) -> pd.DataFrame:
        try:
            query = f"""
            SELECT *
            FROM `{project_id}.{dataset_id}.{table_id}`
            WHERE RAND() < 0.005
            LIMIT 2000
            """
            
            job_config = bigquery.QueryJobConfig()
            job_config.maximum_bytes_billed = 200 * 1024 * 1024
            job_config.job_timeout_ms = 300000
            
            query_job = self.bq_client.query(query, job_config=job_config)
            df = query_job.to_dataframe()
            
            return df if not df.empty else None
            
        except Exception as e:
            logger.debug(f"Raw sampling failed {project_id}.{dataset_id}.{table_id}: {e}")
            return None

    def _looks_like_json(self, value: str) -> bool:
        return (value.strip().startswith('{') and value.strip().endswith('}')) or \
               (value.strip().startswith('[') and value.strip().endswith(']'))

    def _extract_json_content(self, value: str) -> List[str]:
        try:
            data = json.loads(value)
            content = []
            self._extract_json_values(data, content)
            return content
        except:
            return []

    def _extract_json_values(self, obj: Any, content: List[str]):
        if isinstance(obj, dict):
            for key, val in obj.items():
                content.append(str(key))
                self._extract_json_values(val, content)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_json_values(item, content)
        else:
            content.append(str(obj))

    def _looks_like_xml(self, value: str) -> bool:
        return value.strip().startswith('<') and value.strip().endswith('>')

    def _extract_xml_content(self, value: str) -> List[str]:
        try:
            root = ET.fromstring(value)
            content = []
            for elem in root.iter():
                if elem.text:
                    content.append(elem.text.strip())
                if elem.tail:
                    content.append(elem.tail.strip())
                content.append(elem.tag)
            return [c for c in content if c]
        except:
            return []

    def _looks_like_base64(self, value: str) -> bool:
        if len(value) < 4 or len(value) % 4 != 0:
            return False
        try:
            base64.b64decode(value, validate=True)
            return True
        except:
            return False

    def _decode_base64_safe(self, value: str) -> str:
        try:
            decoded = base64.b64decode(value).decode('utf-8', errors='ignore')
            return decoded
        except:
            return None

    def _map_detector_to_coverage(self, detector_name: str) -> str:
        mapping = {
            'network_ips': 'network_coverage',
            'domains': 'network_coverage',
            'hostnames': 'endpoint_coverage',
            'usernames': 'identity_coverage',
            'event_codes': 'application_coverage',
            'cloud_resources': 'cloud_coverage'
        }
        return mapping.get(detector_name, 'application_coverage')

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
