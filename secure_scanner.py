import logging
import asyncio
from typing import List, Dict
from google.cloud import bigquery
from google.cloud import resourcemanager_v3
from database import db
from secure_config import SecurityConfig, DataPrivacy
from content_value_scanner import ValueContentScanner
import pandas as pd

logger = logging.getLogger(__name__)

class SecureBigQueryScanner:
    def __init__(self, user_id: str):
        SecurityConfig.validate_config()
        self.user_id = user_id
        self.value_scanner = ValueContentScanner(user_id)
        
    async def scan_user_projects(self) -> Dict:
        logger.info(f"Starting comprehensive value-based scan for user {self.user_id}")
        
        results = await self.value_scanner.scan_all_values_for_ao1()
        
        return {
            'mappings': self._format_mappings_for_compatibility(results),
            'errors': [],
            'user_id': self.user_id,
            'scan_type': 'comprehensive_value_analysis',
            'ao1_summary': results.get('value_summary', {}),
            'projects_scanned': results.get('scan_metadata', {}).get('total_projects_scanned', 0)
        }
    
    def _format_mappings_for_compatibility(self, results: Dict) -> List[Dict]:
        mappings = []
        
        for coverage_type in ['network_coverage', 'endpoint_coverage', 'identity_coverage', 'application_coverage', 'cloud_coverage']:
            findings = results.get(coverage_type, [])
            
            for finding in findings:
                mapping = {
                    'project': finding['project'],
                    'dataset': finding['dataset'],
                    'table': finding['table'],
                    'column': finding['column'],
                    'source': f"{finding['project']}.{finding['dataset']}.{finding['table']}.{finding['column']}",
                    'metric': coverage_type,
                    'confidence': finding['confidence'],
                    'pattern_type': finding['detector'],
                    'detection_methods': ['value_content_analysis'],
                    'sample_values': finding.get('sample_matches', []),
                    'full_table_name': f"{finding['project']}.{finding['dataset']}.{finding['table']}",
                    'user_id': self.user_id,
                    'ao1_detector': finding['detector'],
                    'match_count': finding.get('match_count', 0)
                }
                mappings.append(mapping)
        
        return sorted(mappings, key=lambda x: x['confidence'], reverse=True)
