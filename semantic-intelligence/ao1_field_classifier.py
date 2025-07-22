#!/usr/bin/env python3

from typing import List, Dict, Set
from models import FieldIntelligence
from dataclasses import dataclass

@dataclass
class AO1FieldClassification:
    asset_fields: List[FieldIntelligence]
    logging_fields: List[FieldIntelligence] 
    time_fields: List[FieldIntelligence]
    platform_fields: List[FieldIntelligence]
    infrastructure_fields: List[FieldIntelligence]
    role_fields: List[FieldIntelligence]
    log_type_fields: List[FieldIntelligence]
    cmdb_fields: List[FieldIntelligence]
    security_control_fields: List[FieldIntelligence]

class AO1FieldClassifier:
    """
    Classifies fields specifically for AO1 Log Visibility requirements.
    Uses both semantic analysis AND your actual table structure.
    """
    
    def __init__(self):
        # Known patterns from your database schema
        self.known_patterns = {
            'asset_identifiers': {
                'names': ['hostname', 'device_hostname', 'asset', 'ci_name', 'host', 'endpoint_nme', 'agent_nme'],
                'tables': ['cmdb', 'all_sources', 'appmap', 'chronicle', 'crowdstrike']
            },
            'logging_activity': {
                'names': ['log_count', 'event_count', 'log_type', 'message', 'data', 'source'],
                'tables': ['chronicle', 'all_sources', 'splunk', 'log_type_priority']
            },
            'platforms': {
                'names': ['sourcetype', 'source', 'index', 'platform', 'tool'],
                'tables': ['splunk', 'all_sources', 'splunk_mapping']
            },
            'time_fields': {
                'names': ['timestamp', '_time', 'created', 'occurred', 'date'],
                'tables': ['chronicle', 'all_sources', 'splunk']
            },
            'cmdb_data': {
                'names': ['ci_name', 'asset', 'class', 'class_type', 'name', 'status'],
                'tables': ['cmdb', 'all_sources']
            }
        }
        
    def classify_fields_for_ao1(self, fields: List[FieldIntelligence]) -> AO1FieldClassification:
        """
        Classify fields into AO1-relevant categories using intelligence + your schema knowledge
        """
        
        classification = AO1FieldClassification(
            asset_fields=[],
            logging_fields=[],
            time_fields=[],
            platform_fields=[],
            infrastructure_fields=[],
            role_fields=[],
            log_type_fields=[],
            cmdb_fields=[],
            security_control_fields=[]
        )
        
        for field in fields:
            # Only consider fields with reasonable intelligence
            if field.intelligence_score < 0.4:
                continue
                
            field_categories = self._categorize_field(field)
            
            # Assign to appropriate categories
            if 'asset' in field_categories:
                classification.asset_fields.append(field)
            if 'logging' in field_categories:
                classification.logging_fields.append(field)
            if 'time' in field_categories:
                classification.time_fields.append(field)
            if 'platform' in field_categories:
                classification.platform_fields.append(field)
            if 'infrastructure' in field_categories:
                classification.infrastructure_fields.append(field)
            if 'role' in field_categories:
                classification.role_fields.append(field)
            if 'log_type' in field_categories:
                classification.log_type_fields.append(field)
            if 'cmdb' in field_categories:
                classification.cmdb_fields.append(field)
            if 'security_control' in field_categories:
                classification.security_control_fields.append(field)
        
        # Sort by intelligence score
        for field_list in [
            classification.asset_fields, classification.logging_fields, 
            classification.time_fields, classification.platform_fields,
            classification.infrastructure_fields, classification.role_fields,
            classification.log_type_fields, classification.cmdb_fields,
            classification.security_control_fields
        ]:
            field_list.sort(key=lambda f: f.intelligence_score, reverse=True)
            
        return classification
    
    def _categorize_field(self, field: FieldIntelligence) -> Set[str]:
        """Determine which AO1 categories this field belongs to"""
        categories = set()
        
        name_lower = field.name.lower()
        table_lower = field.table.lower()
        
        # Asset identification
        if self._is_asset_field(field):
            categories.add('asset')
            
        # Logging activity
        if self._is_logging_field(field):
            categories.add('logging')
            
        # Temporal data
        if self._is_time_field(field):
            categories.add('time')
            
        # Platform/source identification
        if self._is_platform_field(field):
            categories.add('platform')
            
        # Infrastructure classification
        if self._is_infrastructure_field(field):
            categories.add('infrastructure')
            
        # Role/function classification
        if self._is_role_field(field):
            categories.add('role')
            
        # Log type classification
        if self._is_log_type_field(field):
            categories.add('log_type')
            
        # CMDB data
        if self._is_cmdb_field(field):
            categories.add('cmdb')
            
        # Security controls
        if self._is_security_control_field(field):
            categories.add('security_control')
            
        return categories
    
    def _is_asset_field(self, field: FieldIntelligence) -> bool:
        """Determine if field represents asset identifiers"""
        name_lower = field.name.lower()
        table_lower = field.table.lower()
        
        # Known asset patterns from your schema
        asset_patterns = [
            'hostname', 'device_hostname', 'asset', 'ci_name', 'host', 
            'endpoint_nme', 'agent_nme', 'server', 'computer', 'device'
        ]
        
        # Direct name match
        if any(pattern in name_lower for pattern in asset_patterns):
            return True
            
        # Table context clues
        if table_lower == 'cmdb' and any(kw in name_lower for kw in ['name', 'id']):
            return True
            
        # High uniqueness suggests identifier
        if (field.semantic_profile and 
            field.semantic_profile.behavioral_indicators and
            field.semantic_profile.behavioral_indicators.get('variability', 0) > 0.7):
            if any(kw in name_lower for kw in ['name', 'id', 'key']):
                return True
                
        return False
    
    def _is_logging_field(self, field: FieldIntelligence) -> bool:
        """Determine if field represents logging activity"""
        name_lower = field.name.lower()
        table_lower = field.table.lower()
        
        # Direct logging indicators
        logging_patterns = [
            'log_count', 'event_count', 'log', 'event', 'message', 
            'data', 'record', 'entry', 'activity'
        ]
        
        if any(pattern in name_lower for pattern in logging_patterns):
            return True
            
        # Table context - if it's in a logging table, likely logging data
        if table_lower in ['chronicle', 'splunk', 'log_type_priority']:
            return True
            
        # High variability suggests event data
        if (field.semantic_profile and 
            field.semantic_profile.behavioral_indicators and
            field.semantic_profile.behavioral_indicators.get('variability', 0) > 0.6):
            return True
            
        return False
    
    def _is_time_field(self, field: FieldIntelligence) -> bool:
        """Determine if field represents timestamps"""
        name_lower = field.name.lower()
        
        time_patterns = [
            'timestamp', '_time', 'time', 'date', 'created', 'occurred', 
            'modified', '_at', 'when'
        ]
        
        # Check for temporal patterns in semantic profile
        if (field.semantic_profile and 
            field.semantic_profile.pattern_features and
            'temporal_patterns' in field.semantic_profile.pattern_features):
            temporal_scores = field.semantic_profile.pattern_features['temporal_patterns']
            if temporal_scores and max(temporal_scores) > 0.3:
                return True
                
        return any(pattern in name_lower for pattern in time_patterns)
    
    def _is_platform_field(self, field: FieldIntelligence) -> bool:
        """Determine if field represents platforms/sources"""
        name_lower = field.name.lower()
        table_lower = field.table.lower()
        
        platform_patterns = [
            'sourcetype', 'source', 'platform', 'tool', 'system', 
            'index', 'provider'
        ]
        
        if any(pattern in name_lower for pattern in platform_patterns):
            return True
            
        # Check sample values for platform names
        if field.sample_values:
            sample_text = ' '.join(str(v).lower() for v in field.sample_values[:20])
            platform_names = [
                'splunk', 'chronicle', 'bigquery', 'crowdstrike', 'theom', 
                'wiz', 'axonius', 'tanium', 'edr', 'siem'
            ]
            if any(platform in sample_text for platform in platform_names):
                return True
                
        return False
    
    def _is_infrastructure_field(self, field: FieldIntelligence) -> bool:
        """Determine if field classifies infrastructure types"""
        name_lower = field.name.lower()
        
        infra_patterns = [
            'environment', 'env', 'tier', 'region', 'zone', 'datacenter', 
            'location', 'site', 'type', 'category'
        ]
        
        if any(pattern in name_lower for pattern in infra_patterns):
            # Check sample values for infrastructure terms
            if field.sample_values:
                sample_text = ' '.join(str(v).lower() for v in field.sample_values[:20])
                infra_terms = [
                    'cloud', 'onprem', 'saas', 'api', 'aws', 'azure', 'gcp', 
                    'datacenter', 'prod', 'dev', 'test'
                ]
                if any(term in sample_text for term in infra_terms):
                    return True
                    
        return False
    
    def _is_role_field(self, field: FieldIntelligence) -> bool:
        """Determine if field defines asset roles/functions"""
        name_lower = field.name.lower()
        
        role_patterns = [
            'role', 'function', 'service', 'type', 'category', 'class'
        ]
        
        if any(pattern in name_lower for pattern in role_patterns):
            # Check sample values for role terms
            if field.sample_values:
                sample_text = ' '.join(str(v).lower() for v in field.sample_values[:20])
                role_terms = [
                    'network', 'endpoint', 'cloud', 'application', 'identity', 
                    'server', 'workstation', 'firewall', 'router'
                ]
                if any(term in sample_text for term in role_terms):
                    return True
                    
        return False
    
    def _is_log_type_field(self, field: FieldIntelligence) -> bool:
        """Determine if field represents log types"""
        name_lower = field.name.lower()
        
        log_type_patterns = [
            'log_type', 'event_type', 'sourcetype', 'normal_type', 
            'category', 'class'
        ]
        
        if any(pattern in name_lower for pattern in log_type_patterns):
            # Check sample values for log type terms
            if field.sample_values:
                sample_text = ' '.join(str(v).lower() for v in field.sample_values[:20])
                log_terms = [
                    'firewall', 'dns', 'http', 'auth', 'os', 'syslog', 
                    'security', 'audit', 'traffic', 'ids', 'ips'
                ]
                if any(term in sample_text for term in log_terms):
                    return True
                    
        return False
    
    def _is_cmdb_field(self, field: FieldIntelligence) -> bool:
        """Determine if field represents CMDB data"""
        name_lower = field.name.lower()
        table_lower = field.table.lower()
        
        # Direct CMDB indicators
        if table_lower == 'cmdb':
            return True
            
        cmdb_patterns = [
            'cmdb', 'ci_name', 'configuration_item', 'asset', 'inventory'
        ]
        
        return any(pattern in name_lower for pattern in cmdb_patterns)
    
    def _is_security_control_field(self, field: FieldIntelligence) -> bool:
        """Determine if field represents security controls"""
        name_lower = field.name.lower()
        table_lower = field.table.lower()
        
        # Security control patterns
        security_patterns = [
            'edr', 'dlp', 'fim', 'av', 'antivirus', 'security', 'control',
            'agent', 'protection', 'monitoring'
        ]
        
        if any(pattern in name_lower for pattern in security_patterns):
            return True
            
        # CrowdStrike table is security control
        if table_lower == 'crowdstrike':
            return True
            
        # Check sample values for security terms
        if field.sample_values:
            sample_text = ' '.join(str(v).lower() for v in field.sample_values[:20])
            security_terms = [
                'crowdstrike', 'tanium', 'symantec', 'mcafee', 'carbon', 
                'falcon', 'edr', 'endpoint'
            ]
            if any(term in sample_text for term in security_terms):
                return True
                
        return False
    
    def get_ao1_readiness_score(self, classification: AO1FieldClassification) -> Dict[str, float]:
        """Calculate readiness scores for each AO1 metric"""
        
        scores = {}
        
        # Global Visibility Score readiness
        asset_score = len(classification.asset_fields) * 0.4
        logging_score = len(classification.logging_fields) * 0.4  
        time_score = len(classification.time_fields) * 0.2
        scores['global_visibility'] = min(1.0, asset_score + logging_score + time_score)
        
        # Platform Coverage readiness
        platform_score = len(classification.platform_fields) * 0.6
        asset_score = len(classification.asset_fields) * 0.4
        scores['platform_coverage'] = min(1.0, platform_score + asset_score)
        
        # Infrastructure Visibility readiness
        infra_score = len(classification.infrastructure_fields) * 0.7
        asset_score = len(classification.asset_fields) * 0.3
        scores['infrastructure_visibility'] = min(1.0, infra_score + asset_score)
        
        # Log Role Coverage readiness
        role_score = len(classification.role_fields) * 0.6
        log_type_score = len(classification.log_type_fields) * 0.4
        scores['log_role_coverage'] = min(1.0, role_score + log_type_score)
        
        # Silent Assets readiness (same as global visibility)
        scores['silent_assets'] = scores['global_visibility']
        
        # CMDB Completeness readiness
        cmdb_score = len(classification.cmdb_fields) * 1.0
        scores['cmdb_completeness'] = min(1.0, cmdb_score)
        
        # Security Control Visibility readiness
        security_score = len(classification.security_control_fields) * 1.0
        scores['security_control_visibility'] = min(1.0, security_score)
        
        return scores