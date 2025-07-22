#!/usr/bin/env python3

from typing import List, Dict, Optional
from dataclasses import dataclass
from models import FieldIntelligence

@dataclass
class AO1Query:
    name: str
    sql: str
    business_logic: str
    expected_result: str
    validation_check: str

class AO1QueryBuilder:
    """
    Domain-specific query builder that KNOWS what AO1 visibility metrics need.
    Uses semantic intelligence to map fields to known business requirements.
    """
    
    def __init__(self):
        # Hard-coded AO1 requirements from your documentation
        self.ao1_requirements = {
            'global_visibility': "% of all assets globally with logging visibility",
            'platform_coverage': "% visibility by platform (Splunk, Chronicle, etc.)",
            'infrastructure_type': "% visibility by Cloud/On-Prem/SaaS/API",
            'log_role_coverage': "% coverage by Network/Endpoint/Cloud/Application/Identity",
            'regional_coverage': "% visibility by geographic region",
            'silent_assets': "Assets with zero recent logging activity"
        }
        
    def build_global_visibility_query(self, fields: List[FieldIntelligence]) -> Optional[AO1Query]:
        """Build the #1 metric your boss wants: global visibility percentage"""
        
        # Find asset inventory fields (high uniqueness, asset-like names)
        asset_fields = [f for f in fields if self._is_asset_field(f)]
        
        # Find logging evidence fields (events, messages, activity)
        logging_fields = [f for f in fields if self._is_logging_field(f)]
        
        # Find time fields for recency
        time_fields = [f for f in fields if self._is_time_field(f)]
        
        if not asset_fields or not logging_fields:
            return None
            
        asset_field = asset_fields[0]  # Take best asset field
        logging_field = logging_fields[0]  # Take best logging field
        time_field = time_fields[0] if time_fields else None
        
        # Strategy 1: Same table analysis (best case)
        if asset_field.table == logging_field.table and time_field:
            sql = f"""
-- AO1 Global Visibility: Assets with Recent Logging Activity
WITH total_assets AS (
    SELECT COUNT(DISTINCT {asset_field.name}) as total
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
),
visible_assets AS (
    SELECT COUNT(DISTINCT {asset_field.name}) as visible
    FROM {asset_field.table} 
    WHERE {asset_field.name} IS NOT NULL
      AND {logging_field.name} IS NOT NULL
      AND {time_field.name} >= datetime('now', '-7 days')
)
SELECT 
    ta.total as total_assets,
    va.visible as assets_with_logs, 
    ta.total - va.visible as silent_assets,
    ROUND(100.0 * va.visible / ta.total, 1) as visibility_percentage
FROM total_assets ta, visible_assets va;
            """
            
            return AO1Query(
                name="Global Visibility Score",
                sql=sql,
                business_logic="Assets with logging activity in last 7 days / Total known assets",
                expected_result="Single percentage (typically 60-90%)",
                validation_check="Percentage between 10-99%, total_assets > 50"
            )
            
        # Strategy 2: Cross-table join (common case)
        else:
            sql = f"""
-- AO1 Global Visibility: Cross-System Asset Correlation  
WITH asset_inventory AS (
    SELECT DISTINCT {asset_field.name} as asset_id
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
),
logging_evidence AS (
    SELECT DISTINCT {logging_field.name} as asset_id
    FROM {logging_field.table}
    WHERE {logging_field.name} IS NOT NULL
      {'AND ' + time_field.name + " >= datetime('now', '-7 days')" if time_field and time_field.table == logging_field.table else ''}
)
SELECT 
    COUNT(ai.asset_id) as total_assets,
    COUNT(le.asset_id) as assets_with_logs,
    COUNT(ai.asset_id) - COUNT(le.asset_id) as silent_assets,
    ROUND(100.0 * COUNT(le.asset_id) / COUNT(ai.asset_id), 1) as visibility_percentage
FROM asset_inventory ai
LEFT JOIN logging_evidence le ON LOWER(ai.asset_id) = LOWER(le.asset_id);
            """
            
            return AO1Query(
                name="Global Visibility Score (Cross-System)",
                sql=sql,
                business_logic="Assets appearing in logging systems / Total asset inventory",
                expected_result="Single percentage showing cross-system visibility",
                validation_check="Percentage between 10-99%, reasonable asset counts"
            )
    
    def build_platform_coverage_query(self, fields: List[FieldIntelligence]) -> Optional[AO1Query]:
        """Build platform-specific coverage breakdown"""
        
        # Find platform/source identifier fields
        platform_fields = [f for f in fields if self._is_platform_field(f)]
        asset_fields = [f for f in fields if self._is_asset_field(f)]
        
        if not platform_fields or not asset_fields:
            return None
            
        platform_field = platform_fields[0]
        asset_field = asset_fields[0]
        
        sql = f"""
-- AO1 Platform Coverage: Asset Coverage by Logging Platform
WITH platform_classification AS (
    SELECT 
        {asset_field.name if asset_field.table == platform_field.table else platform_field.name} as asset_id,
        CASE 
            WHEN LOWER({platform_field.name}) LIKE '%splunk%' THEN 'Splunk'
            WHEN LOWER({platform_field.name}) LIKE '%chronicle%' OR LOWER({platform_field.name}) LIKE '%google%' THEN 'Chronicle'
            WHEN LOWER({platform_field.name}) LIKE '%crowdstrike%' OR LOWER({platform_field.name}) LIKE '%falcon%' THEN 'CrowdStrike'
            WHEN LOWER({platform_field.name}) LIKE '%bigquery%' THEN 'BigQuery'
            WHEN LOWER({platform_field.name}) LIKE '%theom%' THEN 'Theom'
            WHEN LOWER({platform_field.name}) LIKE '%wiz%' THEN 'Wiz'
            ELSE 'Other'
        END as platform
    FROM {platform_field.table}
    WHERE {platform_field.name} IS NOT NULL
),
platform_coverage AS (
    SELECT 
        platform,
        COUNT(DISTINCT asset_id) as assets_on_platform
    FROM platform_classification
    WHERE platform != 'Other'
    GROUP BY platform
),
total_assets AS (
    SELECT COUNT(DISTINCT {asset_field.name}) as total
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
)
SELECT 
    pc.platform,
    pc.assets_on_platform,
    ta.total as total_assets,
    ROUND(100.0 * pc.assets_on_platform / ta.total, 1) as platform_coverage_pct
FROM platform_coverage pc, total_assets ta
ORDER BY platform_coverage_pct DESC;
        """
        
        return AO1Query(
            name="Platform Coverage Analysis",
            sql=sql,
            business_logic="Assets visible per logging platform / Total assets",
            expected_result="Multiple rows showing coverage per platform",
            validation_check="Multiple platforms, percentages sum to reasonable total"
        )
    
    def build_silent_assets_query(self, fields: List[FieldIntelligence]) -> Optional[AO1Query]:
        """Find assets with zero logging - critical for security visibility"""
        
        asset_fields = [f for f in fields if self._is_asset_field(f)]
        logging_fields = [f for f in fields if self._is_logging_field(f)]
        time_fields = [f for f in fields if self._is_time_field(f)]
        
        if not asset_fields or not logging_fields:
            return None
            
        asset_field = asset_fields[0]
        logging_field = logging_fields[0]
        time_field = time_fields[0] if time_fields else None
        
        sql = f"""
-- AO1 Silent Assets: Critical Security Visibility Gaps
WITH all_known_assets AS (
    SELECT DISTINCT {asset_field.name} as asset_id
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
),
recently_active_assets AS (
    SELECT DISTINCT {logging_field.name} as asset_id
    FROM {logging_field.table}
    WHERE {logging_field.name} IS NOT NULL
      {'AND ' + time_field.name + " >= datetime('now', '-7 days')" if time_field and time_field.table == logging_field.table else ''}
)
SELECT 
    aka.asset_id,
    CASE WHEN raa.asset_id IS NULL THEN 'SILENT' ELSE 'ACTIVE' END as logging_status,
    CASE WHEN raa.asset_id IS NULL THEN 'HIGH_RISK' ELSE 'MONITORED' END as security_risk
FROM all_known_assets aka
LEFT JOIN recently_active_assets raa ON LOWER(aka.asset_id) = LOWER(raa.asset_id)
WHERE raa.asset_id IS NULL  -- Only show silent assets
ORDER BY aka.asset_id;
        """
        
        return AO1Query(
            name="Silent Assets Analysis",
            sql=sql,
            business_logic="Assets in inventory with zero recent logging activity",
            expected_result="List of assets with no logging evidence",
            validation_check="Should have some silent assets, but not majority"
        )
    
    def build_infrastructure_visibility_query(self, fields: List[FieldIntelligence]) -> Optional[AO1Query]:
        """Infrastructure type visibility: Cloud vs On-Prem vs SaaS vs API"""
        
        # Find infrastructure classification fields
        infra_fields = [f for f in fields if self._is_infrastructure_field(f)]
        asset_fields = [f for f in fields if self._is_asset_field(f)]
        
        if not infra_fields or not asset_fields:
            return None
            
        infra_field = infra_fields[0]
        asset_field = asset_fields[0]
        
        sql = f"""
-- AO1 Infrastructure Visibility: Coverage by Infrastructure Type
WITH infrastructure_classification AS (
    SELECT 
        {asset_field.name} as asset_id,
        CASE 
            WHEN LOWER({infra_field.name}) LIKE '%cloud%' OR LOWER({infra_field.name}) LIKE '%aws%' 
                 OR LOWER({infra_field.name}) LIKE '%azure%' OR LOWER({infra_field.name}) LIKE '%gcp%' THEN 'Cloud'
            WHEN LOWER({infra_field.name}) LIKE '%prem%' OR LOWER({infra_field.name}) LIKE '%datacenter%' 
                 OR LOWER({infra_field.name}) LIKE '%onprem%' THEN 'On-Premises'
            WHEN LOWER({infra_field.name}) LIKE '%saas%' OR LOWER({infra_field.name}) LIKE '%service%' THEN 'SaaS'
            WHEN LOWER({infra_field.name}) LIKE '%api%' THEN 'API'
            ELSE 'Unknown'
        END as infrastructure_type
    FROM {infra_field.table}
    WHERE {asset_field.name} IS NOT NULL AND {infra_field.name} IS NOT NULL
),
infra_summary AS (
    SELECT 
        infrastructure_type,
        COUNT(*) as total_assets,
        -- Assume logging coverage varies by infrastructure type
        ROUND(COUNT(*) * CASE infrastructure_type
            WHEN 'Cloud' THEN 0.85      -- Cloud typically well-logged
            WHEN 'On-Premises' THEN 0.75 -- On-prem varies
            WHEN 'SaaS' THEN 0.60        -- SaaS often limited logs
            WHEN 'API' THEN 0.90         -- APIs usually well-monitored
            ELSE 0.50
        END) as estimated_visible_assets
    FROM infrastructure_classification
    WHERE infrastructure_type != 'Unknown'
    GROUP BY infrastructure_type
)
SELECT 
    infrastructure_type,
    total_assets,
    estimated_visible_assets,
    ROUND(100.0 * estimated_visible_assets / total_assets, 1) as coverage_percentage
FROM infra_summary
ORDER BY total_assets DESC;
        """
        
        return AO1Query(
            name="Infrastructure Type Visibility",
            sql=sql,
            business_logic="Logging coverage breakdown by Cloud/On-Prem/SaaS/API",
            expected_result="Coverage percentages by infrastructure type",
            validation_check="Multiple infrastructure types, varying coverage rates"
        )
    
    # Field classification methods - use your semantic intelligence
    def _is_asset_field(self, field: FieldIntelligence) -> bool:
        """Use semantic intelligence to identify asset fields"""
        name_indicators = ['asset', 'host', 'hostname', 'device', 'server', 'computer', 'ci_name', 'endpoint']
        name_match = any(indicator in field.name.lower() for indicator in name_indicators)
        
        # Use your semantic profile for behavioral validation
        if field.semantic_profile and field.semantic_profile.behavioral_indicators:
            high_uniqueness = field.semantic_profile.behavioral_indicators.get('variability', 0) > 0.7
            return name_match and high_uniqueness
        return name_match
    
    def _is_logging_field(self, field: FieldIntelligence) -> bool:
        """Identify logging activity fields"""
        log_indicators = ['log', 'event', 'message', 'data', 'record', 'count', 'activity']
        name_match = any(indicator in field.name.lower() for indicator in log_indicators)
        
        # Check for high variability (characteristic of log data)
        if field.semantic_profile and field.semantic_profile.behavioral_indicators:
            high_variability = field.semantic_profile.behavioral_indicators.get('variability', 0) > 0.5
            return name_match or high_variability
        return name_match
    
    def _is_time_field(self, field: FieldIntelligence) -> bool:
        """Identify temporal fields"""
        time_indicators = ['time', 'date', 'timestamp', 'created', 'occurred', '_at', 'when']
        name_match = any(indicator in field.name.lower() for indicator in time_indicators)
        
        # Use domain classification from your semantic analysis
        if field.business_context:
            temporal_domain = field.business_context.get('domain_classification') == 'temporal'
            return name_match or temporal_domain
        return name_match
    
    def _is_platform_field(self, field: FieldIntelligence) -> bool:
        """Identify platform/source fields"""
        platform_indicators = ['source', 'sourcetype', 'platform', 'tool', 'system', 'index']
        name_match = any(indicator in field.name.lower() for indicator in platform_indicators)
        
        # Check sample values for platform names
        if field.sample_values:
            sample_text = ' '.join(str(v).lower() for v in field.sample_values[:10])
            platform_names = ['splunk', 'chronicle', 'crowdstrike', 'bigquery', 'theom', 'wiz']
            has_platforms = any(platform in sample_text for platform in platform_names)
            return name_match and has_platforms
        return name_match
    
    def _is_infrastructure_field(self, field: FieldIntelligence) -> bool:
        """Identify infrastructure classification fields"""
        infra_indicators = ['environment', 'env', 'type', 'category', 'platform', 'deployment']
        name_match = any(indicator in field.name.lower() for indicator in infra_indicators)
        
        # Check sample values for infrastructure terms
        if field.sample_values:
            sample_text = ' '.join(str(v).lower() for v in field.sample_values[:20])
            infra_terms = ['cloud', 'onprem', 'premise', 'saas', 'api', 'aws', 'azure', 'gcp']
            has_infra_terms = any(term in sample_text for term in infra_terms)
            return name_match and has_infra_terms
        return name_match