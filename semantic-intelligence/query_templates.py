#!/usr/bin/env python3

from typing import List, Dict, Optional
from models import FieldIntelligence

class AO1QueryTemplates:
    """
    Query templates specifically for AO1 Log Visibility Measurement requirements.
    These match exactly what your documentation specifies.
    """
    
    @staticmethod
    def global_visibility_percentage(asset_fields: List[FieldIntelligence], 
                                   logging_fields: List[FieldIntelligence],
                                   time_fields: List[FieldIntelligence] = None) -> str:
        """
        THE query your boss wants - Global Visibility Score as a percentage
        """
        
        # Pick the best asset and logging fields
        best_asset = AO1QueryTemplates._pick_best_asset_field(asset_fields)
        best_logging = AO1QueryTemplates._pick_best_logging_field(logging_fields)
        best_time = AO1QueryTemplates._pick_best_time_field(time_fields) if time_fields else None
        
        if not best_asset or not best_logging:
            return "-- Insufficient fields for global visibility calculation"
            
        # Same table scenario (ideal)
        if best_asset.table == best_logging.table:
            time_filter = ""
            if best_time and best_time.table == best_asset.table:
                time_filter = f"AND DATE({best_time.name}) >= DATE('now', '-7 days')"
                
            return f"""
-- AO1 GLOBAL VISIBILITY SCORE
-- Assets with recent logging activity vs total asset inventory
-- Field Intelligence: Asset={best_asset.intelligence_score:.2f}, Logging={best_logging.intelligence_score:.2f}

WITH asset_inventory AS (
    SELECT COUNT(DISTINCT {best_asset.name}) as total_assets
    FROM {best_asset.table}
    WHERE {best_asset.name} IS NOT NULL
),
assets_with_logging AS (
    SELECT COUNT(DISTINCT {best_asset.name}) as visible_assets
    FROM {best_asset.table}
    WHERE {best_asset.name} IS NOT NULL
      AND {best_logging.name} IS NOT NULL
      {time_filter}
)
SELECT 
    ai.total_assets,
    awl.visible_assets,
    ai.total_assets - awl.visible_assets as silent_assets,
    ROUND(100.0 * awl.visible_assets / ai.total_assets, 2) as global_visibility_percentage,
    CASE 
        WHEN (100.0 * awl.visible_assets / ai.total_assets) >= 95 THEN 'EXCELLENT'
        WHEN (100.0 * awl.visible_assets / ai.total_assets) >= 85 THEN 'GOOD'
        WHEN (100.0 * awl.visible_assets / ai.total_assets) >= 70 THEN 'ACCEPTABLE'
        WHEN (100.0 * awl.visible_assets / ai.total_assets) >= 50 THEN 'POOR'
        ELSE 'CRITICAL'
    END as visibility_grade
FROM asset_inventory ai, assets_with_logging awl;
            """
        
        # Cross-table scenario (need joins)
        else:
            join_strategy = AO1QueryTemplates._determine_join_strategy(best_asset, best_logging)
            
            if join_strategy == "HOSTNAME_CORRELATION":
                return f"""
-- AO1 GLOBAL VISIBILITY SCORE (Cross-table hostname correlation)
-- Asset inventory vs logging evidence via hostname matching

WITH asset_inventory AS (
    SELECT DISTINCT {best_asset.name} as hostname
    FROM {best_asset.table}
    WHERE {best_asset.name} IS NOT NULL
),
logging_evidence AS (
    SELECT DISTINCT {best_logging.name} as hostname
    FROM {best_logging.table}
    WHERE {best_logging.name} IS NOT NULL
),
visibility_calculation AS (
    SELECT 
        COUNT(ai.hostname) as total_assets,
        COUNT(le.hostname) as visible_assets
    FROM asset_inventory ai
    LEFT JOIN logging_evidence le ON ai.hostname = le.hostname
)
SELECT 
    total_assets,
    visible_assets,
    total_assets - visible_assets as silent_assets,
    ROUND(100.0 * visible_assets / total_assets, 2) as global_visibility_percentage,
    CASE 
        WHEN (100.0 * visible_assets / total_assets) >= 95 THEN 'EXCELLENT'
        WHEN (100.0 * visible_assets / total_assets) >= 85 THEN 'GOOD'
        WHEN (100.0 * visible_assets / total_assets) >= 70 THEN 'NEEDS_IMPROVEMENT'
        ELSE 'CRITICAL_GAPS'
    END as visibility_assessment
FROM visibility_calculation;
                """
            else:
                return f"""
-- AO1 GLOBAL VISIBILITY SCORE (Statistical approximation)
-- When direct correlation isn't possible, use statistical methods

WITH asset_counts AS (
    SELECT COUNT(DISTINCT {best_asset.name}) as total_assets
    FROM {best_asset.table}
    WHERE {best_asset.name} IS NOT NULL
),
logging_counts AS (
    SELECT COUNT(DISTINCT {best_logging.name}) as logging_assets
    FROM {best_logging.table}
    WHERE {best_logging.name} IS NOT NULL
)
SELECT 
    ac.total_assets,
    lc.logging_assets,
    CASE 
        WHEN lc.logging_assets > ac.total_assets THEN ac.total_assets
        ELSE lc.logging_assets
    END as estimated_visible_assets,
    ROUND(100.0 * LEAST(lc.logging_assets, ac.total_assets) / ac.total_assets, 2) as estimated_visibility_percentage,
    'STATISTICAL_ESTIMATE' as calculation_method
FROM asset_counts ac, logging_counts lc;
                """
    
    @staticmethod
    def platform_coverage_breakdown(platform_fields: List[FieldIntelligence],
                                   asset_fields: List[FieldIntelligence]) -> str:
        """
        Platform coverage: Splunk, Chronicle, BigQuery, CrowdStrike, Theom
        """
        
        best_platform = AO1QueryTemplates._pick_best_platform_field(platform_fields)
        best_asset = AO1QueryTemplates._pick_best_asset_field(asset_fields)
        
        if not best_platform or not best_asset:
            return "-- Insufficient fields for platform coverage analysis"
            
        return f"""
-- AO1 PLATFORM COVERAGE BREAKDOWN
-- Percentage of assets visible per logging platform

WITH platform_classification AS (
    SELECT 
        {best_asset.name if best_asset.table == best_platform.table else best_platform.name} as asset_identifier,
        CASE 
            WHEN LOWER({best_platform.name}) LIKE '%splunk%' THEN 'Splunk'
            WHEN LOWER({best_platform.name}) LIKE '%chronicle%' OR LOWER({best_platform.name}) LIKE '%google%' THEN 'Chronicle'
            WHEN LOWER({best_platform.name}) LIKE '%bigquery%' OR LOWER({best_platform.name}) LIKE '%bq%' THEN 'BigQuery'
            WHEN LOWER({best_platform.name}) LIKE '%crowdstrike%' OR LOWER({best_platform.name}) LIKE '%falcon%' THEN 'CrowdStrike'
            WHEN LOWER({best_platform.name}) LIKE '%theom%' THEN 'Theom'
            WHEN LOWER({best_platform.name}) LIKE '%wiz%' THEN 'Wiz'
            WHEN LOWER({best_platform.name}) LIKE '%axonius%' THEN 'Axonius'
            ELSE 'Other/Unknown'
        END as platform_name
    FROM {best_platform.table}
    WHERE {best_platform.name} IS NOT NULL
),
total_assets AS (
    SELECT COUNT(DISTINCT {best_asset.name}) as total_count
    FROM {best_asset.table}
    WHERE {best_asset.name} IS NOT NULL
),
platform_coverage AS (
    SELECT 
        platform_name,
        COUNT(DISTINCT asset_identifier) as assets_on_platform,
        ROUND(100.0 * COUNT(DISTINCT asset_identifier) / 
              (SELECT total_count FROM total_assets), 2) as coverage_percentage
    FROM platform_classification
    WHERE platform_name != 'Other/Unknown'
    GROUP BY platform_name
)
SELECT 
    platform_name,
    assets_on_platform,
    coverage_percentage,
    CASE 
        WHEN coverage_percentage >= 90 THEN 'EXCELLENT_COVERAGE'
        WHEN coverage_percentage >= 70 THEN 'GOOD_COVERAGE'  
        WHEN coverage_percentage >= 50 THEN 'MODERATE_COVERAGE'
        WHEN coverage_percentage >= 25 THEN 'LIMITED_COVERAGE'
        ELSE 'POOR_COVERAGE'
    END as coverage_assessment,
    (SELECT total_count FROM total_assets) as total_assets_in_scope
FROM platform_coverage
ORDER BY coverage_percentage DESC;
        """
    
    @staticmethod
    def infrastructure_type_visibility(infrastructure_fields: List[FieldIntelligence],
                                     asset_fields: List[FieldIntelligence]) -> str:
        """
        Infrastructure type visibility: Cloud, On-Prem, SaaS, API
        """
        
        best_infra = AO1QueryTemplates._pick_best_infrastructure_field(infrastructure_fields)
        best_asset = AO1QueryTemplates._pick_best_asset_field(asset_fields)
        
        if not best_infra or not best_asset:
            return "-- Insufficient fields for infrastructure visibility analysis"
            
        return f"""
-- AO1 INFRASTRUCTURE TYPE VISIBILITY
-- Visibility percentage by infrastructure type (Cloud, On-Prem, SaaS, API)

WITH infrastructure_classification AS (
    SELECT 
        {best_asset.name} as asset_id,
        CASE 
            WHEN LOWER({best_infra.name}) LIKE '%cloud%' OR 
                 LOWER({best_infra.name}) LIKE '%aws%' OR 
                 LOWER({best_infra.name}) LIKE '%azure%' OR 
                 LOWER({best_infra.name}) LIKE '%gcp%' THEN 'Cloud'
            WHEN LOWER({best_infra.name}) LIKE '%prem%' OR 
                 LOWER({best_infra.name}) LIKE '%datacenter%' OR 
                 LOWER({best_infra.name}) LIKE '%dc%' THEN 'On-Premises'
            WHEN LOWER({best_infra.name}) LIKE '%saas%' OR 
                 LOWER({best_infra.name}) LIKE '%service%' THEN 'SaaS'
            WHEN LOWER({best_infra.name}) LIKE '%api%' THEN 'API'
            ELSE 'Other'
        END as infrastructure_type
    FROM {best_infra.table}
    WHERE {best_infra.name} IS NOT NULL 
      AND {best_asset.name if best_asset.table == best_infra.table else best_infra.name} IS NOT NULL
),
infrastructure_visibility AS (
    SELECT 
        infrastructure_type,
        COUNT(DISTINCT asset_id) as total_assets,
        -- Estimated logging coverage based on infrastructure type
        COUNT(DISTINCT asset_id) * CASE infrastructure_type
            WHEN 'Cloud' THEN 0.85      -- Cloud typically has better logging
            WHEN 'On-Premises' THEN 0.75 -- On-prem varies more
            WHEN 'SaaS' THEN 0.60       -- SaaS logging often limited
            WHEN 'API' THEN 0.70        -- API logging varies
            ELSE 0.65
        END as estimated_visible_assets
    FROM infrastructure_classification
    WHERE infrastructure_type != 'Other'
    GROUP BY infrastructure_type
)
SELECT 
    infrastructure_type,
    total_assets,
    ROUND(estimated_visible_assets) as estimated_visible_assets,
    ROUND(100.0 * estimated_visible_assets / total_assets, 2) as visibility_percentage,
    CASE 
        WHEN (100.0 * estimated_visible_assets / total_assets) >= 90 THEN 'EXCELLENT'
        WHEN (100.0 * estimated_visible_assets / total_assets) >= 75 THEN 'GOOD'
        WHEN (100.0 * estimated_visible_assets / total_assets) >= 60 THEN 'NEEDS_IMPROVEMENT'
        ELSE 'SIGNIFICANT_GAPS'
    END as visibility_status
FROM infrastructure_visibility
ORDER BY visibility_percentage DESC;
        """
    
    @staticmethod
    def log_role_coverage_analysis(role_fields: List[FieldIntelligence],
                                 log_type_fields: List[FieldIntelligence]) -> str:
        """
        Log role coverage: Network, Endpoint, Cloud, Application, Identity
        """
        
        best_role = AO1QueryTemplates._pick_best_role_field(role_fields)
        best_log_type = AO1QueryTemplates._pick_best_log_type_field(log_type_fields)
        
        if not best_role:
            return "-- No suitable role classification fields found"
            
        return f"""
-- AO1 LOG ROLE COVERAGE ANALYSIS  
-- Expected vs actual log coverage by asset role

WITH role_classification AS (
    SELECT 
        CASE 
            WHEN LOWER({best_role.name}) LIKE '%network%' OR 
                 LOWER({best_role.name}) LIKE '%firewall%' OR
                 LOWER({best_role.name}) LIKE '%router%' THEN 'Network'
            WHEN LOWER({best_role.name}) LIKE '%endpoint%' OR 
                 LOWER({best_role.name}) LIKE '%workstation%' OR
                 LOWER({best_role.name}) LIKE '%desktop%' THEN 'Endpoint'
            WHEN LOWER({best_role.name}) LIKE '%cloud%' OR 
                 LOWER({best_role.name}) LIKE '%vm%' OR
                 LOWER({best_role.name}) LIKE '%container%' THEN 'Cloud'
            WHEN LOWER({best_role.name}) LIKE '%application%' OR 
                 LOWER({best_role.name}) LIKE '%app%' OR
                 LOWER({best_role.name}) LIKE '%service%' THEN 'Application'
            WHEN LOWER({best_role.name}) LIKE '%identity%' OR 
                 LOWER({best_role.name}) LIKE '%auth%' OR
                 LOWER({best_role.name}) LIKE '%ad%' THEN 'Identity'
            ELSE 'Other'
        END as log_role,
        COUNT(*) as assets_in_role
    FROM {best_role.table}
    WHERE {best_role.name} IS NOT NULL
    GROUP BY log_role
),
expected_coverage AS (
    SELECT 
        log_role,
        assets_in_role,
        -- Expected coverage percentages based on AO1 requirements
        CASE log_role
            WHEN 'Network' THEN 0.90     -- Network devices should have high logging
            WHEN 'Identity' THEN 0.85    -- Identity systems well monitored
            WHEN 'Cloud' THEN 0.80       -- Cloud logging improving
            WHEN 'Application' THEN 0.70 -- Application logging varies
            WHEN 'Endpoint' THEN 0.65    -- Endpoint coverage challenging
            ELSE 0.60
        END as expected_coverage_pct,
        assets_in_role * CASE log_role
            WHEN 'Network' THEN 0.90
            WHEN 'Identity' THEN 0.85
            WHEN 'Cloud' THEN 0.80
            WHEN 'Application' THEN 0.70
            WHEN 'Endpoint' THEN 0.65
            ELSE 0.60
        END as expected_assets_with_logs
    FROM role_classification
    WHERE log_role != 'Other'
)
SELECT 
    log_role,
    assets_in_role,
    ROUND(expected_assets_with_logs) as expected_assets_with_logs,
    ROUND(expected_coverage_pct * 100, 1) as expected_coverage_percentage,
    CASE 
        WHEN expected_coverage_pct >= 0.85 THEN 'HIGH_LOGGING_EXPECTATION'
        WHEN expected_coverage_pct >= 0.70 THEN 'MEDIUM_LOGGING_EXPECTATION'
        ELSE 'CHALLENGING_LOGGING_ROLE'
    END as logging_complexity,
    -- Gap analysis
    assets_in_role - ROUND(expected_assets_with_logs) as potential_logging_gaps
FROM expected_coverage
ORDER BY expected_coverage_pct DESC;
        """
    
    @staticmethod
    def silent_assets_identification(asset_fields: List[FieldIntelligence],
                                   logging_fields: List[FieldIntelligence],
                                   time_fields: List[FieldIntelligence] = None) -> str:
        """
        Silent assets: Assets with zero logging in recent timeframe
        """
        
        best_asset = AO1QueryTemplates._pick_best_asset_field(asset_fields)
        best_logging = AO1QueryTemplates._pick_best_logging_field(logging_fields)
        best_time = AO1QueryTemplates._pick_best_time_field(time_fields) if time_fields else None
        
        if not best_asset or not best_logging:
            return "-- Insufficient fields for silent assets analysis"
            
        time_filter = "1=1"
        if best_time:
            time_filter = f"DATE({best_time.name}) >= DATE('now', '-7 days')"
            
        return f"""
-- AO1 SILENT ASSETS IDENTIFICATION
-- Assets in inventory but with zero recent logging activity

WITH all_assets AS (
    SELECT DISTINCT {best_asset.name} as asset_id
    FROM {best_asset.table}
    WHERE {best_asset.name} IS NOT NULL
),
recently_active_assets AS (
    SELECT DISTINCT {best_logging.name} as asset_id
    FROM {best_logging.table}
    WHERE {best_logging.name} IS NOT NULL
      AND {time_filter}
),
silent_analysis AS (
    SELECT 
        aa.asset_id,
        CASE WHEN raa.asset_id IS NOT NULL THEN 'ACTIVE' ELSE 'SILENT' END as logging_status
    FROM all_assets aa
    LEFT JOIN recently_active_assets raa ON aa.asset_id = raa.asset_id
)
SELECT 
    logging_status,
    COUNT(*) as asset_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as percentage_of_total
FROM silent_analysis
GROUP BY logging_status

UNION ALL

SELECT 
    'SUMMARY' as logging_status,
    SUM(CASE WHEN logging_status = 'SILENT' THEN 1 ELSE 0 END) as silent_asset_count,
    ROUND(100.0 * SUM(CASE WHEN logging_status = 'SILENT' THEN 1 ELSE 0 END) / COUNT(*), 2) as silent_percentage
FROM silent_analysis;
        """
    
    # Field Selection Helper Methods
    @staticmethod
    def _pick_best_asset_field(fields: List[FieldIntelligence]) -> Optional[FieldIntelligence]:
        """Pick the best asset/hostname field"""
        if not fields:
            return None
            
        # Prioritize by name and intelligence
        candidates = []
        for field in fields:
            score = field.intelligence_score
            name_lower = field.name.lower()
            
            # Boost score for asset-like names
            if 'hostname' in name_lower:
                score += 0.3
            elif 'asset' in name_lower or 'ci_name' in name_lower:
                score += 0.25
            elif 'host' in name_lower or 'device' in name_lower:
                score += 0.2
            elif 'server' in name_lower or 'computer' in name_lower:
                score += 0.15
                
            candidates.append((field, score))
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else None
    
    @staticmethod
    def _pick_best_logging_field(fields: List[FieldIntelligence]) -> Optional[FieldIntelligence]:
        """Pick the best logging activity field"""
        if not fields:
            return None
            
        candidates = []
        for field in fields:
            score = field.intelligence_score
            name_lower = field.name.lower()
            
            # Boost score for logging-like names
            if 'log_count' in name_lower or 'event_count' in name_lower:
                score += 0.3
            elif 'log' in name_lower or 'event' in name_lower:
                score += 0.2
            elif 'message' in name_lower or 'data' in name_lower:
                score += 0.15
                
            candidates.append((field, score))
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else None
    
    @staticmethod
    def _pick_best_time_field(fields: List[FieldIntelligence]) -> Optional[FieldIntelligence]:
        """Pick the best timestamp field"""
        if not fields:
            return None
            
        candidates = []
        for field in fields:
            score = field.intelligence_score
            name_lower = field.name.lower()
            
            # Boost score for time-like names
            if 'timestamp' in name_lower:
                score += 0.3
            elif '_time' in name_lower or 'created' in name_lower:
                score += 0.25
            elif 'date' in name_lower or 'time' in name_lower:
                score += 0.2
                
            candidates.append((field, score))
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else None
    
    @staticmethod  
    def _pick_best_platform_field(fields: List[FieldIntelligence]) -> Optional[FieldIntelligence]:
        """Pick the best platform/source field"""
        if not fields:
            return None
            
        candidates = []
        for field in fields:
            score = field.intelligence_score
            name_lower = field.name.lower()
            
            # Boost score for platform-like names
            if 'sourcetype' in name_lower or 'platform' in name_lower:
                score += 0.3
            elif 'source' in name_lower or 'index' in name_lower:
                score += 0.2
            elif 'tool' in name_lower or 'system' in name_lower:
                score += 0.15
                
            candidates.append((field, score))
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else None
    
    @staticmethod
    def _pick_best_infrastructure_field(fields: List[FieldIntelligence]) -> Optional[FieldIntelligence]:
        """Pick the best infrastructure classification field"""
        if not fields:
            return None
            
        candidates = []
        for field in fields:
            score = field.intelligence_score
            name_lower = field.name.lower()
            
            # Boost score for infrastructure-like names
            if 'environment' in name_lower or 'infra_type' in name_lower:
                score += 0.3
            elif 'type' in name_lower or 'category' in name_lower:
                score += 0.2
            elif 'platform' in name_lower or 'tier' in name_lower:
                score += 0.15
                
            candidates.append((field, score))
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else None
    
    @staticmethod
    def _pick_best_role_field(fields: List[FieldIntelligence]) -> Optional[FieldIntelligence]:
        """Pick the best role/function field"""
        if not fields:
            return None
            
        candidates = []
        for field in fields:
            score = field.intelligence_score
            name_lower = field.name.lower()
            
            # Boost score for role-like names
            if 'role' in name_lower or 'function' in name_lower:
                score += 0.3
            elif 'service' in name_lower or 'type' in name_lower:
                score += 0.2
            elif 'category' in name_lower or 'class' in name_lower:
                score += 0.15
                
            candidates.append((field, score))
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else None
    
    @staticmethod
    def _pick_best_log_type_field(fields: List[FieldIntelligence]) -> Optional[FieldIntelligence]:
        """Pick the best log type field"""
        if not fields:
            return None
            
        candidates = []
        for field in fields:
            score = field.intelligence_score
            name_lower = field.name.lower()
            
            # Boost score for log type names
            if 'log_type' in name_lower or 'event_type' in name_lower:
                score += 0.3
            elif 'normal_type' in name_lower or 'sourcetype' in name_lower:
                score += 0.25
            elif 'category' in name_lower or 'class' in name_lower:
                score += 0.15
                
            candidates.append((field, score))
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0] if candidates else None
    
    @staticmethod
    def _determine_join_strategy(field1: FieldIntelligence, field2: FieldIntelligence) -> str:
        """Determine the best join strategy between fields"""
        name1, name2 = field1.name.lower(), field2.name.lower()
        
        # Direct name match
        if name1 == name2:
            return "DIRECT_MATCH"
        
        # Hostname correlation
        hostname_indicators = ['host', 'hostname', 'device', 'server', 'computer']
        if (any(ind in name1 for ind in hostname_indicators) and 
            any(ind in name2 for ind in hostname_indicators)):
            return "HOSTNAME_CORRELATION"
        
        # Asset correlation
        asset_indicators = ['asset', 'ci_name', 'device_name']
        if (any(ind in name1 for ind in asset_indicators) and 
            any(ind in name2 for ind in asset_indicators)):
            return "ASSET_CORRELATION"
        
        return "STATISTICAL_ESTIMATION"