#!/usr/bin/env python3

from typing import List, Dict
from models import FieldIntelligence

class QueryTemplates:
    
    @staticmethod
    def asset_discovery(fields: List[FieldIntelligence], relationships: Dict) -> str:
        if not fields:
            return "-- No suitable fields for asset discovery"
            
        primary_field = fields[0]
        
        return f"""
-- Asset Discovery and Classification
WITH asset_inventory AS (
    SELECT 
        {primary_field.name} as asset_id,
        COUNT(*) as total_occurrences,
        COUNT(DISTINCT {primary_field.name}) as unique_assets,
        MIN({primary_field.name}) as first_seen,
        MAX({primary_field.name}) as last_seen
    FROM {primary_field.table}
    WHERE {primary_field.name} IS NOT NULL
    GROUP BY {primary_field.name}
),
asset_analysis AS (
    SELECT 
        asset_id,
        total_occurrences,
        unique_assets,
        first_seen,
        last_seen,
        CASE 
            WHEN total_occurrences > (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY total_occurrences) FROM asset_inventory) 
                THEN 'HIGH_ACTIVITY'
            WHEN total_occurrences > (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_occurrences) FROM asset_inventory) 
                THEN 'MEDIUM_ACTIVITY'
            ELSE 'LOW_ACTIVITY'
        END as activity_level,
        ROUND(100.0 * total_occurrences / SUM(total_occurrences) OVER(), 2) as prevalence_pct,
        ROW_NUMBER() OVER (ORDER BY total_occurrences DESC) as asset_rank
    FROM asset_inventory
)
SELECT 
    asset_id,
    activity_level,
    total_occurrences,
    prevalence_pct,
    asset_rank,
    CASE 
        WHEN asset_rank <= 10 THEN 'CRITICAL'
        WHEN asset_rank <= 50 THEN 'HIGH'
        WHEN asset_rank <= 200 THEN 'MEDIUM'
        ELSE 'LOW'
    END as priority_level
FROM asset_analysis
ORDER BY total_occurrences DESC
LIMIT 100;
        """
        
    @staticmethod
    def security_analysis(fields: List[FieldIntelligence], relationships: Dict) -> str:
        if not fields:
            return "-- No suitable fields for security analysis"
            
        primary_field = fields[0]
        
        return f"""
-- Security Event Analysis
WITH security_events AS (
    SELECT 
        {primary_field.name} as event_type,
        COUNT(*) as event_count,
        COUNT(DISTINCT {primary_field.name}) as unique_events,
        COUNT(DISTINCT DATE({primary_field.name})) as active_days
    FROM {primary_field.table}
    WHERE {primary_field.name} IS NOT NULL
    GROUP BY {primary_field.name}
),
threat_analysis AS (
    SELECT 
        event_type,
        event_count,
        unique_events,
        active_days,
        CASE 
            WHEN LOWER(CAST(event_type AS STRING)) LIKE '%critical%' OR 
                 LOWER(CAST(event_type AS STRING)) LIKE '%alert%' THEN 9
            WHEN LOWER(CAST(event_type AS STRING)) LIKE '%warning%' OR
                 LOWER(CAST(event_type AS STRING)) LIKE '%block%' THEN 7
            WHEN LOWER(CAST(event_type AS STRING)) LIKE '%error%' THEN 5
            ELSE 3
        END as severity_score,
        CASE 
            WHEN event_count > (SELECT AVG(event_count) * 3 FROM security_events) THEN 'ANOMALOUS'
            WHEN event_count > (SELECT AVG(event_count) FROM security_events) THEN 'ELEVATED'
            ELSE 'NORMAL'
        END as frequency_pattern
    FROM security_events
)
SELECT 
    event_type,
    severity_score,
    frequency_pattern,
    event_count,
    active_days,
    CASE 
        WHEN severity_score >= 8 AND frequency_pattern = 'ANOMALOUS' THEN 'IMMEDIATE'
        WHEN severity_score >= 7 OR frequency_pattern = 'ANOMALOUS' THEN 'HIGH'
        WHEN severity_score >= 5 OR frequency_pattern = 'ELEVATED' THEN 'MEDIUM'
        ELSE 'LOW'
    END as alert_priority
FROM threat_analysis
WHERE severity_score >= 5 OR frequency_pattern != 'NORMAL'
ORDER BY severity_score DESC, event_count DESC
LIMIT 50;
        """
        
    @staticmethod
    def user_behavior(fields: List[FieldIntelligence], relationships: Dict) -> str:
        if not fields:
            return "-- No suitable fields for user behavior analysis"
            
        primary_field = fields[0]
        
        return f"""
-- User Behavior Analysis
WITH user_activity AS (
    SELECT 
        {primary_field.name} as user_identifier,
        COUNT(*) as total_actions,
        COUNT(DISTINCT DATE({primary_field.name})) as active_days,
        MIN({primary_field.name}) as first_activity,
        MAX({primary_field.name}) as last_activity
    FROM {primary_field.table}
    WHERE {primary_field.name} IS NOT NULL
    GROUP BY {primary_field.name}
),
behavior_patterns AS (
    SELECT 
        user_identifier,
        total_actions,
        active_days,
        ROUND(total_actions::FLOAT / NULLIF(active_days, 0), 2) as avg_daily_actions,
        CASE 
            WHEN total_actions > (SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY total_actions) FROM user_activity) 
                THEN 'HIGH_ACTIVITY'
            WHEN total_actions < (SELECT PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY total_actions) FROM user_activity)
                THEN 'LOW_ACTIVITY'
            ELSE 'NORMAL_ACTIVITY'
        END as activity_pattern
    FROM user_activity
)
SELECT 
    user_identifier,
    activity_pattern,
    total_actions,
    active_days,
    avg_daily_actions,
    CASE 
        WHEN activity_pattern = 'HIGH_ACTIVITY' AND avg_daily_actions > 100 THEN 'MONITOR'
        WHEN activity_pattern = 'LOW_ACTIVITY' AND active_days < 5 THEN 'INACTIVE'
        ELSE 'NORMAL'
    END as risk_indicator
FROM behavior_patterns
ORDER BY total_actions DESC
LIMIT 100;
        """
        
    @staticmethod
    def network_topology(fields: List[FieldIntelligence], relationships: Dict) -> str:
        if not fields:
            return "-- No suitable fields for network topology analysis"
            
        primary_field = fields[0]
        
        return f"""
-- Network Topology Analysis
WITH network_nodes AS (
    SELECT 
        {primary_field.name} as node_id,
        COUNT(*) as connection_count,
        COUNT(DISTINCT {primary_field.name}) as unique_connections
    FROM {primary_field.table}
    WHERE {primary_field.name} IS NOT NULL
    GROUP BY {primary_field.name}
),
topology_analysis AS (
    SELECT 
        node_id,
        connection_count,
        unique_connections,
        CASE 
            WHEN connection_count > (SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY connection_count) FROM network_nodes)
                THEN 'HUB'
            WHEN connection_count > (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY connection_count) FROM network_nodes)
                THEN 'CONNECTOR'
            WHEN connection_count = 1 THEN 'LEAF'
            ELSE 'STANDARD'
        END as node_type,
        ROUND(100.0 * connection_count / SUM(connection_count) OVER(), 3) as traffic_share
    FROM network_nodes
)
SELECT 
    node_id,
    node_type,
    connection_count,
    traffic_share,
    CASE 
        WHEN node_type = 'HUB' THEN 'CRITICAL'
        WHEN node_type = 'CONNECTOR' THEN 'IMPORTANT'
        ELSE 'STANDARD'
    END as infrastructure_importance
FROM topology_analysis
ORDER BY connection_count DESC
LIMIT 100;
        """
        
    @staticmethod
    def compliance_monitoring(fields: List[FieldIntelligence], relationships: Dict) -> str:
        if not fields:
            return "-- No suitable fields for compliance monitoring"
            
        primary_field = fields[0]
        
        return f"""
-- Compliance Monitoring
WITH compliance_events AS (
    SELECT 
        {primary_field.name} as compliance_item,
        COUNT(*) as total_checks,
        COUNT(DISTINCT DATE({primary_field.name})) as monitoring_days
    FROM {primary_field.table}
    WHERE {primary_field.name} IS NOT NULL
    GROUP BY {primary_field.name}
),
compliance_status AS (
    SELECT 
        compliance_item,
        total_checks,
        monitoring_days,
        CASE 
            WHEN LOWER(CAST(compliance_item AS STRING)) LIKE '%pass%' OR
                 LOWER(CAST(compliance_item AS STRING)) LIKE '%compliant%' THEN 'COMPLIANT'
            WHEN LOWER(CAST(compliance_item AS STRING)) LIKE '%fail%' OR
                 LOWER(CAST(compliance_item AS STRING)) LIKE '%violation%' THEN 'NON_COMPLIANT'
            ELSE 'UNKNOWN'
        END as compliance_state,
        ROUND(100.0 * total_checks / SUM(total_checks) OVER(), 2) as check_frequency_pct
    FROM compliance_events
)
SELECT 
    compliance_item,
    compliance_state,
    total_checks,
    monitoring_days,
    check_frequency_pct,
    CASE 
        WHEN compliance_state = 'NON_COMPLIANT' THEN 'ACTION_REQUIRED'
        WHEN compliance_state = 'UNKNOWN' THEN 'INVESTIGATION_NEEDED'
        ELSE 'GOOD'
    END as action_status
FROM compliance_status
ORDER BY 
    CASE compliance_state 
        WHEN 'NON_COMPLIANT' THEN 1 
        WHEN 'UNKNOWN' THEN 2 
        ELSE 3 
    END,
    total_checks DESC
LIMIT 100;
        """
        
    @staticmethod
    def performance_analytics(fields: List[FieldIntelligence], relationships: Dict) -> str:
        if not fields:
            return "-- No suitable fields for performance analytics"
            
        primary_field = fields[0]
        
        return f"""
-- Performance Analytics
WITH performance_metrics AS (
    SELECT 
        {primary_field.name} as metric_source,
        COUNT(*) as measurement_count,
        AVG(CAST({primary_field.name} AS FLOAT)) as avg_value,
        MIN(CAST({primary_field.name} AS FLOAT)) as min_value,
        MAX(CAST({primary_field.name} AS FLOAT)) as max_value,
        STDDEV(CAST({primary_field.name} AS FLOAT)) as std_deviation
    FROM {primary_field.table}
    WHERE {primary_field.name} IS NOT NULL 
      AND {primary_field.name} ~ '^[0-9.]+$'
    GROUP BY {primary_field.name}
),
performance_analysis AS (
    SELECT 
        metric_source,
        measurement_count,
        avg_value,
        min_value,
        max_value,
        std_deviation,
        CASE 
            WHEN avg_value > (SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY avg_value) FROM performance_metrics)
                THEN 'HIGH_PERFORMANCE'
            WHEN avg_value < (SELECT PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY avg_value) FROM performance_metrics)
                THEN 'LOW_PERFORMANCE'
            ELSE 'NORMAL_PERFORMANCE'
        END as performance_category,
        CASE 
            WHEN std_deviation > avg_value * 0.5 THEN 'HIGH_VARIABILITY'
            WHEN std_deviation < avg_value * 0.1 THEN 'LOW_VARIABILITY'
            ELSE 'MODERATE_VARIABILITY'
        END as stability_indicator
    FROM performance_metrics
)
SELECT 
    metric_source,
    performance_category,
    stability_indicator,
    ROUND(avg_value, 2) as average_performance,
    ROUND(std_deviation, 2) as performance_variance,
    measurement_count,
    CASE 
        WHEN performance_category = 'LOW_PERFORMANCE' OR stability_indicator = 'HIGH_VARIABILITY' 
            THEN 'NEEDS_ATTENTION'
        WHEN performance_category = 'HIGH_PERFORMANCE' AND stability_indicator = 'LOW_VARIABILITY' 
            THEN 'OPTIMAL'
        ELSE 'ACCEPTABLE'
    END as optimization_status
FROM performance_analysis
ORDER BY avg_value DESC
LIMIT 100;
        """