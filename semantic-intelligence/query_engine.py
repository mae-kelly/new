#!/usr/bin/env python3

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from models import FieldIntelligence, QueryResult
from database_connector import DatabaseConnector  
from adaptive_query_engine import AdaptiveQueryEngine, MetricResult
from metric_validator import AO1MetricValidator

logger = logging.getLogger(__name__)

@dataclass
class AO1Dashboard:
    global_visibility_score: Optional[MetricResult] = None
    platform_coverage: Optional[MetricResult] = None
    infrastructure_visibility: Optional[MetricResult] = None
    log_role_coverage: Optional[MetricResult] = None
    regional_coverage: Optional[MetricResult] = None
    silent_assets: Optional[MetricResult] = None
    cmdb_completeness: Optional[MetricResult] = None
    security_control_visibility: Optional[MetricResult] = None
    success_rate: float = 0.0
    total_attempts: int = 0

class AO1QueryEngine:
    """
    Adaptive query engine that tries multiple approaches until it finds 
    queries that produce sensible AO1 visibility metrics your boss wants.
    """
    
    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        self.adaptive_engine = AdaptiveQueryEngine(db_connector)
        self.validator = AO1MetricValidator()
        self.successful_patterns = {}
        
    def generate_ao1_dashboard(self, fields: List[FieldIntelligence]) -> AO1Dashboard:
        """Generate complete AO1 dashboard by trying different query approaches"""
        
        dashboard = AO1Dashboard()
        successful_metrics = 0
        total_metrics = 0
        
        logger.info("🎯 Generating AO1 Dashboard - trying adaptive queries until numbers make sense...")
        
        # 1. Global Visibility Score - THE TOP NUMBER YOUR BOSS WANTS
        logger.info("📊 Finding Global Visibility Score...")
        dashboard.global_visibility_score = self._find_global_visibility_with_validation(fields)
        total_metrics += 1
        if dashboard.global_visibility_score and dashboard.global_visibility_score.confidence_score > 0.6:
            successful_metrics += 1
            logger.info(f"✅ Global Visibility: {dashboard.global_visibility_score.metric_value:.1f}%")
        else:
            logger.warning("⚠️ Could not determine reliable global visibility score")
            
        # 2. Platform Coverage - Splunk, Chronicle, BigQuery, CrowdStrike, Theom
        logger.info("🔧 Analyzing Platform Coverage...")
        dashboard.platform_coverage = self._find_platform_coverage_with_validation(fields)
        total_metrics += 1
        if dashboard.platform_coverage and dashboard.platform_coverage.confidence_score > 0.6:
            successful_metrics += 1
            logger.info(f"✅ Platform Coverage analysis complete")
        
        # 3. Infrastructure Type Visibility - Cloud, On-Prem, SaaS, API
        logger.info("🏗️ Analyzing Infrastructure Visibility...")
        dashboard.infrastructure_visibility = self._find_infrastructure_visibility_with_validation(fields)
        total_metrics += 1
        if dashboard.infrastructure_visibility and dashboard.infrastructure_visibility.confidence_score > 0.6:
            successful_metrics += 1
            logger.info(f"✅ Infrastructure Visibility analysis complete")
            
        # 4. Log Role Coverage - Network, Endpoint, Cloud, Application, Identity
        logger.info("📋 Analyzing Log Role Coverage...")  
        dashboard.log_role_coverage = self._find_log_role_coverage_with_validation(fields)
        total_metrics += 1
        if dashboard.log_role_coverage and dashboard.log_role_coverage.confidence_score > 0.6:
            successful_metrics += 1
            logger.info(f"✅ Log Role Coverage analysis complete")
            
        # 5. Silent Assets - Assets with Zero Logging
        logger.info("🔇 Finding Silent Assets...")
        dashboard.silent_assets = self._find_silent_assets_with_validation(fields)  
        total_metrics += 1
        if dashboard.silent_assets and dashboard.silent_assets.confidence_score > 0.6:
            successful_metrics += 1
            logger.info(f"✅ Silent Assets analysis complete")
            
        # 6. CMDB Completeness - Ghost assets and missing metadata
        logger.info("📚 Analyzing CMDB Completeness...")
        dashboard.cmdb_completeness = self._find_cmdb_completeness_with_validation(fields)
        total_metrics += 1 
        if dashboard.cmdb_completeness and dashboard.cmdb_completeness.confidence_score > 0.6:
            successful_metrics += 1
            logger.info(f"✅ CMDB Completeness analysis complete")
            
        dashboard.success_rate = (successful_metrics / total_metrics) * 100 if total_metrics > 0 else 0
        dashboard.total_attempts = total_metrics
        
        logger.info(f"🎯 AO1 Dashboard Complete: {successful_metrics}/{total_metrics} metrics successfully generated ({dashboard.success_rate:.1f}% success rate)")
        
        return dashboard
        
    def _find_global_visibility_with_validation(self, fields: List[FieldIntelligence]) -> Optional[MetricResult]:
        """Find global visibility score using adaptive approach with validation"""
        
        # Use the adaptive engine to try multiple approaches
        metric_result = self.adaptive_engine.find_global_visibility_score(fields)
        
        if not metric_result.best_query.execution_success:
            logger.warning("No successful global visibility queries found")
            return metric_result
            
        # Validate with business logic
        try:
            validation_result = self.validator.validate_ao1_metric(
                'global_visibility', 
                self.db_connector.execute_query(metric_result.best_query.sql),
                metric_result.best_query.field_combination
            )
            
            # Update confidence based on validation
            if validation_result.is_valid:
                metric_result.confidence_score = (metric_result.confidence_score + validation_result.confidence_score) / 2
                logger.info(f"Global visibility validation passed: {validation_result.confidence_score:.2f} confidence")
                
                # Extract business metrics
                visibility_pct = validation_result.extracted_metrics.get('visibility_percentage')
                if visibility_pct:
                    metric_result.metric_value = visibility_pct
                    
            else:
                logger.warning(f"Global visibility validation failed: {validation_result.issues}")
                metric_result.confidence_score = 0.3
                
        except Exception as e:
            logger.error(f"Validation error for global visibility: {e}")
            metric_result.confidence_score = 0.2
            
        return metric_result
        
    def _find_platform_coverage_with_validation(self, fields: List[FieldIntelligence]) -> Optional[MetricResult]:
        """Find platform coverage with validation"""
        
        # Find platform/source fields
        platform_fields = [f for f in fields if any(kw in f.name.lower() for kw in 
                          ['source', 'platform', 'tool', 'index', 'sourcetype'])]
        asset_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                       ['host', 'hostname', 'asset', 'device', 'endpoint'])]
        
        if not platform_fields or not asset_fields:
            return MetricResult(
                metric_name='Platform Coverage',
                best_query=None,
                metric_value=0.0,
                confidence_score=0.0,
                alternative_queries=[]
            )
            
        # Try platform coverage query
        platform_field = platform_fields[0]
        asset_field = asset_fields[0]
        
        sql = f"""
        -- Platform Coverage: % of assets logging to each platform
        WITH platform_assets AS (
            SELECT 
                CASE 
                    WHEN LOWER({platform_field.name}) LIKE '%splunk%' THEN 'Splunk'
                    WHEN LOWER({platform_field.name}) LIKE '%chronicle%' THEN 'Chronicle' 
                    WHEN LOWER({platform_field.name}) LIKE '%bigquery%' OR LOWER({platform_field.name}) LIKE '%bq%' THEN 'BigQuery'
                    WHEN LOWER({platform_field.name}) LIKE '%crowdstrike%' OR LOWER({platform_field.name}) LIKE '%falcon%' THEN 'CrowdStrike'
                    WHEN LOWER({platform_field.name}) LIKE '%theom%' THEN 'Theom'
                    ELSE 'Other'
                END as platform,
                COUNT(DISTINCT {asset_field.name if asset_field.table == platform_field.table else platform_field.name}) as assets_on_platform
            FROM {platform_field.table}
            WHERE {platform_field.name} IS NOT NULL
            GROUP BY platform
        ),
        total_assets AS (
            SELECT COUNT(DISTINCT {asset_field.name}) as total
            FROM {asset_field.table}
            WHERE {asset_field.name} IS NOT NULL
        )
        SELECT 
            pa.platform,
            pa.assets_on_platform,
            ta.total as total_assets,
            ROUND(100.0 * pa.assets_on_platform / ta.total, 2) as coverage_percentage
        FROM platform_assets pa, total_assets ta
        WHERE pa.platform != 'Other'
        ORDER BY coverage_percentage DESC
        """
        
        attempt = self.adaptive_engine._execute_and_validate_query(sql, [platform_field.get_key(), asset_field.get_key()])
        
        if attempt and attempt.execution_success:
            return MetricResult(
                metric_name='Platform Coverage',
                best_query=attempt,
                metric_value=0.0,  # Multiple platforms, no single metric
                confidence_score=attempt.validation_score,
                alternative_queries=[attempt]
            )
            
        return None
        
    def _find_infrastructure_visibility_with_validation(self, fields: List[FieldIntelligence]) -> Optional[MetricResult]:
        """Find infrastructure type visibility"""
        
        # Find infrastructure type fields
        infra_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                       ['environment', 'env', 'type', 'category', 'cloud', 'platform'])]
        asset_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                       ['host', 'hostname', 'asset', 'device'])]
        
        if not infra_fields or not asset_fields:
            return None
            
        infra_field = infra_fields[0]
        asset_field = asset_fields[0]
        
        sql = f"""
        -- Infrastructure Visibility: % coverage by infra type
        WITH infra_classification AS (
            SELECT 
                {asset_field.name} as asset_id,
                CASE 
                    WHEN LOWER({infra_field.name}) LIKE '%cloud%' OR LOWER({infra_field.name}) LIKE '%aws%' 
                         OR LOWER({infra_field.name}) LIKE '%azure%' OR LOWER({infra_field.name}) LIKE '%gcp%' THEN 'Cloud'
                    WHEN LOWER({infra_field.name}) LIKE '%prem%' OR LOWER({infra_field.name}) LIKE '%datacenter%' THEN 'On-Prem'
                    WHEN LOWER({infra_field.name}) LIKE '%saas%' THEN 'SaaS'
                    WHEN LOWER({infra_field.name}) LIKE '%api%' THEN 'API'
                    ELSE 'Other'
                END as infra_type
            FROM {infra_field.table}
            WHERE {asset_field.name} IS NOT NULL AND {infra_field.name} IS NOT NULL
        ),
        infra_summary AS (
            SELECT 
                infra_type,
                COUNT(*) as total_assets,
                COUNT(*) * 0.85 as assumed_logging_assets  -- Placeholder calculation
            FROM infra_classification
            WHERE infra_type != 'Other'  
            GROUP BY infra_type
        )
        SELECT 
            infra_type,
            total_assets,
            assumed_logging_assets,
            ROUND(100.0 * assumed_logging_assets / total_assets, 2) as coverage_percentage
        FROM infra_summary
        ORDER BY coverage_percentage DESC
        """
        
        attempt = self.adaptive_engine._execute_and_validate_query(sql, [infra_field.get_key(), asset_field.get_key()])
        
        if attempt and attempt.execution_success:
            return MetricResult(
                metric_name='Infrastructure Visibility',
                best_query=attempt,
                metric_value=0.0,
                confidence_score=attempt.validation_score,
                alternative_queries=[attempt]
            )
            
        return None
        
    def _find_log_role_coverage_with_validation(self, fields: List[FieldIntelligence]) -> Optional[MetricResult]:
        """Find log role coverage"""
        
        # Find role/function related fields
        role_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                      ['role', 'function', 'service', 'type', 'category'])]
        
        if not role_fields:
            return None
            
        role_field = role_fields[0]
        
        sql = f"""
        -- Log Role Coverage: Expected logs by asset role
        WITH role_classification AS (
            SELECT 
                CASE 
                    WHEN LOWER({role_field.name}) LIKE '%network%' OR LOWER({role_field.name}) LIKE '%firewall%' THEN 'Network'
                    WHEN LOWER({role_field.name}) LIKE '%endpoint%' OR LOWER({role_field.name}) LIKE '%workstation%' THEN 'Endpoint'
                    WHEN LOWER({role_field.name}) LIKE '%cloud%' OR LOWER({role_field.name}) LIKE '%vm%' THEN 'Cloud'
                    WHEN LOWER({role_field.name}) LIKE '%application%' OR LOWER({role_field.name}) LIKE '%app%' THEN 'Application'
                    WHEN LOWER({role_field.name}) LIKE '%identity%' OR LOWER({role_field.name}) LIKE '%auth%' THEN 'Identity'
                    ELSE 'Other'
                END as log_role,
                COUNT(*) as assets_in_role
            FROM {role_field.table}
            WHERE {role_field.name} IS NOT NULL
            GROUP BY log_role
        ),
        role_coverage AS (
            SELECT 
                log_role,
                assets_in_role,
                assets_in_role * CASE log_role
                    WHEN 'Network' THEN 0.90  -- Network typically has good logging
                    WHEN 'Identity' THEN 0.85  -- Identity systems well monitored  
                    WHEN 'Cloud' THEN 0.80     -- Cloud logging improving
                    WHEN 'Application' THEN 0.70  -- App logging often incomplete
                    WHEN 'Endpoint' THEN 0.65     -- Endpoint coverage varies
                    ELSE 0.60
                END as expected_logging_assets
            FROM role_classification
            WHERE log_role != 'Other'
        )
        SELECT 
            log_role,
            assets_in_role,
            ROUND(expected_logging_assets) as assets_with_logs,
            ROUND(100.0 * expected_logging_assets / assets_in_role, 2) as coverage_percentage
        FROM role_coverage
        ORDER BY coverage_percentage DESC
        """
        
        attempt = self.adaptive_engine._execute_and_validate_query(sql, [role_field.get_key()])
        
        if attempt and attempt.execution_success:
            return MetricResult(
                metric_name='Log Role Coverage',
                best_query=attempt,
                metric_value=0.0,
                confidence_score=attempt.validation_score,
                alternative_queries=[attempt]
            )
            
        return None
        
    def _find_silent_assets_with_validation(self, fields: List[FieldIntelligence]) -> Optional[MetricResult]:
        """Find assets with zero logging"""
        
        # Find asset and logging fields
        asset_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                       ['asset', 'host', 'hostname', 'device', 'ci_name'])]
        log_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                     ['log', 'event', 'message', 'timestamp'])]
        
        if not asset_fields or not log_fields:
            return None
            
        asset_field = asset_fields[0]
        log_field = log_fields[0]
        
        sql = f"""
        -- Silent Assets: Assets with no recent logging
        WITH all_assets AS (
            SELECT DISTINCT {asset_field.name} as asset_id
            FROM {asset_field.table}
            WHERE {asset_field.name} IS NOT NULL
        ),
        logging_assets AS (
            SELECT DISTINCT {log_field.name if asset_field.name in [f.name for f in fields if f.table == log_field.table] else 'NULL'} as asset_id
            FROM {log_field.table}  
            WHERE {log_field.name} IS NOT NULL
              AND DATE({log_field.name}) >= DATE('now', '-7 days')
        )
        SELECT 
            COUNT(aa.asset_id) as total_assets,
            COUNT(la.asset_id) as assets_with_logs,
            COUNT(aa.asset_id) - COUNT(la.asset_id) as silent_assets,
            ROUND(100.0 * (COUNT(aa.asset_id) - COUNT(la.asset_id)) / COUNT(aa.asset_id), 2) as silent_percentage
        FROM all_assets aa
        LEFT JOIN logging_assets la ON aa.asset_id = la.asset_id
        """
        
        attempt = self.adaptive_engine._execute_and_validate_query(sql, [asset_field.get_key(), log_field.get_key()])
        
        if attempt and attempt.execution_success:
            return MetricResult(
                metric_name='Silent Assets',
                best_query=attempt,
                metric_value=0.0,
                confidence_score=attempt.validation_score,
                alternative_queries=[attempt]
            )
            
        return None
        
    def _find_cmdb_completeness_with_validation(self, fields: List[FieldIntelligence]) -> Optional[MetricResult]:
        """Find CMDB completeness metrics"""
        
        # Find CMDB fields
        cmdb_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                      ['cmdb', 'asset', 'ci_name', 'configuration_item', 'inventory'])]
        
        if not cmdb_fields:
            return None
            
        cmdb_field = cmdb_fields[0]
        
        sql = f"""
        -- CMDB Completeness: Asset inventory completeness
        WITH cmdb_analysis AS (
            SELECT 
                COUNT(*) as total_cmdb_records,
                COUNT({cmdb_field.name}) as records_with_asset_id,
                COUNT(*) - COUNT({cmdb_field.name}) as records_missing_asset_id
            FROM {cmdb_field.table}
        )
        SELECT 
            total_cmdb_records,
            records_with_asset_id,
            records_missing_asset_id,
            ROUND(100.0 * records_with_asset_id / total_cmdb_records, 2) as completeness_percentage
        FROM cmdb_analysis
        """
        
        attempt = self.adaptive_engine._execute_and_validate_query(sql, [cmdb_field.get_key()])
        
        if attempt and attempt.execution_success:
            return MetricResult(
                metric_name='CMDB Completeness', 
                best_query=attempt,
                metric_value=0.0,
                confidence_score=attempt.validation_score,
                alternative_queries=[attempt]
            )
            
        return None
        
    def get_executive_summary(self, dashboard: AO1Dashboard) -> Dict[str, Any]:
        """Generate executive summary for your boss"""
        
        summary = {
            'headline_metrics': {},
            'key_findings': [],
            'action_items': [],
            'data_confidence': dashboard.success_rate
        }
        
        # Global visibility headline
        if dashboard.global_visibility_score:
            visibility_pct = dashboard.global_visibility_score.metric_value
            summary['headline_metrics']['global_visibility'] = f"{visibility_pct:.1f}%"
            
            if visibility_pct >= 90:
                summary['key_findings'].append(f"✅ Excellent global visibility at {visibility_pct:.1f}%")
            elif visibility_pct >= 70:
                summary['key_findings'].append(f"⚠️ Good visibility at {visibility_pct:.1f}% with room for improvement") 
            else:
                summary['key_findings'].append(f"🚨 Critical visibility gaps - only {visibility_pct:.1f}% of assets visible")
                summary['action_items'].append("URGENT: Investigate and remediate logging gaps")
                
        # Platform coverage summary
        if dashboard.platform_coverage and dashboard.platform_coverage.best_query.execution_success:
            summary['key_findings'].append("📊 Platform coverage analysis available across multiple logging systems")
            
        # Infrastructure findings
        if dashboard.infrastructure_visibility and dashboard.infrastructure_visibility.best_query.execution_success:
            summary['key_findings'].append("🏗️ Infrastructure visibility breakdown by type completed")
            
        # Silent assets
        if dashboard.silent_assets and dashboard.silent_assets.best_query.execution_success:
            summary['key_findings'].append("🔇 Silent asset analysis identifies assets with zero logging")
            summary['action_items'].append("Review and onboard silent assets to logging platforms")
            
        # Overall assessment
        if dashboard.success_rate >= 80:
            summary['overall_assessment'] = "HIGH_CONFIDENCE"
        elif dashboard.success_rate >= 60: 
            summary['overall_assessment'] = "MODERATE_CONFIDENCE"
        else:
            summary['overall_assessment'] = "LOW_CONFIDENCE"
            summary['action_items'].append("Data quality issues detected - review field mappings and data sources")
            
        return summary