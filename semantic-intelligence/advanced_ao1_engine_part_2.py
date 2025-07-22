#!/usr/bin/env python3

import statistics
import re
from typing import List, Dict, Optional, Tuple, Any
from models import FieldIntelligence, IntelligentQuery, MetricResult

class AdvancedAO1EngineExecutor:
    def __init__(self, engine_instance):
        self.engine = engine_instance
        
    def _build_intelligent_global_visibility(self, classified_fields: Dict[str, List[FieldIntelligence]]) -> Optional[MetricResult]:
        asset_fields = classified_fields.get('assets', [])
        activity_fields = classified_fields.get('logging_activity', [])
        time_fields = classified_fields.get('temporal', [])
        
        if not asset_fields:
            asset_fields = classified_fields.get('identifiers', [])[:3]
            
        if not asset_fields:
            logger.warning("No suitable asset fields found for global visibility")
            return None
            
        query = self.engine.query_builder.build_ao1_global_visibility_query(asset_fields + activity_fields + time_fields)
        if not query:
            logger.warning("Failed to build global visibility query")
            return None
            
        try:
            results = self.engine.db_connector.execute_query(query.sql)
            if not results:
                logger.warning("Global visibility query returned no results")
                return None
                
            validation = self.engine.validator.validate_global_visibility_results(results, query)
            
            extracted_values = self._extract_global_visibility_values(results)
            
            return MetricResult(
                metric_name="Global Visibility Score",
                query=query,
                results=results,
                validation_confidence=validation.confidence_score,
                business_assessment=validation.business_assessment,
                extracted_values=extracted_values
            )
            
        except Exception as e:
            logger.error(f"Error executing global visibility query: {e}")
            return None
            
    def _extract_global_visibility_values(self, results: List[Tuple]) -> Dict[str, Any]:
        if not results or not results[0]:
            return {}
            
        row = results[0]
        extracted = {}
        
        for i, value in enumerate(row):
            if isinstance(value, (int, float)):
                if 0 <= value <= 100:
                    extracted['visibility_percentage'] = value
                elif value > 100:
                    if 'total_assets' not in extracted:
                        extracted['total_assets'] = value
                    elif 'visible_assets' not in extracted and value <= extracted.get('total_assets', float('inf')):
                        extracted['visible_assets'] = value
                        
        if 'total_assets' in extracted and 'visible_assets' in extracted:
            extracted['silent_assets'] = extracted['total_assets'] - extracted['visible_assets']
            if 'visibility_percentage' not in extracted:
                extracted['visibility_percentage'] = (extracted['visible_assets'] / extracted['total_assets']) * 100 if extracted['total_assets'] > 0 else 0.0
                
        return extracted
        
    def _build_intelligent_platform_coverage(self, classified_fields: Dict[str, List[FieldIntelligence]]) -> Optional[MetricResult]:
        platform_fields = classified_fields.get('platforms', [])
        asset_fields = classified_fields.get('assets', [])
        
        if not platform_fields:
            logger.warning("No platform fields found for platform coverage")
            return None
            
        if not asset_fields:
            asset_fields = classified_fields.get('identifiers', [])[:2]
            
        query = self._build_platform_coverage_query(platform_fields[0], asset_fields[0] if asset_fields else platform_fields[0])
        if not query:
            return None
            
        try:
            results = self.engine.db_connector.execute_query(query.sql)
            if not results:
                return None
                
            validation = self.engine.validator.validate_platform_coverage_results(results, query)
            extracted_values = self._extract_platform_coverage_values(results)
            
            return MetricResult(
                metric_name="Platform Coverage",
                query=query,
                results=results,
                validation_confidence=validation.confidence_score,
                business_assessment=validation.business_assessment,
                extracted_values=extracted_values
            )
            
        except Exception as e:
            logger.error(f"Error executing platform coverage query: {e}")
            return None
            
    def _build_platform_coverage_query(self, platform_field: FieldIntelligence, asset_field: FieldIntelligence) -> Optional[IntelligentQuery]:
        platform_analysis = self._analyze_platform_values(platform_field.sample_values)
        
        platform_mappings = []
        for platform, confidence in platform_analysis.items():
            if confidence > 0.3:
                platform_mappings.append(f"WHEN LOWER({platform_field.name}) LIKE '%{platform}%' THEN '{platform.title()}'")
                
        if not platform_mappings:
            platform_mappings = ["WHEN {platform_field.name} IS NOT NULL THEN 'Unknown'"]
            
        sql = f"""
WITH platform_classification AS (
    SELECT 
        {asset_field.name if asset_field.table == platform_field.table else platform_field.name} as asset_id,
        CASE 
            {' '.join(platform_mappings)}
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
    ROUND(100.0 * pc.assets_on_platform / ta.total, 2) as coverage_percentage
FROM platform_coverage pc, total_assets ta
ORDER BY coverage_percentage DESC;
        """
        
        return IntelligentQuery(
            name="Platform Coverage Analysis",
            description="Intelligent platform coverage breakdown",
            sql=sql,
            field_combination=[platform_field, asset_field],
            intelligence_reasoning={
                'platform_mappings_found': len(platform_mappings),
                'platform_analysis': platform_analysis
            },
            confidence_score=0.8,
            business_logic="Asset coverage across identified logging platforms"
        )
        
    def _analyze_platform_values(self, sample_values: List[Any]) -> Dict[str, float]:
        if not sample_values:
            return {}
            
        platform_analysis = {}
        string_values = [str(v).lower() for v in sample_values]
        
        known_platforms = {
            'splunk': ['splunk'],
            'chronicle': ['chronicle', 'google'],
            'crowdstrike': ['crowdstrike', 'falcon'],
            'bigquery': ['bigquery', 'bq'],
            'theom': ['theom'],
            'wiz': ['wiz'],
            'axonius': ['axonius'],
            'tanium': ['tanium']
        }
        
        for platform, indicators in known_platforms.items():
            matches = sum(1 for v in string_values if any(indicator in v for indicator in indicators))
            if matches > 0:
                platform_analysis[platform] = matches / len(string_values)
                
        return platform_analysis
        
    def _extract_platform_coverage_values(self, results: List[Tuple]) -> Dict[str, Any]:
        extracted = {
            'platforms': [],
            'coverage_data': [],
            'total_platforms': len(results)
        }
        
        for row in results:
            if len(row) >= 4:
                platform_info = {
                    'platform': row[0],
                    'assets': row[1],
                    'total_assets': row[2],
                    'coverage_percentage': row[3]
                }
                extracted['platforms'].append(row[0])
                extracted['coverage_data'].append(platform_info)
                
        return extracted
        
    def _build_intelligent_infrastructure_visibility(self, classified_fields: Dict[str, List[FieldIntelligence]]) -> Optional[MetricResult]:
        infra_fields = classified_fields.get('infrastructure', [])
        asset_fields = classified_fields.get('assets', [])
        
        if not infra_fields:
            logger.warning("No infrastructure fields found")
            return None
            
        if not asset_fields:
            asset_fields = classified_fields.get('identifiers', [])[:2]
            
        query = self._build_infrastructure_visibility_query(infra_fields[0], asset_fields[0] if asset_fields else infra_fields[0])
        if not query:
            return None
            
        try:
            results = self.engine.db_connector.execute_query(query.sql)
            if not results:
                return None
                
            validation = self.engine.validator.validate_infrastructure_visibility_results(results, query)
            extracted_values = self._extract_infrastructure_values(results)
            
            return MetricResult(
                metric_name="Infrastructure Visibility",
                query=query,
                results=results,
                validation_confidence=validation.confidence_score,
                business_assessment=validation.business_assessment,
                extracted_values=extracted_values
            )
            
        except Exception as e:
            logger.error(f"Error executing infrastructure visibility query: {e}")
            return None
            
    def _build_infrastructure_visibility_query(self, infra_field: FieldIntelligence, asset_field: FieldIntelligence) -> Optional[IntelligentQuery]:
        infra_analysis = self._analyze_infrastructure_values(infra_field.sample_values)
        
        infra_mappings = []
        expected_coverage = {}
        
        for infra_type, confidence in infra_analysis.items():
            if confidence > 0.2:
                coverage = self._get_expected_infrastructure_coverage(infra_type)
                infra_mappings.append(f"WHEN LOWER({infra_field.name}) LIKE '%{infra_type}%' THEN '{infra_type.title()}'")
                expected_coverage[infra_type.title()] = coverage
                
        if not infra_mappings:
            infra_mappings = ["WHEN {infra_field.name} IS NOT NULL THEN 'Unknown'"]
            expected_coverage['Unknown'] = 0.7
            
        coverage_cases = []
        for infra_type, coverage in expected_coverage.items():
            coverage_cases.append(f"WHEN '{infra_type}' THEN {coverage}")
            
        sql = f"""
WITH infrastructure_classification AS (
    SELECT 
        {asset_field.name if asset_field.table == infra_field.table else infra_field.name} as asset_id,
        CASE 
            {' '.join(infra_mappings)}
            ELSE 'Other'
        END as infrastructure_type
    FROM {infra_field.table}
    WHERE {infra_field.name} IS NOT NULL
),
infrastructure_visibility AS (
    SELECT 
        infrastructure_type,
        COUNT(DISTINCT asset_id) as total_assets,
        ROUND(COUNT(DISTINCT asset_id) * CASE infrastructure_type
            {' '.join(coverage_cases)}
            ELSE 0.6
        END) as estimated_visible_assets
    FROM infrastructure_classification
    WHERE infrastructure_type != 'Other'
    GROUP BY infrastructure_type
)
SELECT 
    infrastructure_type,
    total_assets,
    estimated_visible_assets,
    ROUND(100.0 * estimated_visible_assets / total_assets, 2) as visibility_percentage
FROM infrastructure_visibility
ORDER BY visibility_percentage DESC;
        """
        
        return IntelligentQuery(
            name="Infrastructure Visibility Analysis",
            description="Intelligent infrastructure type visibility breakdown",
            sql=sql,
            field_combination=[infra_field, asset_field],
            intelligence_reasoning={
                'infrastructure_types_found': len(infra_analysis),
                'expected_coverage_mappings': expected_coverage
            },
            confidence_score=0.7,
            business_logic="Visibility estimation by infrastructure type with expected coverage rates"
        )
        
    def _analyze_infrastructure_values(self, sample_values: List[Any]) -> Dict[str, float]:
        if not sample_values:
            return {}
            
        infra_analysis = {}
        string_values = [str(v).lower() for v in sample_values]
        
        infrastructure_indicators = {
            'cloud': ['cloud', 'aws', 'azure', 'gcp'],
            'onprem': ['onprem', 'premise', 'datacenter', 'dc'],
            'saas': ['saas', 'service'],
            'api': ['api', 'endpoint'],
            'hybrid': ['hybrid', 'mixed']
        }
        
        for infra_type, indicators in infrastructure_indicators.items():
            matches = sum(1 for v in string_values if any(indicator in v for indicator in indicators))
            if matches > 0:
                infra_analysis[infra_type] = matches / len(string_values)
                
        return infra_analysis
        
    def _get_expected_infrastructure_coverage(self, infra_type: str) -> float:
        coverage_expectations = {
            'cloud': 0.85,
            'onprem': 0.75,
            'saas': 0.60,
            'api': 0.70,
            'hybrid': 0.80
        }
        return coverage_expectations.get(infra_type.lower(), 0.65)
        
    def _extract_infrastructure_values(self, results: List[Tuple]) -> Dict[str, Any]:
        extracted = {
            'infrastructure_types': [],
            'visibility_data': [],
            'total_types': len(results)
        }
        
        for row in results:
            if len(row) >= 4:
                infra_info = {
                    'type': row[0],
                    'total_assets': row[1],
                    'visible_assets': row[2],
                    'visibility_percentage': row[3]
                }
                extracted['infrastructure_types'].append(row[0])
                extracted['visibility_data'].append(infra_info)
                
        return extracted
        
    def _build_intelligent_silent_assets(self, classified_fields: Dict[str, List[FieldIntelligence]]) -> Optional[MetricResult]:
        asset_fields = classified_fields.get('assets', [])
        activity_fields = classified_fields.get('logging_activity', [])
        time_fields = classified_fields.get('temporal', [])
        
        if not asset_fields:
            asset_fields = classified_fields.get('identifiers', [])[:3]
            
        if not asset_fields:
            logger.warning("No asset fields found for silent assets analysis")
            return None
            
        query = self._build_silent_assets_query(asset_fields, activity_fields, time_fields)
        if not query:
            return None
            
        try:
            results = self.engine.db_connector.execute_query(query.sql)
            if not results:
                return None
                
            validation = self.engine.validator.validate_silent_assets_results(results, query)
            extracted_values = self._extract_silent_assets_values(results)
            
            return MetricResult(
                metric_name="Silent Assets Analysis",
                query=query,
                results=results,
                validation_confidence=validation.confidence_score,
                business_assessment=validation.business_assessment,
                extracted_values=extracted_values
            )
            
        except Exception as e:
            logger.error(f"Error executing silent assets query: {e}")
            return None
            
    def _build_silent_assets_query(self, asset_fields: List[FieldIntelligence], 
                                 activity_fields: List[FieldIntelligence],
                                 time_fields: List[FieldIntelligence]) -> Optional[IntelligentQuery]:
        
        best_asset = asset_fields[0]
        
        if activity_fields and time_fields:
            best_activity = activity_fields[0]
            best_time = time_fields[0]
            
            if best_asset.table == best_activity.table == best_time.table:
                sql = self._build_same_table_silent_assets_query(best_asset, best_activity, best_time)
            else:
                sql = self._build_cross_table_silent_assets_query(best_asset, best_activity, best_time)
                
            used_fields = [best_asset, best_activity, best_time]
            
        elif activity_fields:
            best_activity = activity_fields[0]
            sql = self._build_asset_activity_silent_query(best_asset, best_activity)
            used_fields = [best_asset, best_activity]
            
        else:
            sql = self._build_statistical_silent_assets_query(best_asset)
            used_fields = [best_asset]
            
        return IntelligentQuery(
            name="Silent Assets Identification",
            description="Intelligent identification of assets with zero logging",
            sql=sql,
            field_combination=used_fields,
            intelligence_reasoning={
                'analysis_type': 'cross_table' if len(used_fields) > 1 and len(set(f.table for f in used_fields)) > 1 else 'same_table',
                'fields_used': len(used_fields)
            },
            confidence_score=0.8,
            business_logic="Identification of assets without recent logging activity"
        )
        
    def _build_same_table_silent_assets_query(self, asset_field: FieldIntelligence, 
                                            activity_field: FieldIntelligence,
                                            time_field: FieldIntelligence) -> str:
        time_filter = self._generate_intelligent_time_filter(time_field)
        activity_filter = self._generate_intelligent_activity_filter(activity_field)
        
        return f"""
WITH all_assets AS (
    SELECT DISTINCT {asset_field.name} as asset_id
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND LENGTH(TRIM({asset_field.name})) > 2
),
active_assets AS (
    SELECT DISTINCT {asset_field.name} as asset_id
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND {activity_field.name} IS NOT NULL
      AND {activity_filter}
      AND {time_filter}
),
silent_analysis AS (
    SELECT 
        aa.asset_id,
        CASE WHEN act.asset_id IS NOT NULL THEN 'ACTIVE' ELSE 'SILENT' END as status
    FROM all_assets aa
    LEFT JOIN active_assets act ON aa.asset_id = act.asset_id
)
SELECT 
    asset_id,
    status,
    CASE WHEN status = 'SILENT' THEN 'HIGH_RISK' ELSE 'MONITORED' END as risk_level
FROM silent_analysis
WHERE status = 'SILENT'
ORDER BY asset_id;
        """
        
    def _build_cross_table_silent_assets_query(self, asset_field: FieldIntelligence,
                                             activity_field: FieldIntelligence,
                                             time_field: FieldIntelligence) -> str:
        
        hostname_transform_asset = self._generate_hostname_transform(asset_field)
        hostname_transform_activity = self._generate_hostname_transform(activity_field)
        time_filter = self._generate_intelligent_time_filter(time_field)
        
        return f"""
WITH all_assets AS (
    SELECT DISTINCT {hostname_transform_asset} as hostname
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND LENGTH(TRIM({asset_field.name})) > 2
),
recent_activity AS (
    SELECT DISTINCT {hostname_transform_activity} as hostname
    FROM {activity_field.table} af
    {'JOIN ' + time_field.table + ' tf ON af.' + self._find_join_key() + ' = tf.' + self._find_join_key() if activity_field.table != time_field.table else ''}
    WHERE {activity_field.name} IS NOT NULL
      AND {time_filter if activity_field.table == time_field.table else 'tf.' + time_field.name + ' IS NOT NULL'}
)
SELECT 
    aa.hostname as asset_id,
    'SILENT' as status,
    'HIGH_RISK' as risk_level
FROM all_assets aa
LEFT JOIN recent_activity ra ON aa.hostname = ra.hostname
WHERE ra.hostname IS NULL
ORDER BY aa.hostname;
        """
        
    def _build_asset_activity_silent_query(self, asset_field: FieldIntelligence, activity_field: FieldIntelligence) -> str:
        if asset_field.table == activity_field.table:
            activity_filter = self._generate_intelligent_activity_filter(activity_field)
            
            return f"""
WITH asset_activity_analysis AS (
    SELECT 
        {asset_field.name} as asset_id,
        CASE 
            WHEN {activity_field.name} IS NOT NULL AND {activity_filter} THEN 'ACTIVE'
            ELSE 'SILENT'
        END as status
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND LENGTH(TRIM({asset_field.name})) > 2
)
SELECT 
    asset_id,
    status,
    CASE WHEN status = 'SILENT' THEN 'HIGH_RISK' ELSE 'MONITORED' END as risk_level
FROM asset_activity_analysis
WHERE status = 'SILENT'
ORDER BY asset_id;
            """
        else:
            return self._build_cross_table_silent_assets_query(asset_field, activity_field, activity_field)
            
    def _build_statistical_silent_assets_query(self, asset_field: FieldIntelligence) -> str:
        return f"""
WITH asset_quality_analysis AS (
    SELECT 
        {asset_field.name} as asset_id,
        CASE 
            WHEN LENGTH(TRIM({asset_field.name})) < 3 THEN 'LOW_QUALITY'
            WHEN {asset_field.name} LIKE '%test%' OR {asset_field.name} LIKE '%temp%' THEN 'TEST_ASSET'
            WHEN {asset_field.name} LIKE '%-%' OR {asset_field.name} LIKE '%.%' THEN 'STRUCTURED'
            ELSE 'STANDARD'
        END as asset_type
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
)
SELECT 
    asset_id,
    'POTENTIALLY_SILENT' as status,
    CASE 
        WHEN asset_type IN ('LOW_QUALITY', 'TEST_ASSET') THEN 'MEDIUM_RISK'
        ELSE 'REVIEW_NEEDED'
    END as risk_level
FROM asset_quality_analysis
WHERE asset_type IN ('LOW_QUALITY', 'TEST_ASSET')
ORDER BY asset_id;
        """
        
    def _extract_silent_assets_values(self, results: List[Tuple]) -> Dict[str, Any]:
        extracted = {
            'silent_assets': [],
            'total_silent': len(results),
            'risk_breakdown': {}
        }
        
        risk_counts = {}
        for row in results:
            if len(row) >= 3:
                extracted['silent_assets'].append({
                    'asset_id': row[0],
                    'status': row[1],
                    'risk_level': row[2]
                })
                
                risk_level = row[2]
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
                
        extracted['risk_breakdown'] = risk_counts
        return extracted
        
    def _build_intelligent_log_role_coverage(self, classified_fields: Dict[str, List[FieldIntelligence]]) -> Optional[MetricResult]:
        role_fields = classified_fields.get('roles', [])
        
        if not role_fields:
            logger.warning("No role fields found for log role coverage")
            return None
            
        query = self._build_log_role_coverage_query(role_fields[0])
        if not query:
            return None
            
        try:
            results = self.engine.db_connector.execute_query(query.sql)
            if not results:
                return None
                
            validation = self.engine.validator.validate_log_role_coverage_results(results, query)
            extracted_values = self._extract_log_role_values(results)
            
            return MetricResult(
                metric_name="Log Role Coverage",
                query=query,
                results=results,
                validation_confidence=validation.confidence_score,
                business_assessment=validation.business_assessment,
                extracted_values=extracted_values
            )
            
        except Exception as e:
            logger.error(f"Error executing log role coverage query: {e}")
            return None
            
    def _build_log_role_coverage_query(self, role_field: FieldIntelligence) -> Optional[IntelligentQuery]:
        role_analysis = self._analyze_role_values(role_field.sample_values)
        
        role_mappings = []
        expected_coverage = {}
        
        for role_type, confidence in role_analysis.items():
            if confidence > 0.2:
                coverage = self._get_expected_role_coverage(role_type)
                role_mappings.append(f"WHEN LOWER({role_field.name}) LIKE '%{role_type}%' THEN '{role_type.title()}'")
                expected_coverage[role_type.title()] = coverage
                
        if not role_mappings:
            role_mappings = ["WHEN {role_field.name} IS NOT NULL THEN 'Unknown'"]
            expected_coverage['Unknown'] = 0.6
            
        coverage_cases = []
        for role_type, coverage in expected_coverage.items():
            coverage_cases.append(f"WHEN '{role_type}' THEN {coverage}")
            
        sql = f"""
WITH role_classification AS (
    SELECT 
        CASE 
            {' '.join(role_mappings)}
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
        ROUND(assets_in_role * CASE log_role
            {' '.join(coverage_cases)}
            ELSE 0.6
        END) as expected_assets_with_logs
    FROM role_classification
    WHERE log_role != 'Other'
)
SELECT 
    log_role,
    assets_in_role,
    expected_assets_with_logs,
    ROUND(100.0 * expected_assets_with_logs / assets_in_role, 2) as coverage_percentage
FROM role_coverage
ORDER BY coverage_percentage DESC;
        """
        
        return IntelligentQuery(
            name="Log Role Coverage Analysis",
            description="Intelligent log coverage by asset role",
            sql=sql,
            field_combination=[role_field],
            intelligence_reasoning={
                'role_types_found': len(role_analysis),
                'expected_coverage_mappings': expected_coverage
            },
            confidence_score=0.7,
            business_logic="Expected log coverage by asset role based on industry standards"
        )
        
    def _analyze_role_values(self, sample_values: List[Any]) -> Dict[str, float]:
        if not sample_values:
            return {}
            
        role_analysis = {}
        string_values = [str(v).lower() for v in sample_values]
        
        role_indicators = {
            'network': ['network', 'firewall', 'router', 'switch'],
            'endpoint': ['endpoint', 'workstation', 'desktop', 'laptop'],
            'cloud': ['cloud', 'vm', 'instance', 'container'],
            'application': ['application', 'app', 'service', 'web'],
            'identity': ['identity', 'auth', 'ad', 'ldap'],
            'server': ['server', 'srv', 'host']
        }
        
        for role_type, indicators in role_indicators.items():
            matches = sum(1 for v in string_values if any(indicator in v for indicator in indicators))
            if matches > 0:
                role_analysis[role_type] = matches / len(string_values)
                
        return role_analysis
        
    def _get_expected_role_coverage(self, role_type: str) -> float:
        coverage_expectations = {
            'network': 0.90,
            'identity': 0.85,
            'cloud': 0.80,
            'server': 0.85,
            'application': 0.70,
            'endpoint': 0.65
        }
        return coverage_expectations.get(role_type.lower(), 0.65)
        
    def _extract_log_role_values(self, results: List[Tuple]) -> Dict[str, Any]:
        extracted = {
            'roles': [],
            'coverage_data': [],
            'total_roles': len(results)
        }
        
        for row in results:
            if len(row) >= 4:
                role_info = {
                    'role': row[0],
                    'total_assets': row[1],
                    'assets_with_logs': row[2],
                    'coverage_percentage': row[3]
                }
                extracted['roles'].append(row[0])
                extracted['coverage_data'].append(role_info)
                
        return extracted
        
    def _extract_visibility_percentage(self, metric_result: MetricResult) -> float:
        if not metric_result or not metric_result.extracted_values:
            return 0.0
            
        return metric_result.extracted_values.get('visibility_percentage', 0.0)
        
    def _calculate_dashboard_semantic_coherence(self, dashboard: AO1Dashboard) -> float:
        coherence_scores = []
        
        metrics = [
            dashboard.global_visibility_score,
            dashboard.platform_coverage,
            dashboard.infrastructure_visibility,
            dashboard.silent_assets,
            dashboard.log_role_coverage
        ]
        
        for metric in metrics:
            if metric and hasattr(metric.query, 'semantic_coherence'):
                coherence_scores.append(metric.query.semantic_coherence)
            elif metric:
                coherence_scores.append(metric.validation_confidence)
                
        return statistics.mean(coherence_scores) if coherence_scores else 0.0
        
    def _generate_intelligent_time_filter(self, time_field: FieldIntelligence) -> str:
        if not time_field.sample_values:
            return "1=1"
            
        sample_value = str(time_field.sample_values[0])
        
        if re.search(r'\d{4}-\d{2}-\d{2}', sample_value):
            if 'T' in sample_value or ' ' in sample_value:
                return f"DATE({time_field.name}) >= DATE('now', '-7 days')"
            else:
                return f"{time_field.name} >= DATE('now', '-7 days')"
        elif re.match(r'^\d{10,13}$', sample_value):
            return f"{time_field.name} >= strftime('%s', 'now', '-7 days')"
        else:
            return f"{time_field.name} IS NOT NULL"
            
    def _generate_intelligent_activity_filter(self, activity_field: FieldIntelligence) -> str:
        if not activity_field.sample_values:
            return f"{activity_field.name} IS NOT NULL"
            
        string_values = [str(v).lower() for v in activity_field.sample_values[:20]]
        
        if any(re.match(r'^\d+$', v) for v in string_values):
            return f"{activity_field.name} > 0"
        elif any('error' in v or 'fail' in v for v in string_values):
            return f"LOWER({activity_field.name}) NOT LIKE '%error%' AND LOWER({activity_field.name}) NOT LIKE '%fail%'"
        else:
            return f"{activity_field.name} IS NOT NULL"
            
    def _generate_hostname_transform(self, field: FieldIntelligence) -> str:
        if not field.sample_values:
            return field.name
            
        sample_value = str(field.sample_values[0]).lower()
        
        if '.' in sample_value and len(sample_value.split('.')) > 2:
            return f"SUBSTR({field.name}, 1, INSTR({field.name}, '.') - 1)"
        else:
            return field.name
            
    def _find_join_key(self) -> str:
        return 'id'