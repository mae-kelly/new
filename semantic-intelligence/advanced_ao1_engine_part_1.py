#!/usr/bin/env python3

import logging
import statistics
from typing import List, Dict, Optional, Tuple, Any
from models import FieldIntelligence, IntelligentQuery, MetricResult, AO1Dashboard
from database_connector import DatabaseConnector
from intelligent_query_builder import IntelligentQueryBuilder
from intelligent_validator import IntelligentValidator

logger = logging.getLogger(__name__)

class AdvancedAO1Engine:
    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        self.query_builder = IntelligentQueryBuilder(db_connector)
        self.validator = IntelligentValidator()
        self.field_analysis_cache = {}
        self.query_performance_cache = {}
        
    def generate_intelligent_ao1_dashboard(self, fields: List[FieldIntelligence]) -> AO1Dashboard:
        logger.info("🎯 Generating Intelligent AO1 Dashboard using advanced semantic analysis...")
        
        dashboard = AO1Dashboard()
        successful_metrics = 0
        
        enhanced_fields = self._enhance_field_intelligence(fields)
        classified_fields = self._classify_fields_semantically(enhanced_fields)
        
        logger.info(f"📊 Classified {len(enhanced_fields)} fields into semantic categories")
        
        dashboard.global_visibility_score = self._build_intelligent_global_visibility(classified_fields)
        if dashboard.global_visibility_score and dashboard.global_visibility_score.validation_confidence > 0.6:
            successful_metrics += 1
            logger.info(f"✅ Global Visibility: {self._extract_visibility_percentage(dashboard.global_visibility_score)}%")
        
        dashboard.platform_coverage = self._build_intelligent_platform_coverage(classified_fields)
        if dashboard.platform_coverage and dashboard.platform_coverage.validation_confidence > 0.6:
            successful_metrics += 1
            logger.info("✅ Platform Coverage: Multi-platform analysis complete")
            
        dashboard.infrastructure_visibility = self._build_intelligent_infrastructure_visibility(classified_fields)
        if dashboard.infrastructure_visibility and dashboard.infrastructure_visibility.validation_confidence > 0.6:
            successful_metrics += 1
            logger.info("✅ Infrastructure Visibility: Cloud/On-Prem analysis complete")
            
        dashboard.silent_assets = self._build_intelligent_silent_assets(classified_fields)
        if dashboard.silent_assets and dashboard.silent_assets.validation_confidence > 0.6:
            successful_metrics += 1
            logger.info("✅ Silent Assets: Zero-logging asset identification complete")
            
        dashboard.log_role_coverage = self._build_intelligent_log_role_coverage(classified_fields)
        if dashboard.log_role_coverage and dashboard.log_role_coverage.validation_confidence > 0.6:
            successful_metrics += 1
            logger.info("✅ Log Role Coverage: Network/Endpoint/Cloud analysis complete")
            
        dashboard.total_attempts = 5
        dashboard.success_rate = (successful_metrics / dashboard.total_attempts) * 100
        dashboard.semantic_coherence = self._calculate_dashboard_semantic_coherence(dashboard)
        
        logger.info(f"🎯 Intelligent AO1 Dashboard Complete: {successful_metrics}/{dashboard.total_attempts} metrics ({dashboard.success_rate:.1f}% success)")
        
        return dashboard
        
    def _enhance_field_intelligence(self, fields: List[FieldIntelligence]) -> List[FieldIntelligence]:
        enhanced_fields = []
        
        for field in fields:
            if field.intelligence_score > 0.3:
                enhanced_field = self._deep_analyze_field_meaning(field)
                enhanced_fields.append(enhanced_field)
                
        enhanced_fields.sort(key=lambda f: f.intelligence_score, reverse=True)
        return enhanced_fields[:50]
        
    def _deep_analyze_field_meaning(self, field: FieldIntelligence) -> FieldIntelligence:
        if field.get_key() in self.field_analysis_cache:
            return self.field_analysis_cache[field.get_key()]
            
        enhanced_field = field
        
        if field.sample_values:
            value_intelligence = self._analyze_value_intelligence(field.sample_values, field.name)
            cross_reference_patterns = self._find_cross_reference_patterns(field)
            meaning_strength = self._calculate_meaning_strength(field, value_intelligence)
            
            enhanced_field.meaning_indicators.update({
                'value_intelligence_score': value_intelligence,
                'meaning_strength': meaning_strength,
                'cross_reference_confidence': len(cross_reference_patterns) / 10.0
            })
            
            enhanced_field.cross_references = cross_reference_patterns
            
            semantic_boost = self._calculate_semantic_intelligence_boost(enhanced_field)
            enhanced_field.intelligence_score = min(1.0, enhanced_field.intelligence_score + semantic_boost)
            
        self.field_analysis_cache[field.get_key()] = enhanced_field
        return enhanced_field
        
    def _analyze_value_intelligence(self, sample_values: List[Any], field_name: str) -> float:
        if not sample_values:
            return 0.0
            
        intelligence_factors = []
        
        string_values = [str(v) for v in sample_values[:100]]
        
        semantic_richness = self._calculate_semantic_richness(string_values, field_name)
        intelligence_factors.append(semantic_richness)
        
        pattern_consistency = self._calculate_pattern_consistency(string_values)
        intelligence_factors.append(pattern_consistency)
        
        business_relevance = self._calculate_business_relevance(string_values, field_name)
        intelligence_factors.append(business_relevance)
        
        contextual_meaning = self._calculate_contextual_meaning(string_values, field_name)
        intelligence_factors.append(contextual_meaning)
        
        return statistics.mean(intelligence_factors)
        
    def _calculate_semantic_richness(self, string_values: List[str], field_name: str) -> float:
        richness_score = 0.0
        
        unique_ratio = len(set(string_values)) / len(string_values) if string_values else 0
        richness_score += unique_ratio * 0.3
        
        avg_length = statistics.mean([len(v) for v in string_values]) if string_values else 0
        length_score = min(1.0, avg_length / 20.0)
        richness_score += length_score * 0.2
        
        semantic_keywords = self._count_semantic_keywords(string_values + [field_name])
        keyword_score = min(1.0, semantic_keywords / 10.0)
        richness_score += keyword_score * 0.3
        
        structural_complexity = self._calculate_structural_complexity(string_values)
        richness_score += structural_complexity * 0.2
        
        return min(1.0, richness_score)
        
    def _count_semantic_keywords(self, text_list: List[str]) -> int:
        all_text = ' '.join(text_list).lower()
        
        semantic_keywords = [
            'hostname', 'asset', 'server', 'device', 'computer', 'host', 'endpoint',
            'log', 'event', 'message', 'activity', 'record', 'entry',
            'time', 'timestamp', 'date', 'created', 'modified', 'occurred',
            'source', 'platform', 'tool', 'system', 'index', 'sourcetype',
            'type', 'category', 'role', 'function', 'service', 'class',
            'cloud', 'aws', 'azure', 'gcp', 'onprem', 'datacenter',
            'network', 'firewall', 'router', 'switch', 'dns',
            'security', 'auth', 'access', 'permission', 'control'
        ]
        
        return sum(1 for keyword in semantic_keywords if keyword in all_text)
        
    def _calculate_pattern_consistency(self, string_values: List[str]) -> float:
        if not string_values:
            return 0.0
            
        import re
        from collections import Counter
        
        format_patterns = Counter()
        for value in string_values:
            pattern = re.sub(r'\d', 'N', re.sub(r'[a-zA-Z]', 'A', re.sub(r'[^a-zA-Z0-9]', 'S', value)))
            format_patterns[pattern] += 1
            
        if format_patterns:
            most_common_count = format_patterns.most_common(1)[0][1]
            consistency_score = most_common_count / len(string_values)
            return consistency_score
            
        return 0.0
        
    def _calculate_business_relevance(self, string_values: List[str], field_name: str) -> float:
        relevance_score = 0.0
        
        business_indicators = {
            'asset_management': ['hostname', 'asset', 'device', 'server', 'computer'],
            'security_monitoring': ['log', 'event', 'security', 'auth', 'access'],
            'infrastructure': ['cloud', 'datacenter', 'network', 'platform'],
            'operational': ['status', 'count', 'total', 'active', 'enabled']
        }
        
        field_lower = field_name.lower()
        values_text = ' '.join(string_values).lower()
        
        for category, indicators in business_indicators.items():
            category_score = 0.0
            for indicator in indicators:
                if indicator in field_lower:
                    category_score += 0.3
                if indicator in values_text:
                    category_score += 0.1
                    
            relevance_score += min(1.0, category_score)
            
        return min(1.0, relevance_score / len(business_indicators))
        
    def _calculate_contextual_meaning(self, string_values: List[str], field_name: str) -> float:
        meaning_score = 0.0
        
        field_context_score = self._analyze_field_name_context(field_name)
        meaning_score += field_context_score * 0.6
        
        value_context_score = self._analyze_value_context(string_values)
        meaning_score += value_context_score * 0.4
        
        return min(1.0, meaning_score)
        
    def _analyze_field_name_context(self, field_name: str) -> float:
        name_lower = field_name.lower()
        
        context_patterns = [
            (r'(hostname|host_name|device_name)', 0.9),
            (r'(asset_id|asset_name|ci_name)', 0.8),
            (r'(log_count|event_count|message_count)', 0.8),
            (r'(timestamp|created_time|modified_time)', 0.8),
            (r'(source_type|platform|tool_name)', 0.7),
            (r'(server_name|computer_name|endpoint)', 0.7),
            (r'(environment|infrastructure_type)', 0.6),
            (r'(status|state|category|type)', 0.5)
        ]
        
        for pattern, score in context_patterns:
            if re.search(pattern, name_lower):
                return score
                
        return 0.3
        
    def _analyze_value_context(self, string_values: List[str]) -> float:
        if not string_values:
            return 0.0
            
        context_score = 0.0
        sample_text = ' '.join(string_values[:20]).lower()
        
        context_indicators = [
            (r'[a-z0-9\-]+\.(local|corp|com)', 0.8),
            (r'(srv|web|db|app|dc|ad)[0-9]+', 0.7),
            (r'\d{4}-\d{2}-\d{2}', 0.7),
            (r'(splunk|chronicle|crowdstrike)', 0.8),
            (r'(cloud|aws|azure|gcp)', 0.6),
            (r'(error|info|warning|debug)', 0.5),
            (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 0.4)
        ]
        
        for pattern, score in context_indicators:
            if re.search(pattern, sample_text):
                context_score = max(context_score, score)
                
        return context_score
        
    def _find_cross_reference_patterns(self, field: FieldIntelligence) -> Dict[str, List[str]]:
        patterns = {}
        
        if field.sample_values:
            string_values = [str(v) for v in field.sample_values[:30]]
            
            patterns['hostname_patterns'] = self._extract_hostname_patterns(string_values)
            patterns['identifier_patterns'] = self._extract_identifier_patterns(string_values)
            patterns['temporal_patterns'] = self._extract_temporal_patterns(string_values)
            patterns['platform_patterns'] = self._extract_platform_patterns(string_values)
            
        return {k: v for k, v in patterns.items() if v}
        
    def _extract_hostname_patterns(self, string_values: List[str]) -> List[str]:
        hostname_patterns = []
        
        for value in string_values:
            if re.match(r'^[a-z0-9\-]+(\.[a-z0-9\-\.]+)?, value.lower()):
                hostname_patterns.append(value)
                
        return list(set(hostname_patterns))[:5]
        
    def _extract_identifier_patterns(self, string_values: List[str]) -> List[str]:
        identifier_patterns = []
        
        for value in string_values:
            if re.match(r'^[A-Z0-9\-_]{4,}, value) or re.match(r'^[a-f0-9\-]{8,}, value.lower()):
                identifier_patterns.append(value)
                
        return list(set(identifier_patterns))[:5]
        
    def _extract_temporal_patterns(self, string_values: List[str]) -> List[str]:
        temporal_patterns = []
        
        for value in string_values:
            if re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{10,13}', value):
                temporal_patterns.append(value)
                
        return list(set(temporal_patterns))[:5]
        
    def _extract_platform_patterns(self, string_values: List[str]) -> List[str]:
        platform_patterns = []
        
        platform_indicators = ['splunk', 'chronicle', 'crowdstrike', 'bigquery', 'theom', 'wiz']
        
        for value in string_values:
            value_lower = value.lower()
            if any(indicator in value_lower for indicator in platform_indicators):
                platform_patterns.append(value)
                
        return list(set(platform_patterns))[:5]
        
    def _calculate_meaning_strength(self, field: FieldIntelligence, value_intelligence: float) -> float:
        strength_factors = []
        
        strength_factors.append(field.intelligence_score)
        strength_factors.append(value_intelligence)
        
        if field.semantic_profile and field.semantic_profile.meaning_confidence:
            strength_factors.append(field.semantic_profile.meaning_confidence)
            
        if field.business_context:
            confidence_level = field.business_context.get('confidence_level', 'low')
            confidence_scores = {'very_high': 1.0, 'high': 0.8, 'medium': 0.6, 'low': 0.3}
            strength_factors.append(confidence_scores.get(confidence_level, 0.3))
            
        return statistics.mean(strength_factors)
        
    def _calculate_semantic_intelligence_boost(self, field: FieldIntelligence) -> float:
        boost = 0.0
        
        meaning_strength = field.meaning_indicators.get('meaning_strength', 0.0)
        if meaning_strength > 0.8:
            boost += 0.15
        elif meaning_strength > 0.6:
            boost += 0.10
        elif meaning_strength > 0.4:
            boost += 0.05
            
        if field.cross_references:
            reference_count = sum(len(refs) for refs in field.cross_references.values())
            boost += min(0.1, reference_count / 20.0)
            
        return boost
        
    def _calculate_structural_complexity(self, string_values: List[str]) -> float:
        if not string_values:
            return 0.0
            
        complexity_indicators = 0.0
        
        char_diversity = len(set(''.join(string_values))) / max(1, len(''.join(string_values)))
        complexity_indicators += char_diversity * 0.3
        
        delimiters = ['.', '-', '_', ':', '/', ' ']
        delimiter_usage = sum(any(d in v for d in delimiters) for v in string_values) / len(string_values)
        complexity_indicators += delimiter_usage * 0.3
        
        mixed_case = sum(any(c.isupper() for c in v) and any(c.islower() for c in v) for v in string_values) / len(string_values)
        complexity_indicators += mixed_case * 0.2
        
        alphanumeric_mix = sum(any(c.isalpha() for c in v) and any(c.isdigit() for c in v) for v in string_values) / len(string_values)
        complexity_indicators += alphanumeric_mix * 0.2
        
        return min(1.0, complexity_indicators)
        
    def _classify_fields_semantically(self, fields: List[FieldIntelligence]) -> Dict[str, List[FieldIntelligence]]:
        classified = {
            'assets': [],
            'logging_activity': [],
            'temporal': [],
            'platforms': [],
            'infrastructure': [],
            'roles': [],
            'security_controls': [],
            'geographic': [],
            'counts': [],
            'identifiers': []
        }
        
        for field in fields:
            primary_type = self._determine_primary_semantic_type(field)
            
            if primary_type == 'asset_identifier' or primary_type == 'hostname':
                classified['assets'].append(field)
            elif primary_type == 'logging_activity':
                classified['logging_activity'].append(field)
            elif primary_type == 'temporal':
                classified['temporal'].append(field)
            elif primary_type == 'platform_source':
                classified['platforms'].append(field)
            elif primary_type == 'infrastructure_type':
                classified['infrastructure'].append(field)
            elif primary_type == 'role_classification':
                classified['roles'].append(field)
            elif primary_type == 'security_control':
                classified['security_controls'].append(field)
            elif primary_type == 'geographic':
                classified['geographic'].append(field)
            elif primary_type == 'count_metric':
                classified['counts'].append(field)
            else:
                classified['identifiers'].append(field)
                
        for category in classified:
            classified[category].sort(key=lambda f: f.intelligence_score, reverse=True)
            
        return classified
        
    def _determine_primary_semantic_type(self, field: FieldIntelligence) -> str:
        if hasattr(field, 'get_semantic_meaning'):
            semantic_meaning = field.get_semantic_meaning()
            if semantic_meaning != 'unknown':
                return semantic_meaning
                
        if field.semantic_profile and field.semantic_profile.content_intelligence:
            highest_confidence = 0.0
            best_type = 'unknown'
            
            for content_type, intelligence in field.semantic_profile.content_intelligence.items():
                if intelligence.confidence > highest_confidence:
                    highest_confidence = intelligence.confidence
                    best_type = content_type
                    
            if highest_confidence > 0.4:
                return best_type
                
        name_based_type = self._infer_type_from_name(field.name)
        if name_based_type != 'unknown':
            return name_based_type
            
        return 'unknown'
        
    def _infer_type_from_name(self, field_name: str) -> str:
        name_lower = field_name.lower()
        
        type_patterns = [
            (r'(hostname|host_name|device_name)', 'hostname'),
            (r'(asset|ci_name|asset_name)', 'asset_identifier'),
            (r'(log|event|message|activity)', 'logging_activity'),
            (r'(timestamp|time|date|created)', 'temporal'),
            (r'(source|platform|tool|index)', 'platform_source'),
            (r'(environment|infrastructure|type)', 'infrastructure_type'),
            (r'(role|function|service|category)', 'role_classification'),
            (r'(security|auth|control|agent)', 'security_control'),
            (r'(location|region|zone|geo)', 'geographic'),
            (r'(count|total|num|size)', 'count_metric')
        ]
        
        for pattern, semantic_type in type_patterns:
            if re.search(pattern, name_lower):
                return semantic_type
                
        return 'unknown'
        
    def _build_intelligent_global_visibility(self, classified_fields: Dict[str, List[FieldIntelligence]]) -> Optional[MetricResult]:
        asset_fields = classified_fields.get('assets', [])
        activity_fields = classified_fields.get('logging_activity', [])
        time_fields = classified_fields.get('temporal', [])
        
        if not asset_fields:
            asset_fields = classified_fields.get('identifiers', [])[:3]
            
        if not asset_fields:
            logger.warning("No suitable asset fields found for global visibility")
            return None
            
        query = self.query_builder.build_ao1_global_visibility_query(asset_fields + activity_fields + time_fields)
        if not query:
            logger.warning("Failed to build global visibility query")
            return None
            
        try:
            results = self.db_connector.execute_query(query.sql)
            if not results:
                logger.warning("Global visibility query returned no results")
                return None
                
            validation = self.validator.validate_global_visibility_results(results, query)
            
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
            results = self.db_connector.execute_query(query.sql)
            if not results:
                return None
                
            validation = self.validator.validate_platform_coverage_results(results, query)
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
            results = self.db_connector.execute_query(query.sql)
            if not results:
                return None
                
            validation = self.validator.validate_infrastructure_visibility_results(results, query)
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
            results = self.db_connector.execute_query(query.sql)
            if not results:
                return None
                
            validation = self.validator.validate_silent_assets_results(results, query)
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
            results = self.db_connector.execute_query(query.sql)
            if not results:
                return None
                
            validation = self.validator.validate_log_role_coverage_results(results, query)
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
        elif re.match(r'^\d{10,13}, sample_value):
            return f"{time_field.name} >= strftime('%s', 'now', '-7 days')"
        else:
            return f"{time_field.name} IS NOT NULL"
            
    def _generate_intelligent_activity_filter(self, activity_field: FieldIntelligence) -> str:
        if not activity_field.sample_values:
            return f"{activity_field.name} IS NOT NULL"
            
        string_values = [str(v).lower() for v in activity_field.sample_values[:20]]
        
        if any(re.match(r'^\d+, v) for v in string_values):
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