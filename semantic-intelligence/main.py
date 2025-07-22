#!/usr/bin/env python3

import logging
import statistics
import re
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
            if re.match(r'^[a-z0-9\-]+(\.[a-z0-9\-\.]+)?$', value.lower()):
                hostname_patterns.append(value)
                
        return list(set(hostname_patterns))[:5]
        
    def _extract_identifier_patterns(self, string_values: List[str]) -> List[str]:
        identifier_patterns = []
        
        for value in string_values:
            if re.match(r'^[A-Z0-9\-_]{4,}$', value) or re.match(r'^[a-f0-9\-]{8,}$', value.lower()):
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