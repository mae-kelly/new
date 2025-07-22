#!/usr/bin/env python3

import statistics
import re
import hashlib
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import Counter, defaultdict
from models import SemanticProfile, FieldIntelligence, ContentIntelligence
from entropy_calculator import EntropyCalculator
from pattern_detector import PatternDetector
import json

class AdvancedSemanticAnalyzer:
    def __init__(self):
        self.entropy_calculator = EntropyCalculator()
        self.pattern_detector = PatternDetector()
        self.content_analyzers = {
            'asset_identifier': self._analyze_asset_content,
            'hostname': self._analyze_hostname_content,
            'logging_activity': self._analyze_logging_content,
            'platform_source': self._analyze_platform_content,
            'temporal': self._analyze_temporal_content,
            'infrastructure_type': self._analyze_infrastructure_content,
            'role_classification': self._analyze_role_content,
            'security_control': self._analyze_security_content,
            'geographic': self._analyze_geographic_content,
            'count_metric': self._analyze_count_content,
        }
        
    def analyze_field_deeply(self, field_name: str, table_name: str, data_type: str, 
                           sample_values: List[Any], cross_field_context: Dict = None) -> FieldIntelligence:
        
        if not sample_values:
            return FieldIntelligence(field_name, table_name, data_type)
            
        clean_values = [v for v in sample_values if v is not None]
        if not clean_values:
            return FieldIntelligence(field_name, table_name, data_type)
            
        profile = self._create_advanced_semantic_profile(field_name, clean_values, cross_field_context)
        
        intelligence_score = self._calculate_deep_intelligence_score(profile, field_name, clean_values)
        confidence_level = self._calculate_meaning_confidence(profile, clean_values)
        business_context = self._derive_intelligent_business_context(field_name, profile, clean_values)
        value_patterns = self._extract_value_patterns(clean_values)
        meaning_indicators = self._calculate_meaning_indicators(field_name, clean_values, profile)
        
        field_intelligence = FieldIntelligence(
            name=field_name,
            table=table_name,
            data_type=data_type,
            semantic_profile=profile,
            intelligence_score=intelligence_score,
            confidence_level=confidence_level,
            business_context=business_context,
            sample_values=clean_values[:100],
            value_patterns=value_patterns,
            meaning_indicators=meaning_indicators
        )
        
        return field_intelligence
        
    def _create_advanced_semantic_profile(self, field_name: str, values: List[Any], 
                                        cross_field_context: Dict = None) -> SemanticProfile:
        string_values = [str(v) for v in values]
        
        profile = SemanticProfile(field_name)
        
        profile.entropy_metrics = self.entropy_calculator.calculate_all_entropy_metrics(string_values)
        
        profile.pattern_features = {
            'format_patterns': self.pattern_detector.detect_format_patterns(values),
            'semantic_patterns': self.pattern_detector.detect_semantic_patterns(field_name, values),
            'structural_patterns': self.pattern_detector.detect_structural_patterns(values),
            'temporal_patterns': self.pattern_detector.detect_temporal_patterns(values),
            'regex_matches': list(self.pattern_detector.detect_regex_patterns(values).values())
        }
        
        profile.statistical_properties = self._calculate_advanced_statistics(values)
        profile.behavioral_indicators = self._analyze_advanced_behavioral_patterns(values)
        profile.value_analysis = self._analyze_value_semantics(values, field_name)
        
        profile.content_intelligence = self._perform_content_analysis(field_name, values)
        
        if cross_field_context:
            profile.cross_field_patterns = self._analyze_cross_field_patterns(values, cross_field_context)
        
        profile.semantic_density = self._calculate_semantic_density(profile)
        profile.complexity_score = self._calculate_complexity_score(profile)
        profile.meaning_confidence = self._calculate_meaning_confidence_score(profile)
        
        return profile
        
    def _perform_content_analysis(self, field_name: str, values: List[Any]) -> Dict[str, ContentIntelligence]:
        content_intel = {}
        
        for content_type, analyzer in self.content_analyzers.items():
            intelligence = analyzer(field_name, values)
            if intelligence.confidence > 0.1:
                content_intel[content_type] = intelligence
                
        return content_intel
        
    def _analyze_asset_content(self, field_name: str, values: List[Any]) -> ContentIntelligence:
        evidence = []
        pattern_matches = {}
        confidence = 0.0
        
        name_lower = field_name.lower()
        string_values = [str(v).lower() for v in values[:50]]
        
        asset_name_patterns = ['hostname', 'asset', 'ci_name', 'device', 'server', 'computer', 'host', 'endpoint', 'node', 'machine']
        for pattern in asset_name_patterns:
            if pattern in name_lower:
                confidence += 0.15
                evidence.append(f"field_name_contains_{pattern}")
                pattern_matches[pattern] = 1.0
        
        asset_value_patterns = [
            (r'^[a-z0-9\-]+\.(domain\.com|local|corp)$', 'fqdn_pattern', 0.4),
            (r'^[A-Z]{2,4}[0-9]{3,6}$', 'asset_code_pattern', 0.3),
            (r'^(srv|web|db|app|dc|ad)[0-9]{1,3}$', 'server_naming_pattern', 0.35),
            (r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', 'ip_address_pattern', 0.25),
            (r'^[a-z0-9\-]{4,20}$', 'hostname_pattern', 0.2)
        ]
        
        for regex, pattern_name, weight in asset_value_patterns:
            matches = sum(1 for v in string_values if re.match(regex, v))
            match_ratio = matches / len(string_values) if string_values else 0
            if match_ratio > 0.1:
                confidence += weight * match_ratio
                evidence.append(f"{pattern_name}_match_ratio_{match_ratio:.2f}")
                pattern_matches[pattern_name] = match_ratio
        
        uniqueness_ratio = len(set(string_values)) / len(string_values) if string_values else 0
        if uniqueness_ratio > 0.8:
            confidence += 0.2
            evidence.append(f"high_uniqueness_{uniqueness_ratio:.2f}")
        
        semantic_class = "asset_identifier" if confidence > 0.3 else "possible_asset_identifier"
        
        return ContentIntelligence(
            content_type="asset_identifier",
            confidence=min(1.0, confidence),
            evidence=evidence,
            pattern_matches=pattern_matches,
            semantic_classification=semantic_class
        )
        
    def _analyze_hostname_content(self, field_name: str, values: List[Any]) -> ContentIntelligence:
        evidence = []
        pattern_matches = {}
        confidence = 0.0
        
        name_lower = field_name.lower()
        string_values = [str(v).lower() for v in values[:50]]
        
        if 'host' in name_lower:
            confidence += 0.3
            evidence.append("field_contains_host")
            
        hostname_patterns = [
            (r'^[a-z0-9\-\.]+$', 'valid_hostname_chars', 0.2),
            (r'^[a-z0-9\-]+\.[a-z0-9\-\.]+$', 'fqdn_format', 0.4),
            (r'^[a-z0-9\-]{1,63}$', 'simple_hostname', 0.25),
            (r'(srv|web|db|app|mail|dns|dc|ad)', 'service_prefix', 0.15)
        ]
        
        for regex, pattern_name, weight in hostname_patterns:
            matches = sum(1 for v in string_values if re.search(regex, v))
            match_ratio = matches / len(string_values) if string_values else 0
            if match_ratio > 0.1:
                confidence += weight * match_ratio
                evidence.append(f"{pattern_name}_ratio_{match_ratio:.2f}")
                pattern_matches[pattern_name] = match_ratio
        
        avg_length = statistics.mean([len(str(v)) for v in values]) if values else 0
        if 4 <= avg_length <= 64:
            confidence += 0.1
            evidence.append(f"hostname_length_range_{avg_length:.1f}")
        
        semantic_class = "hostname" if confidence > 0.4 else "possible_hostname"
        
        return ContentIntelligence(
            content_type="hostname",
            confidence=min(1.0, confidence),
            evidence=evidence,
            pattern_matches=pattern_matches,
            semantic_classification=semantic_class
        )
        
    def _analyze_logging_content(self, field_name: str, values: List[Any]) -> ContentIntelligence:
        evidence = []
        pattern_matches = {}
        confidence = 0.0
        
        name_lower = field_name.lower()
        string_values = [str(v).lower() for v in values[:50]]
        
        logging_name_indicators = ['log', 'event', 'message', 'entry', 'record', 'activity', 'count', 'data']
        for indicator in logging_name_indicators:
            if indicator in name_lower:
                confidence += 0.15
                evidence.append(f"name_contains_{indicator}")
                
        logging_value_patterns = [
            (r'(error|warn|info|debug|trace|fatal)', 'log_level', 0.3),
            (r'(failed|success|complete|start|end)', 'status_indicator', 0.2),
            (r'\d{4}-\d{2}-\d{2}', 'contains_date', 0.15),
            (r'(login|logout|access|denied|auth)', 'auth_activity', 0.25),
            (r'^\d+$', 'numeric_count', 0.2)
        ]
        
        for regex, pattern_name, weight in logging_value_patterns:
            matches = sum(1 for v in string_values if re.search(regex, v))
            match_ratio = matches / len(string_values) if string_values else 0
            if match_ratio > 0.05:
                confidence += weight * match_ratio
                evidence.append(f"{pattern_name}_ratio_{match_ratio:.2f}")
                pattern_matches[pattern_name] = match_ratio
        
        variability = len(set(string_values)) / len(string_values) if string_values else 0
        if variability > 0.3:
            confidence += 0.2
            evidence.append(f"high_variability_{variability:.2f}")
            
        semantic_class = "logging_activity" if confidence > 0.3 else "possible_logging"
        
        return ContentIntelligence(
            content_type="logging_activity",
            confidence=min(1.0, confidence),
            evidence=evidence,
            pattern_matches=pattern_matches,
            semantic_classification=semantic_class
        )
        
    def _analyze_platform_content(self, field_name: str, values: List[Any]) -> ContentIntelligence:
        evidence = []
        pattern_matches = {}
        confidence = 0.0
        
        name_lower = field_name.lower()
        string_values = [str(v).lower() for v in values[:50]]
        
        platform_name_indicators = ['source', 'platform', 'tool', 'system', 'index', 'sourcetype']
        for indicator in platform_name_indicators:
            if indicator in name_lower:
                confidence += 0.2
                evidence.append(f"name_contains_{indicator}")
                
        known_platforms = ['splunk', 'chronicle', 'bigquery', 'crowdstrike', 'falcon', 'theom', 'wiz', 'axonius', 'tanium']
        platform_mentions = 0
        for platform in known_platforms:
            mentions = sum(1 for v in string_values if platform in v)
            if mentions > 0:
                platform_mentions += mentions
                pattern_matches[platform] = mentions / len(string_values)
                
        if platform_mentions > 0:
            confidence += min(0.5, platform_mentions / len(string_values) * 2)
            evidence.append(f"platform_mentions_{platform_mentions}")
            
        semantic_class = "platform_source" if confidence > 0.3 else "possible_platform"
        
        return ContentIntelligence(
            content_type="platform_source",
            confidence=min(1.0, confidence),
            evidence=evidence,
            pattern_matches=pattern_matches,
            semantic_classification=semantic_class
        )
        
    def _analyze_temporal_content(self, field_name: str, values: List[Any]) -> ContentIntelligence:
        evidence = []
        pattern_matches = {}
        confidence = 0.0
        
        name_lower = field_name.lower()
        string_values = [str(v) for v in values[:50]]
        
        temporal_name_indicators = ['time', 'date', 'timestamp', 'created', 'modified', 'occurred', 'when', '_at']
        for indicator in temporal_name_indicators:
            if indicator in name_lower:
                confidence += 0.25
                evidence.append(f"name_contains_{indicator}")
                
        temporal_patterns = [
            (r'\d{4}-\d{2}-\d{2}', 'iso_date', 0.4),
            (r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', 'iso_datetime', 0.5),
            (r'^\d{10,13}$', 'unix_timestamp', 0.4),
            (r'\d{2}/\d{2}/\d{4}', 'us_date', 0.3),
            (r'\d{2}:\d{2}:\d{2}', 'time_format', 0.3)
        ]
        
        for regex, pattern_name, weight in temporal_patterns:
            matches = sum(1 for v in string_values if re.search(regex, v))
            match_ratio = matches / len(string_values) if string_values else 0
            if match_ratio > 0.1:
                confidence += weight * match_ratio
                evidence.append(f"{pattern_name}_ratio_{match_ratio:.2f}")
                pattern_matches[pattern_name] = match_ratio
                
        semantic_class = "temporal" if confidence > 0.4 else "possible_temporal"
        
        return ContentIntelligence(
            content_type="temporal",
            confidence=min(1.0, confidence),
            evidence=evidence,
            pattern_matches=pattern_matches,
            semantic_classification=semantic_class
        )
        
    def _analyze_infrastructure_content(self, field_name: str, values: List[Any]) -> ContentIntelligence:
        evidence = []
        pattern_matches = {}
        confidence = 0.0
        
        name_lower = field_name.lower()
        string_values = [str(v).lower() for v in values[:50]]
        
        infra_name_indicators = ['environment', 'env', 'type', 'category', 'tier', 'deployment', 'infra']
        for indicator in infra_name_indicators:
            if indicator in name_lower:
                confidence += 0.2
                evidence.append(f"name_contains_{indicator}")
                
        infra_terms = ['cloud', 'aws', 'azure', 'gcp', 'onprem', 'premise', 'datacenter', 'saas', 'api', 'hybrid']
        infra_mentions = 0
        for term in infra_terms:
            mentions = sum(1 for v in string_values if term in v)
            if mentions > 0:
                infra_mentions += mentions
                pattern_matches[term] = mentions / len(string_values)
                
        if infra_mentions > 0:
            confidence += min(0.4, infra_mentions / len(string_values) * 2)
            evidence.append(f"infra_terms_{infra_mentions}")
            
        semantic_class = "infrastructure_type" if confidence > 0.3 else "possible_infrastructure"
        
        return ContentIntelligence(
            content_type="infrastructure_type",
            confidence=min(1.0, confidence),
            evidence=evidence,
            pattern_matches=pattern_matches,
            semantic_classification=semantic_class
        )
        
    def _analyze_role_content(self, field_name: str, values: List[Any]) -> ContentIntelligence:
        evidence = []
        pattern_matches = {}
        confidence = 0.0
        
        name_lower = field_name.lower()
        string_values = [str(v).lower() for v in values[:50]]
        
        role_name_indicators = ['role', 'function', 'type', 'category', 'service', 'class']
        for indicator in role_name_indicators:
            if indicator in name_lower:
                confidence += 0.2
                evidence.append(f"name_contains_{indicator}")
                
        role_terms = ['network', 'endpoint', 'cloud', 'application', 'identity', 'server', 'workstation', 'firewall', 'router']
        role_mentions = 0
        for term in role_terms:
            mentions = sum(1 for v in string_values if term in v)
            if mentions > 0:
                role_mentions += mentions
                pattern_matches[term] = mentions / len(string_values)
                
        if role_mentions > 0:
            confidence += min(0.4, role_mentions / len(string_values) * 2)
            evidence.append(f"role_terms_{role_mentions}")
            
        semantic_class = "role_classification" if confidence > 0.3 else "possible_role"
        
        return ContentIntelligence(
            content_type="role_classification",
            confidence=min(1.0, confidence),
            evidence=evidence,
            pattern_matches=pattern_matches,
            semantic_classification=semantic_class
        )
        
    def _analyze_security_content(self, field_name: str, values: List[Any]) -> ContentIntelligence:
        evidence = []
        pattern_matches = {}
        confidence = 0.0
        
        name_lower = field_name.lower()
        string_values = [str(v).lower() for v in values[:50]]
        
        security_name_indicators = ['security', 'auth', 'permission', 'access', 'control', 'agent', 'edr', 'dlp']
        for indicator in security_name_indicators:
            if indicator in name_lower:
                confidence += 0.2
                evidence.append(f"name_contains_{indicator}")
                
        security_terms = ['crowdstrike', 'tanium', 'edr', 'endpoint', 'antivirus', 'firewall', 'blocked', 'allowed', 'denied']
        security_mentions = 0
        for term in security_terms:
            mentions = sum(1 for v in string_values if term in v)
            if mentions > 0:
                security_mentions += mentions
                pattern_matches[term] = mentions / len(string_values)
                
        if security_mentions > 0:
            confidence += min(0.4, security_mentions / len(string_values) * 2)
            evidence.append(f"security_terms_{security_mentions}")
            
        semantic_class = "security_control" if confidence > 0.3 else "possible_security"
        
        return ContentIntelligence(
            content_type="security_control",
            confidence=min(1.0, confidence),
            evidence=evidence,
            pattern_matches=pattern_matches,
            semantic_classification=semantic_class
        )
        
    def _analyze_geographic_content(self, field_name: str, values: List[Any]) -> ContentIntelligence:
        evidence = []
        pattern_matches = {}
        confidence = 0.0
        
        name_lower = field_name.lower()
        string_values = [str(v).lower() for v in values[:50]]
        
        geo_name_indicators = ['location', 'region', 'country', 'city', 'site', 'zone', 'geo', 'datacenter']
        for indicator in geo_name_indicators:
            if indicator in name_lower:
                confidence += 0.25
                evidence.append(f"name_contains_{indicator}")
                
        geo_terms = ['us-east', 'us-west', 'eu-west', 'asia', 'pacific', 'america', 'europe', 'north', 'south', 'east', 'west']
        geo_mentions = 0
        for term in geo_terms:
            mentions = sum(1 for v in string_values if term in v)
            if mentions > 0:
                geo_mentions += mentions
                pattern_matches[term] = mentions / len(string_values)
                
        if geo_mentions > 0:
            confidence += min(0.4, geo_mentions / len(string_values) * 2)
            evidence.append(f"geo_terms_{geo_mentions}")
            
        semantic_class = "geographic" if confidence > 0.3 else "possible_geographic"
        
        return ContentIntelligence(
            content_type="geographic",
            confidence=min(1.0, confidence),
            evidence=evidence,
            pattern_matches=pattern_matches,
            semantic_classification=semantic_class
        )
        
    def _analyze_count_content(self, field_name: str, values: List[Any]) -> ContentIntelligence:
        evidence = []
        pattern_matches = {}
        confidence = 0.0
        
        name_lower = field_name.lower()
        
        count_indicators = ['count', 'total', 'num', 'quantity', 'amount', 'size', 'length']
        for indicator in count_indicators:
            if indicator in name_lower:
                confidence += 0.2
                evidence.append(f"name_contains_{indicator}")
                
        try:
            numeric_values = [float(v) for v in values if str(v).replace('.', '').replace('-', '').isdigit()]
            if len(numeric_values) / len(values) > 0.8:
                confidence += 0.3
                evidence.append("high_numeric_ratio")
                
                if all(v >= 0 for v in numeric_values):
                    confidence += 0.2
                    evidence.append("non_negative_values")
                    
                if all(v == int(v) for v in numeric_values):
                    confidence += 0.2
                    evidence.append("integer_values")
                    
        except:
            pass
            
        semantic_class = "count_metric" if confidence > 0.4 else "possible_count"
        
        return ContentIntelligence(
            content_type="count_metric",
            confidence=min(1.0, confidence),
            evidence=evidence,
            pattern_matches=pattern_matches,
            semantic_classification=semantic_class
        )
        
    def _calculate_advanced_statistics(self, values: List[Any]) -> Dict[str, float]:
        properties = {}
        
        string_values = [str(v) for v in values]
        if string_values:
            lengths = [len(v) for v in string_values]
            properties.update({
                'avg_length': statistics.mean(lengths),
                'length_std': statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
                'min_length': min(lengths),
                'max_length': max(lengths),
                'length_range': max(lengths) - min(lengths),
                'length_coefficient_variation': statistics.stdev(lengths) / statistics.mean(lengths) if statistics.mean(lengths) > 0 and len(lengths) > 1 else 0
            })
            
            all_chars = ''.join(string_values)
            if all_chars:
                properties['character_diversity'] = len(set(all_chars)) / len(all_chars)
            else:
                properties['character_diversity'] = 0.0
                
        numeric_values = self._extract_numeric_values(values)
        if numeric_values:
            properties.update({
                'numeric_mean': statistics.mean(numeric_values),
                'numeric_median': statistics.median(numeric_values),
                'numeric_std': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
                'numeric_range': max(numeric_values) - min(numeric_values),
                'numeric_coefficient_variation': statistics.stdev(numeric_values) / statistics.mean(numeric_values) if statistics.mean(numeric_values) > 0 and len(numeric_values) > 1 else 0
            })
            
        from collections import Counter
        value_counts = Counter(string_values)
        properties.update({
            'unique_ratio': len(value_counts) / len(string_values),
            'singleton_ratio': sum(1 for count in value_counts.values() if count == 1) / len(value_counts) if value_counts else 0,
            'max_frequency': max(value_counts.values()) / len(string_values) if value_counts else 0,
            'entropy': -sum((count/len(string_values)) * statistics.log2(count/len(string_values)) for count in value_counts.values()) if value_counts else 0
        })
        
        return properties
        
    def _analyze_advanced_behavioral_patterns(self, values: List[Any]) -> Dict[str, float]:
        patterns = {}
        
        patterns['consistency'] = self._calculate_advanced_consistency(values)
        patterns['predictability'] = self._calculate_predictability_score(values)
        patterns['variability'] = self._calculate_variability_score(values)
        patterns['anomaly_rate'] = self._calculate_anomaly_rate(values)
        patterns['growth_trend'] = self._calculate_growth_trend(values)
        patterns['cyclical_pattern'] = self._detect_cyclical_patterns(values)
        patterns['distribution_skew'] = self._calculate_distribution_skew(values)
        
        return patterns
        
    def _analyze_value_semantics(self, values: List[Any], field_name: str) -> Dict[str, Any]:
        analysis = {}
        
        string_values = [str(v).lower() for v in values]
        
        analysis['contains_identifiers'] = self._detect_identifier_patterns(string_values)
        analysis['contains_natural_language'] = self._detect_natural_language(string_values)
        analysis['contains_codes'] = self._detect_code_patterns(string_values)
        analysis['contains_timestamps'] = self._detect_timestamp_patterns(string_values)
        analysis['contains_urls'] = self._detect_url_patterns(string_values)
        analysis['contains_filenames'] = self._detect_filename_patterns(string_values)
        analysis['semantic_clusters'] = self._find_semantic_clusters(string_values)
        analysis['value_relationships'] = self._analyze_value_relationships(string_values)
        
        return analysis
        
    def _analyze_cross_field_patterns(self, values: List[Any], context: Dict) -> Dict[str, float]:
        patterns = {}
        
        string_values = [str(v).lower() for v in values]
        
        for other_field, other_values in context.items():
            other_string_values = [str(v).lower() for v in other_values]
            
            overlap = len(set(string_values) & set(other_string_values))
            total_unique = len(set(string_values) | set(other_string_values))
            
            if total_unique > 0:
                patterns[f'overlap_with_{other_field}'] = overlap / total_unique
                
        return patterns
        
    def _calculate_deep_intelligence_score(self, profile: SemanticProfile, field_name: str, values: List[Any]) -> float:
        components = []
        
        components.append(profile.semantic_density * 0.3)
        components.append(profile.complexity_score * 0.2)
        components.append(profile.meaning_confidence * 0.25)
        
        if profile.content_intelligence:
            max_content_confidence = max(ci.confidence for ci in profile.content_intelligence.values())
            components.append(max_content_confidence * 0.2)
        
        if profile.pattern_features:
            max_pattern_scores = []
            for feature_list in profile.pattern_features.values():
                if feature_list:
                    max_pattern_scores.append(max(feature_list))
            if max_pattern_scores:
                pattern_component = statistics.mean(max_pattern_scores) * 0.05
                components.append(pattern_component)
                
        return sum(components)
        
    def _calculate_meaning_confidence(self, profile: SemanticProfile, values: List[Any]) -> float:
        confidence_factors = []
        
        sample_factor = min(1.0, len(values) / 1000.0)
        confidence_factors.append(sample_factor)
        
        if profile.content_intelligence:
            avg_content_confidence = statistics.mean([ci.confidence for ci in profile.content_intelligence.values()])
            confidence_factors.append(avg_content_confidence)
            
        consistency = profile.behavioral_indicators.get('consistency', 0.0)
        confidence_factors.append(consistency)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.0
        
    def _derive_intelligent_business_context(self, field_name: str, profile: SemanticProfile, values: List[Any]) -> Dict[str, Any]:
        context = {
            'primary_semantic_type': 'unknown',
            'confidence_level': 'low',
            'business_importance': 'medium',
            'data_sensitivity': 'standard',
            'operational_role': 'data_field',
            'ao1_relevance': 'none'
        }
        
        if profile.content_intelligence:
            highest_confidence = 0.0
            best_type = 'unknown'
            for content_type, intelligence in profile.content_intelligence.items():
                if intelligence.confidence > highest_confidence:
                    highest_confidence = intelligence.confidence
                    best_type = content_type
                    
            context['primary_semantic_type'] = best_type
            
            if highest_confidence > 0.8:
                context['confidence_level'] = 'very_high'
            elif highest_confidence > 0.6:
                context['confidence_level'] = 'high'
            elif highest_confidence > 0.4:
                context['confidence_level'] = 'medium'
            else:
                context['confidence_level'] = 'low'
                
            if best_type in ['asset_identifier', 'hostname', 'logging_activity', 'platform_source', 'temporal']:
                context['ao1_relevance'] = 'high'
                context['business_importance'] = 'high'
            elif best_type in ['infrastructure_type', 'role_classification', 'security_control']:
                context['ao1_relevance'] = 'medium'
                context['business_importance'] = 'high'
            elif best_type in ['geographic', 'count_metric']:
                context['ao1_relevance'] = 'low'
                
        field_lower = field_name.lower()
        if any(keyword in field_lower for keyword in ['id', 'key', 'primary']):
            context['business_importance'] = 'critical'
        elif any(keyword in field_lower for keyword in ['name', 'email', 'user', 'password']):
            context['business_importance'] = 'high'
            context['data_sensitivity'] = 'sensitive'
        elif any(keyword in field_lower for keyword in ['temp', 'cache', 'debug']):
            context['business_importance'] = 'low'
            
        return context
        
    def _extract_value_patterns(self, values: List[Any]) -> Dict[str, Any]:
        patterns = {}
        
        string_values = [str(v) for v in values]
        
        patterns['common_prefixes'] = self._find_common_prefixes(string_values)
        patterns['common_suffixes'] = self._find_common_suffixes(string_values)
        patterns['format_templates'] = self._extract_format_templates(string_values)
        patterns['delimiter_usage'] = self._analyze_delimiter_usage(string_values)
        patterns['case_patterns'] = self._analyze_case_patterns(string_values)
        
        return patterns
        
    def _calculate_meaning_indicators(self, field_name: str, values: List[Any], profile: SemanticProfile) -> Dict[str, float]:
        indicators = {}
        
        name_semantic_weight = self._calculate_name_semantic_weight(field_name)
        value_semantic_weight = self._calculate_value_semantic_weight(values)
        pattern_coherence = self._calculate_pattern_coherence(profile)
        content_alignment = self._calculate_content_alignment(field_name, values, profile)
        
        indicators['name_semantic_weight'] = name_semantic_weight
        indicators['value_semantic_weight'] = value_semantic_weight
        indicators['pattern_coherence'] = pattern_coherence
        indicators['content_alignment'] = content_alignment
        indicators['overall_meaning_strength'] = statistics.mean([name_semantic_weight, value_semantic_weight, pattern_coherence, content_alignment])
        
        return indicators
        
    def _calculate_advanced_consistency(self, values: List[Any]) -> float:
        if not values:
            return 0.0
            
        string_values = [str(v) for v in values]
        
        format_consistency = self._calculate_format_consistency(string_values)
        length_consistency = self._calculate_length_consistency(string_values) 
        character_consistency = self._calculate_character_consistency(string_values)
        
        return statistics.mean([format_consistency, length_consistency, character_consistency])
        
    def _extract_numeric_values(self, values: List[Any]) -> List[float]:
        numeric_values = []
        for v in values:
            try:
                if isinstance(v, (int, float)):
                    numeric_values.append(float(v))
                else:
                    numeric_values.append(float(str(v)))
            except:
                pass
        return numeric_values
        
    def _calculate_semantic_density(self, profile: SemanticProfile) -> float:
        components = []
        
        if profile.entropy_metrics:
            entropy_avg = statistics.mean(profile.entropy_metrics.values())
            components.append(min(1.0, entropy_avg))
            
        if profile.pattern_features:
            pattern_scores = []
            for feature_list in profile.pattern_features.values():
                if feature_list:
                    pattern_scores.extend(feature_list)
            if pattern_scores:
                pattern_avg = statistics.mean(pattern_scores)
                components.append(pattern_avg)
                
        if profile.behavioral_indicators:
            behavior_avg = statistics.mean(abs(v) for v in profile.behavioral_indicators.values())
            components.append(behavior_avg)
            
        if profile.content_intelligence:
            content_avg = statistics.mean([ci.confidence for ci in profile.content_intelligence.values()])
            components.append(content_avg)
            
        return statistics.mean(components) if components else 0.0
        
    def _calculate_complexity_score(self, profile: SemanticProfile) -> float:
        complexity_factors = []
        
        shannon_entropy = profile.entropy_metrics.get('shannon', 0.0)
        complexity_factors.append(min(1.0, shannon_entropy / 10.0))
        
        if profile.pattern_features:
            pattern_diversity = len([score for scores in profile.pattern_features.values() 
                                   for score in scores if score > 0.1])
            complexity_factors.append(min(1.0, pattern_diversity / 30.0))
            
        if profile.statistical_properties:
            coefficient_variations = [v for k, v in profile.statistical_properties.items() if 'coefficient' in k.lower()]
            if coefficient_variations:
                avg_cv = statistics.mean(coefficient_variations)
                complexity_factors.append(min(1.0, avg_cv))
                
        if profile.content_intelligence:
            intel_diversity = len(profile.content_intelligence)
            complexity_factors.append(min(1.0, intel_diversity / 5.0))
            
        return statistics.mean(complexity_factors) if complexity_factors else 0.0
        
    def _calculate_meaning_confidence_score(self, profile: SemanticProfile) -> float:
        confidence_components = []
        
        if profile.content_intelligence:
            max_confidence = max(ci.confidence for ci in profile.content_intelligence.values())
            confidence_components.append(max_confidence)
            
        if profile.behavioral_indicators:
            consistency = profile.behavioral_indicators.get('consistency', 0.0)
            confidence_components.append(consistency)
            
        if profile.pattern_features:
            pattern_strength = max([max(scores) if scores else 0.0 for scores in profile.pattern_features.values()])
            confidence_components.append(pattern_strength)
            
        return statistics.mean(confidence_components) if confidence_components else 0.0
        
    def _calculate_predictability_score(self, values: List[Any]) -> float:
        numeric_values = self._extract_numeric_values(values)
        if len(numeric_values) < 3:
            return 0.0
            
        try:
            differences = [numeric_values[i+1] - numeric_values[i] for i in range(len(numeric_values)-1)]
            if not differences:
                return 0.0
                
            diff_std = statistics.stdev(differences) if len(differences) > 1 else 0
            diff_mean = abs(statistics.mean(differences))
            
            if diff_mean == 0:
                return 1.0 if diff_std == 0 else 0.0
            else:
                return max(0.0, 1.0 - (diff_std / diff_mean))
        except:
            return 0.0
            
    def _calculate_variability_score(self, values: List[Any]) -> float:
        if not values:
            return 0.0
        string_values = [str(v) for v in values]
        unique_count = len(set(string_values))
        return unique_count / len(string_values)
        
    def _calculate_anomaly_rate(self, values: List[Any]) -> float:
        numeric_values = self._extract_numeric_values(values)
        if len(numeric_values) < 3:
            return 0.0
            
        try:
            mean_val = statistics.mean(numeric_values)
            std_val = statistics.stdev(numeric_values)
            
            if std_val == 0:
                return 0.0
                
            anomalies = sum(1 for v in numeric_values if abs(v - mean_val) > 2 * std_val)
            return anomalies / len(numeric_values)
        except:
            return 0.0
            
    def _calculate_growth_trend(self, values: List[Any]) -> float:
        numeric_values = self._extract_numeric_values(values)
        if len(numeric_values) < 3:
            return 0.0
            
        try:
            differences = [numeric_values[i+1] - numeric_values[i] for i in range(len(numeric_values)-1)]
            positive_diffs = sum(1 for d in differences if d > 0)
            negative_diffs = sum(1 for d in differences if d < 0)
            
            if positive_diffs > len(differences) * 0.8:
                return 1.0
            elif negative_diffs > len(differences) * 0.8:
                return -1.0
            else:
                return (positive_diffs - negative_diffs) / len(differences)
        except:
            return 0.0
            
    def _detect_cyclical_patterns(self, values: List[Any]) -> float:
        numeric_values = self._extract_numeric_values(values)
        if len(numeric_values) < 6:
            return 0.0
            
        try:
            for period in range(2, min(len(numeric_values) // 2, 10)):
                cycles = []
                for i in range(period, len(numeric_values)):
                    cycles.append(numeric_values[i] - numeric_values[i - period])
                    
                if cycles and statistics.stdev(cycles) < statistics.mean(abs(c) for c in cycles) * 0.1:
                    return min(1.0, period / 10.0)
                    
            return 0.0
        except:
            return 0.0
            
    def _calculate_distribution_skew(self, values: List[Any]) -> float:
        numeric_values = self._extract_numeric_values(values)
        if len(numeric_values) < 3:
            return 0.0
            
        try:
            mean_val = statistics.mean(numeric_values)
            median_val = statistics.median(numeric_values)
            std_val = statistics.stdev(numeric_values)
            
            if std_val == 0:
                return 0.0
                
            skewness = (mean_val - median_val) / std_val
            return max(-1.0, min(1.0, skewness))
        except:
            return 0.0
            
    def _detect_identifier_patterns(self, string_values: List[str]) -> float:
        identifier_patterns = [
            r'^[a-z0-9\-_]{8,}$',
            r'^[A-Z0-9]{4,}-[A-Z0-9]{4,}$',
            r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
            r'^\d{6,}$'
        ]
        
        matches = 0
        for pattern in identifier_patterns:
            matches += sum(1 for v in string_values if re.match(pattern, v))
            
        return min(1.0, matches / len(string_values)) if string_values else 0.0
        
    def _detect_natural_language(self, string_values: List[str]) -> float:
        language_indicators = [
            r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',
            r'[.!?]',
            r'\b[a-z]{3,}\s+[a-z]{3,}\b'
        ]
        
        matches = 0
        for pattern in language_indicators:
            matches += sum(1 for v in string_values if re.search(pattern, v))
            
        return min(1.0, matches / len(string_values)) if string_values else 0.0
        
    def _detect_code_patterns(self, string_values: List[str]) -> float:
        code_patterns = [
            r'^[A-Z]{2,3}\d{3,6}$',
            r'^[A-Z]{1,3}-\d{3,6}$',
            r'^\d{3}-\d{3}-\d{4}$',
            r'^[A-Z0-9]{6,12}$'
        ]
        
        matches = 0
        for pattern in code_patterns:
            matches += sum(1 for v in string_values if re.match(pattern, v))
            
        return min(1.0, matches / len(string_values)) if string_values else 0.0
        
    def _detect_timestamp_patterns(self, string_values: List[str]) -> float:
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{4}/\d{2}/\d{2}',
            r'\d{2}:\d{2}:\d{2}',
            r'^\d{10,13}$'
        ]
        
        matches = 0
        for pattern in timestamp_patterns:
            matches += sum(1 for v in string_values if re.search(pattern, v))
            
        return min(1.0, matches / len(string_values)) if string_values else 0.0
        
    def _detect_url_patterns(self, string_values: List[str]) -> float:
        url_pattern = r'https?://[^\s]+|www\.[^\s]+'
        matches = sum(1 for v in string_values if re.search(url_pattern, v))
        return min(1.0, matches / len(string_values)) if string_values else 0.0
        
    def _detect_filename_patterns(self, string_values: List[str]) -> float:
        filename_pattern = r'^[^/\\]+\.[a-z]{2,4}$'
        matches = sum(1 for v in string_values if re.match(filename_pattern, v))
        return min(1.0, matches / len(string_values)) if string_values else 0.0
        
    def _find_semantic_clusters(self, string_values: List[str]) -> List[List[str]]:
        clusters = []
        processed = set()
        
        for value in string_values:
            if value in processed:
                continue
                
            cluster = [value]
            processed.add(value)
            
            for other_value in string_values:
                if other_value != value and other_value not in processed:
                    similarity = self._calculate_string_similarity(value, other_value)
                    if similarity > 0.7:
                        cluster.append(other_value)
                        processed.add(other_value)
                        
            if len(cluster) > 1:
                clusters.append(cluster)
                
        return clusters[:5]
        
    def _analyze_value_relationships(self, string_values: List[str]) -> Dict[str, float]:
        relationships = {}
        
        if len(string_values) < 2:
            return relationships
            
        total_pairs = 0
        similar_pairs = 0
        
        for i, val1 in enumerate(string_values[:20]):
            for val2 in string_values[i+1:20]:
                total_pairs += 1
                similarity = self._calculate_string_similarity(val1, val2)
                if similarity > 0.3:
                    similar_pairs += 1
                    
        relationships['similarity_ratio'] = similar_pairs / total_pairs if total_pairs > 0 else 0.0
        
        return relationships
        
    def _find_common_prefixes(self, string_values: List[str]) -> List[str]:
        if not string_values:
            return []
            
        prefixes = Counter()
        for value in string_values:
            for i in range(1, min(len(value) + 1, 6)):
                prefixes[value[:i]] += 1
                
        common_prefixes = [prefix for prefix, count in prefixes.items() 
                          if count >= max(2, len(string_values) * 0.1)]
        return common_prefixes[:5]
        
    def _find_common_suffixes(self, string_values: List[str]) -> List[str]:
        if not string_values:
            return []
            
        suffixes = Counter()
        for value in string_values:
            for i in range(1, min(len(value) + 1, 6)):
                suffixes[value[-i:]] += 1
                
        common_suffixes = [suffix for suffix, count in suffixes.items() 
                          if count >= max(2, len(string_values) * 0.1)]
        return common_suffixes[:5]
        
    def _extract_format_templates(self, string_values: List[str]) -> List[str]:
        templates = Counter()
        for value in string_values:
            template = re.sub(r'\d', 'N', re.sub(r'[a-zA-Z]', 'A', re.sub(r'[^a-zA-Z0-9]', 'S', value)))
            templates[template] += 1
            
        return [template for template, count in templates.most_common(5)]
        
    def _analyze_delimiter_usage(self, string_values: List[str]) -> Dict[str, float]:
        delimiters = [',', ';', '|', ':', '-', '_', '.', ' ']
        usage = {}
        
        for delimiter in delimiters:
            count = sum(value.count(delimiter) for value in string_values)
            usage[delimiter] = count / len(string_values) if string_values else 0.0
            
        return usage
        
    def _analyze_case_patterns(self, string_values: List[str]) -> Dict[str, float]:
        patterns = {
            'all_upper': sum(1 for v in string_values if v.isupper()) / len(string_values) if string_values else 0.0,
            'all_lower': sum(1 for v in string_values if v.islower()) / len(string_values) if string_values else 0.0,
            'title_case': sum(1 for v in string_values if v.istitle()) / len(string_values) if string_values else 0.0,
            'mixed_case': sum(1 for v in string_values if any(c.isupper() for c in v) and any(c.islower() for c in v)) / len(string_values) if string_values else 0.0
        }
        return patterns
        
    def _calculate_name_semantic_weight(self, field_name: str) -> float:
        semantic_keywords = {
            'host': 0.8, 'hostname': 0.9, 'asset': 0.8, 'device': 0.7,
            'log': 0.8, 'event': 0.8, 'message': 0.7, 'activity': 0.6,
            'time': 0.8, 'timestamp': 0.9, 'date': 0.7, 'created': 0.6,
            'source': 0.7, 'platform': 0.7, 'tool': 0.6, 'system': 0.6,
            'type': 0.5, 'category': 0.5, 'role': 0.6, 'function': 0.6,
            'count': 0.6, 'total': 0.6, 'num': 0.5, 'size': 0.5
        }
        
        name_lower = field_name.lower()
        max_weight = 0.0
        
        for keyword, weight in semantic_keywords.items():
            if keyword in name_lower:
                max_weight = max(max_weight, weight)
                
        return max_weight
        
    def _calculate_value_semantic_weight(self, values: List[Any]) -> float:
        if not values:
            return 0.0
            
        string_values = [str(v).lower() for v in values[:30]]
        
        semantic_patterns = [
            (r'^[a-z0-9\-\.]+$', 0.6),
            (r'\d{4}-\d{2}-\d{2}', 0.8),
            (r'^[a-z0-9\-]{4,20}$', 0.7),
            (r'^\d+$', 0.5),
            (r'(splunk|chronicle|crowdstrike)', 0.9)
        ]
        
        max_weight = 0.0
        for pattern, weight in semantic_patterns:
            matches = sum(1 for v in string_values if re.search(pattern, v))
            match_ratio = matches / len(string_values)
            if match_ratio > 0.1:
                max_weight = max(max_weight, weight * match_ratio)
                
        return min(1.0, max_weight)
        
    def _calculate_pattern_coherence(self, profile: SemanticProfile) -> float:
        if not profile.pattern_features:
            return 0.0
            
        coherence_scores = []
        for feature_name, feature_values in profile.pattern_features.items():
            if feature_values:
                variance = statistics.variance(feature_values) if len(feature_values) > 1 else 0.0
                coherence = 1.0 / (1.0 + variance)
                coherence_scores.append(coherence)
                
        return statistics.mean(coherence_scores) if coherence_scores else 0.0
        
    def _calculate_content_alignment(self, field_name: str, values: List[Any], profile: SemanticProfile) -> float:
        if not profile.content_intelligence:
            return 0.0
            
        name_lower = field_name.lower()
        alignment_scores = []
        
        for content_type, intelligence in profile.content_intelligence.items():
            name_alignment = 0.0
            
            if content_type == 'asset_identifier' and any(kw in name_lower for kw in ['host', 'asset', 'device']):
                name_alignment = 0.8
            elif content_type == 'logging_activity' and any(kw in name_lower for kw in ['log', 'event', 'message']):
                name_alignment = 0.8
            elif content_type == 'temporal' and any(kw in name_lower for kw in ['time', 'date', 'timestamp']):
                name_alignment = 0.8
            elif content_type == 'platform_source' and any(kw in name_lower for kw in ['source', 'platform']):
                name_alignment = 0.8
            else:
                name_alignment = 0.3
                
            combined_alignment = (intelligence.confidence + name_alignment) / 2.0
            alignment_scores.append(combined_alignment)
            
        return max(alignment_scores) if alignment_scores else 0.0
        
    def _calculate_format_consistency(self, string_values: List[str]) -> float:
        if not string_values:
            return 0.0
            
        format_counter = Counter()
        for value in string_values:
            template = re.sub(r'\d', 'N', re.sub(r'[a-zA-Z]', 'A', re.sub(r'[^a-zA-Z0-9]', 'S', value)))
            format_counter[template] += 1
            
        if format_counter:
            most_common_count = format_counter.most_common(1)[0][1]
            return most_common_count / len(string_values)
        else:
            return 0.0
            
    def _calculate_length_consistency(self, string_values: List[str]) -> float:
        if not string_values:
            return 0.0
            
        lengths = [len(v) for v in string_values]
        if len(set(lengths)) == 1:
            return 1.0
        else:
            mean_length = statistics.mean(lengths)
            if mean_length > 0:
                std_length = statistics.stdev(lengths) if len(lengths) > 1 else 0
                return max(0.0, 1.0 - (std_length / mean_length))
            else:
                return 0.0
                
    def _calculate_character_consistency(self, string_values: List[str]) -> float:
        if not string_values:
            return 0.0
            
        char_type_ratios = []
        for value in string_values:
            if value:
                alpha_ratio = sum(c.isalpha() for c in value) / len(value)
                digit_ratio = sum(c.isdigit() for c in value) / len(value)
                special_ratio = sum(not c.isalnum() for c in value) / len(value)
                char_type_ratios.append((alpha_ratio, digit_ratio, special_ratio))
                
        if not char_type_ratios:
            return 0.0
            
        alpha_consistency = 1.0 - statistics.stdev([ratio[0] for ratio in char_type_ratios]) if len(char_type_ratios) > 1 else 1.0
        digit_consistency = 1.0 - statistics.stdev([ratio[1] for ratio in char_type_ratios]) if len(char_type_ratios) > 1 else 1.0
        special_consistency = 1.0 - statistics.stdev([ratio[2] for ratio in char_type_ratios]) if len(char_type_ratios) > 1 else 1.0
        
        return statistics.mean([alpha_consistency, digit_consistency, special_consistency])
        
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        if not str1 or not str2:
            return 0.0
            
        set1, set2 = set(str1.lower()), set(str2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 1.0 if len(str1) == len(str2) == 0 else 0.0
            
        jaccard_similarity = intersection / union
        
        length_similarity = 1.0 - abs(len(str1) - len(str2)) / max(len(str1), len(str2))
        
        return (jaccard_similarity + length_similarity) / 2.0