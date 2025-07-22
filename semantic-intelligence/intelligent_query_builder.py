#!/usr/bin/env python3

import statistics
import itertools
from typing import List, Dict, Any, Optional, Set, Tuple
from models import FieldIntelligence, IntelligentQuery, FieldCombination
from database_connector import DatabaseConnector
import re

class IntelligentQueryBuilder:
    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        self.field_semantic_cache = {}
        self.value_analysis_cache = {}
        self.join_strategy_cache = {}
        
    def build_ao1_global_visibility_query(self, fields: List[FieldIntelligence]) -> Optional[IntelligentQuery]:
        asset_fields = self._find_semantic_assets(fields)
        activity_fields = self._find_semantic_activity(fields)
        time_fields = self._find_semantic_time(fields)
        
        if not asset_fields:
            return None
            
        best_combinations = self._find_optimal_field_combinations(asset_fields, activity_fields, time_fields)
        
        for combination in best_combinations:
            query = self._build_visibility_query_for_combination(combination)
            if query and self._validate_query_intelligence(query):
                return query
                
        return None
        
    def _find_semantic_assets(self, fields: List[FieldIntelligence]) -> List[FieldIntelligence]:
        asset_candidates = []
        
        for field in fields:
            asset_score = self._calculate_asset_semantic_score(field)
            if asset_score > 0.4:
                asset_candidates.append((field, asset_score))
                
        asset_candidates.sort(key=lambda x: x[1], reverse=True)
        return [field for field, score in asset_candidates[:10]]
        
    def _calculate_asset_semantic_score(self, field: FieldIntelligence) -> float:
        if field.get_key() in self.field_semantic_cache:
            return self.field_semantic_cache[field.get_key()]
            
        score = 0.0
        
        if field.semantic_profile and field.semantic_profile.content_intelligence:
            asset_intel = field.semantic_profile.content_intelligence.get('asset_identifier')
            hostname_intel = field.semantic_profile.content_intelligence.get('hostname')
            
            if asset_intel:
                score += asset_intel.confidence * 0.5
            if hostname_intel:
                score += hostname_intel.confidence * 0.4
                
        score += self._analyze_asset_name_semantics(field.name) * 0.3
        score += self._analyze_asset_value_semantics(field.sample_values) * 0.4
        score += self._analyze_asset_uniqueness(field) * 0.2
        
        self.field_semantic_cache[field.get_key()] = min(1.0, score)
        return self.field_semantic_cache[field.get_key()]
        
    def _analyze_asset_name_semantics(self, field_name: str) -> float:
        name_lower = field_name.lower()
        
        asset_patterns = {
            r'\bhostname\b': 0.9,
            r'\basset.*name\b': 0.8,
            r'\bci_name\b': 0.8,
            r'\bdevice.*name\b': 0.7,
            r'\bserver.*name\b': 0.7,
            r'\bhost\b': 0.6,
            r'\basset\b': 0.6,
            r'\bdevice\b': 0.5,
            r'\bcomputer\b': 0.5,
            r'\bendpoint\b': 0.5,
            r'\bnode\b': 0.4,
            r'\bmachine\b': 0.4
        }
        
        max_score = 0.0
        for pattern, score in asset_patterns.items():
            if re.search(pattern, name_lower):
                max_score = max(max_score, score)
                
        return max_score
        
    def _analyze_asset_value_semantics(self, sample_values: List[Any]) -> float:
        if not sample_values:
            return 0.0
            
        string_values = [str(v).lower() for v in sample_values[:50]]
        
        asset_value_patterns = [
            (r'^[a-z0-9\-]+\.(local|corp|com|net|org)$', 0.8),
            (r'^[a-z]{2,4}[0-9]{2,6}$', 0.7),
            (r'^(srv|web|db|app|dc|ad|mail|dns)[0-9]{1,4}$', 0.8),
            (r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', 0.4),
            (r'^[a-z0-9\-]{5,25}$', 0.5),
            (r'^[A-Z]{2,4}\d{3,6}$', 0.6)
        ]
        
        total_score = 0.0
        for pattern, weight in asset_value_patterns:
            matches = sum(1 for v in string_values if re.match(pattern, v))
            match_ratio = matches / len(string_values) if string_values else 0
            total_score += weight * match_ratio
            
        return min(1.0, total_score)
        
    def _analyze_asset_uniqueness(self, field: FieldIntelligence) -> float:
        if not field.sample_values:
            return 0.0
            
        unique_ratio = len(set(str(v) for v in field.sample_values)) / len(field.sample_values)
        
        if field.semantic_profile and field.semantic_profile.statistical_properties:
            variability = field.semantic_profile.statistical_properties.get('unique_ratio', 0.0)
            return max(unique_ratio, variability)
            
        return unique_ratio
        
    def _find_semantic_activity(self, fields: List[FieldIntelligence]) -> List[FieldIntelligence]:
        activity_candidates = []
        
        for field in fields:
            activity_score = self._calculate_activity_semantic_score(field)
            if activity_score > 0.3:
                activity_candidates.append((field, activity_score))
                
        activity_candidates.sort(key=lambda x: x[1], reverse=True)
        return [field for field, score in activity_candidates[:8]]
        
    def _calculate_activity_semantic_score(self, field: FieldIntelligence) -> float:
        if field.get_key() in self.field_semantic_cache:
            return self.field_semantic_cache[field.get_key()]
            
        score = 0.0
        
        if field.semantic_profile and field.semantic_profile.content_intelligence:
            logging_intel = field.semantic_profile.content_intelligence.get('logging_activity')
            count_intel = field.semantic_profile.content_intelligence.get('count_metric')
            
            if logging_intel:
                score += logging_intel.confidence * 0.6
            if count_intel:
                score += count_intel.confidence * 0.4
                
        score += self._analyze_activity_name_semantics(field.name) * 0.3
        score += self._analyze_activity_value_semantics(field.sample_values) * 0.3
        score += self._analyze_activity_variability(field) * 0.2
        
        self.field_semantic_cache[field.get_key()] = min(1.0, score)
        return self.field_semantic_cache[field.get_key()]
        
    def _analyze_activity_name_semantics(self, field_name: str) -> float:
        name_lower = field_name.lower()
        
        activity_patterns = {
            r'\blog.*count\b': 0.9,
            r'\bevent.*count\b': 0.9,
            r'\bactivity.*count\b': 0.8,
            r'\bmessage.*count\b': 0.8,
            r'\blog\b': 0.7,
            r'\bevent\b': 0.7,
            r'\bmessage\b': 0.6,
            r'\bdata\b': 0.5,
            r'\brecord\b': 0.5,
            r'\bentry\b': 0.5,
            r'\bactivity\b': 0.6,
            r'\btotal\b': 0.4,
            r'\bcount\b': 0.5
        }
        
        max_score = 0.0
        for pattern, score in activity_patterns.items():
            if re.search(pattern, name_lower):
                max_score = max(max_score, score)
                
        return max_score
        
    def _analyze_activity_value_semantics(self, sample_values: List[Any]) -> float:
        if not sample_values:
            return 0.0
            
        string_values = [str(v).lower() for v in sample_values[:50]]
        
        activity_indicators = [
            (r'(error|warning|info|debug|trace|fatal)', 0.6),
            (r'(success|failed|complete|started|finished)', 0.5),
            (r'(login|logout|access|denied|granted)', 0.7),
            (r'^\d+$', 0.4),
            (r'\d{4}-\d{2}-\d{2}', 0.3)
        ]
        
        total_score = 0.0
        for pattern, weight in activity_indicators:
            matches = sum(1 for v in string_values if re.search(pattern, v))
            match_ratio = matches / len(string_values) if string_values else 0
            total_score += weight * match_ratio
            
        return min(1.0, total_score)
        
    def _analyze_activity_variability(self, field: FieldIntelligence) -> float:
        if not field.semantic_profile:
            return 0.0
            
        variability = field.semantic_profile.behavioral_indicators.get('variability', 0.0)
        return min(1.0, variability * 1.5)
        
    def _find_semantic_time(self, fields: List[FieldIntelligence]) -> List[FieldIntelligence]:
        time_candidates = []
        
        for field in fields:
            time_score = self._calculate_time_semantic_score(field)
            if time_score > 0.4:
                time_candidates.append((field, time_score))
                
        time_candidates.sort(key=lambda x: x[1], reverse=True)
        return [field for field, score in time_candidates[:5]]
        
    def _calculate_time_semantic_score(self, field: FieldIntelligence) -> float:
        if field.get_key() in self.field_semantic_cache:
            return self.field_semantic_cache[field.get_key()]
            
        score = 0.0
        
        if field.semantic_profile and field.semantic_profile.content_intelligence:
            temporal_intel = field.semantic_profile.content_intelligence.get('temporal')
            if temporal_intel:
                score += temporal_intel.confidence * 0.7
                
        score += self._analyze_time_name_semantics(field.name) * 0.4
        score += self._analyze_time_value_semantics(field.sample_values) * 0.5
        
        self.field_semantic_cache[field.get_key()] = min(1.0, score)
        return self.field_semantic_cache[field.get_key()]
        
    def _analyze_time_name_semantics(self, field_name: str) -> float:
        name_lower = field_name.lower()
        
        time_patterns = {
            r'\btimestamp\b': 0.9,
            r'\b_time\b': 0.9,
            r'\bcreated.*time\b': 0.8,
            r'\bmodified.*time\b': 0.8,
            r'\boccurred.*time\b': 0.8,
            r'\btime\b': 0.7,
            r'\bdate\b': 0.7,
            r'\bcreated\b': 0.6,
            r'\bmodified\b': 0.6,
            r'\boccurred\b': 0.6,
            r'\bwhen\b': 0.5,
            r'_at$': 0.6
        }
        
        max_score = 0.0
        for pattern, score in time_patterns.items():
            if re.search(pattern, name_lower):
                max_score = max(max_score, score)
                
        return max_score
        
    def _analyze_time_value_semantics(self, sample_values: List[Any]) -> float:
        if not sample_values:
            return 0.0
            
        string_values = [str(v) for v in sample_values[:50]]
        
        time_patterns = [
            (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', 0.9),
            (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', 0.9),
            (r'\d{4}-\d{2}-\d{2}', 0.8),
            (r'^\d{10,13}$', 0.7),
            (r'\d{2}/\d{2}/\d{4}', 0.6),
            (r'\d{2}:\d{2}:\d{2}', 0.5)
        ]
        
        total_score = 0.0
        for pattern, weight in time_patterns:
            matches = sum(1 for v in string_values if re.search(pattern, v))
            match_ratio = matches / len(string_values) if string_values else 0
            total_score += weight * match_ratio
            
        return min(1.0, total_score)
        
    def _find_optimal_field_combinations(self, asset_fields: List[FieldIntelligence], 
                                       activity_fields: List[FieldIntelligence],
                                       time_fields: List[FieldIntelligence]) -> List[FieldCombination]:
        combinations = []
        
        for asset_field in asset_fields[:3]:
            
            if activity_fields:
                for activity_field in activity_fields[:3]:
                    
                    if time_fields:
                        for time_field in time_fields[:2]:
                            combination = self._create_three_field_combination(asset_field, activity_field, time_field)
                            if combination:
                                combinations.append(combination)
                    
                    combination = self._create_two_field_combination(asset_field, activity_field, 'asset_activity')
                    if combination:
                        combinations.append(combination)
            
            if time_fields:
                for time_field in time_fields[:2]:
                    combination = self._create_two_field_combination(asset_field, time_field, 'asset_time')
                    if combination:
                        combinations.append(combination)
            
            combination = self._create_single_field_combination(asset_field)
            if combination:
                combinations.append(combination)
                
        combinations.sort(key=lambda c: c.expected_accuracy, reverse=True)
        return combinations[:5]
        
    def _create_three_field_combination(self, asset_field: FieldIntelligence, 
                                      activity_field: FieldIntelligence,
                                      time_field: FieldIntelligence) -> Optional[FieldCombination]:
        
        relationship_score = self._calculate_field_relationship_strength(asset_field, activity_field, time_field)
        if relationship_score < 0.3:
            return None
            
        semantic_reasoning = f"Asset inventory ({asset_field.get_key()}) correlated with activity evidence ({activity_field.get_key()}) filtered by recency ({time_field.get_key()})"
        
        intelligence_basis = {
            'asset_semantic_score': self._calculate_asset_semantic_score(asset_field),
            'activity_semantic_score': self._calculate_activity_semantic_score(activity_field),
            'time_semantic_score': self._calculate_time_semantic_score(time_field),
            'relationship_strength': relationship_score,
            'table_alignment': self._calculate_table_alignment([asset_field, activity_field, time_field])
        }
        
        expected_accuracy = statistics.mean([
            intelligence_basis['asset_semantic_score'],
            intelligence_basis['activity_semantic_score'],
            intelligence_basis['time_semantic_score'],
            relationship_score
        ])
        
        return FieldCombination(
            fields=[asset_field, activity_field, time_field],
            combination_type="ASSET_ACTIVITY_TIME",
            semantic_reasoning=semantic_reasoning,
            intelligence_basis=intelligence_basis,
            expected_accuracy=expected_accuracy
        )
        
    def _create_two_field_combination(self, field1: FieldIntelligence, field2: FieldIntelligence, 
                                    combo_type: str) -> Optional[FieldCombination]:
        
        relationship_score = self._calculate_field_relationship_strength(field1, field2)
        if relationship_score < 0.2:
            return None
            
        if combo_type == 'asset_activity':
            semantic_reasoning = f"Asset inventory ({field1.get_key()}) correlated with activity evidence ({field2.get_key()})"
        elif combo_type == 'asset_time':
            semantic_reasoning = f"Asset inventory ({field1.get_key()}) with temporal activity filtering ({field2.get_key()})"
        else:
            semantic_reasoning = f"Correlation between {field1.get_key()} and {field2.get_key()}"
            
        intelligence_basis = {
            'field1_semantic_score': self._get_cached_semantic_score(field1),
            'field2_semantic_score': self._get_cached_semantic_score(field2),
            'relationship_strength': relationship_score,
            'table_alignment': self._calculate_table_alignment([field1, field2])
        }
        
        expected_accuracy = statistics.mean([
            intelligence_basis['field1_semantic_score'],
            intelligence_basis['field2_semantic_score'],
            relationship_score
        ])
        
        return FieldCombination(
            fields=[field1, field2],
            combination_type=combo_type.upper(),
            semantic_reasoning=semantic_reasoning,
            intelligence_basis=intelligence_basis,
            expected_accuracy=expected_accuracy
        )
        
    def _create_single_field_combination(self, field: FieldIntelligence) -> FieldCombination:
        semantic_reasoning = f"Single field statistical analysis of high-confidence asset field ({field.get_key()})"
        
        intelligence_basis = {
            'field_semantic_score': self._get_cached_semantic_score(field),
            'uniqueness_factor': self._analyze_asset_uniqueness(field),
            'value_quality': self._assess_value_quality(field.sample_values)
        }
        
        expected_accuracy = statistics.mean(list(intelligence_basis.values()))
        
        return FieldCombination(
            fields=[field],
            combination_type="SINGLE_ASSET_ANALYSIS",
            semantic_reasoning=semantic_reasoning,
            intelligence_basis=intelligence_basis,
            expected_accuracy=expected_accuracy
        )
        
    def _calculate_field_relationship_strength(self, *fields: FieldIntelligence) -> float:
        if len(fields) < 2:
            return 1.0
            
        relationship_factors = []
        
        table_factor = self._calculate_table_alignment(list(fields))
        relationship_factors.append(table_factor)
        
        if len(fields) == 2:
            semantic_factor = self._calculate_semantic_compatibility(fields[0], fields[1])
            relationship_factors.append(semantic_factor)
            
        elif len(fields) == 3:
            pairwise_compatibilities = [
                self._calculate_semantic_compatibility(fields[0], fields[1]),
                self._calculate_semantic_compatibility(fields[0], fields[2]),
                self._calculate_semantic_compatibility(fields[1], fields[2])
            ]
            semantic_factor = statistics.mean(pairwise_compatibilities)
            relationship_factors.append(semantic_factor)
            
        return statistics.mean(relationship_factors)
        
    def _calculate_table_alignment(self, fields: List[FieldIntelligence]) -> float:
        tables = [field.table for field in fields]
        unique_tables = len(set(tables))
        
        if unique_tables == 1:
            return 1.0
        elif unique_tables == 2:
            return 0.7
        else:
            return 0.4
            
    def _calculate_semantic_compatibility(self, field1: FieldIntelligence, field2: FieldIntelligence) -> float:
        compatibility_matrix = {
            ('asset_identifier', 'logging_activity'): 0.9,
            ('asset_identifier', 'temporal'): 0.8,
            ('asset_identifier', 'count_metric'): 0.7,
            ('hostname', 'logging_activity'): 0.9,
            ('hostname', 'temporal'): 0.8,
            ('logging_activity', 'temporal'): 0.8,
            ('logging_activity', 'count_metric'): 0.7,
        }
        
        type1 = field1.get_semantic_meaning() if hasattr(field1, 'get_semantic_meaning') else 'unknown'
        type2 = field2.get_semantic_meaning() if hasattr(field2, 'get_semantic_meaning') else 'unknown'
        
        compatibility = compatibility_matrix.get((type1, type2), 
                       compatibility_matrix.get((type2, type1), 0.3))
        
        return compatibility
        
    def _build_visibility_query_for_combination(self, combination: FieldCombination) -> Optional[IntelligentQuery]:
        if combination.combination_type == "ASSET_ACTIVITY_TIME":
            return self._build_three_field_visibility_query(combination)
        elif combination.combination_type in ["ASSET_ACTIVITY", "ASSET_TIME"]:
            return self._build_two_field_visibility_query(combination)
        elif combination.combination_type == "SINGLE_ASSET_ANALYSIS":
            return self._build_single_field_visibility_query(combination)
        else:
            return None
            
    def _build_three_field_visibility_query(self, combination: FieldCombination) -> IntelligentQuery:
        asset_field, activity_field, time_field = combination.fields
        
        if asset_field.table == activity_field.table == time_field.table:
            sql = self._build_same_table_three_field_query(asset_field, activity_field, time_field)
            business_logic = "Same-table analysis: assets with recent activity evidence"
        else:
            join_strategy = self._determine_intelligent_join_strategy(combination.fields)
            sql = self._build_cross_table_three_field_query(asset_field, activity_field, time_field, join_strategy)
            business_logic = f"Cross-table correlation using {join_strategy} strategy"
            
        return IntelligentQuery(
            name="AO1 Global Visibility Score",
            description="Intelligent asset visibility calculation using semantic field analysis",
            sql=sql,
            field_combination=combination.fields,
            intelligence_reasoning=combination.intelligence_basis,
            confidence_score=combination.expected_accuracy,
            business_logic=business_logic,
            semantic_coherence=self._calculate_query_semantic_coherence(combination)
        )
        
    def _build_same_table_three_field_query(self, asset_field: FieldIntelligence, 
                                          activity_field: FieldIntelligence, 
                                          time_field: FieldIntelligence) -> str:
        
        time_filter = self._generate_intelligent_time_filter(time_field)
        activity_filter = self._generate_intelligent_activity_filter(activity_field)
        
        return f"""
WITH asset_universe AS (
    SELECT COUNT(DISTINCT {asset_field.name}) as total_assets
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND {self._generate_quality_filter(asset_field)}
),
active_assets AS (
    SELECT COUNT(DISTINCT {asset_field.name}) as visible_assets
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND {activity_field.name} IS NOT NULL
      AND {activity_filter}
      AND {time_filter}
      AND {self._generate_quality_filter(asset_field)}
),
visibility_analysis AS (
    SELECT 
        au.total_assets,
        aa.visible_assets,
        au.total_assets - aa.visible_assets as silent_assets,
        CASE 
            WHEN au.total_assets = 0 THEN 0.0
            ELSE ROUND(100.0 * aa.visible_assets / au.total_assets, 2)
        END as visibility_percentage
    FROM asset_universe au, active_assets aa
)
SELECT 
    total_assets,
    visible_assets,
    silent_assets,
    visibility_percentage,
    CASE 
        WHEN visibility_percentage >= 95 THEN 'EXCELLENT'
        WHEN visibility_percentage >= 85 THEN 'GOOD'
        WHEN visibility_percentage >= 70 THEN 'ACCEPTABLE'
        WHEN visibility_percentage >= 50 THEN 'POOR'
        ELSE 'CRITICAL'
    END as visibility_grade,
    '{asset_field.get_key()}' as asset_source,
    '{activity_field.get_key()}' as activity_source,
    '{time_field.get_key()}' as time_source
FROM visibility_analysis;
        """
        
    def _build_cross_table_three_field_query(self, asset_field: FieldIntelligence, 
                                           activity_field: FieldIntelligence,
                                           time_field: FieldIntelligence,
                                           join_strategy: str) -> str:
        
        if join_strategy == "INTELLIGENT_HOSTNAME_JOIN":
            return self._build_hostname_correlation_query(asset_field, activity_field, time_field)
        elif join_strategy == "SEMANTIC_VALUE_JOIN":
            return self._build_semantic_value_join_query(asset_field, activity_field, time_field)
        else:
            return self._build_statistical_correlation_query(asset_field, activity_field, time_field)
            
    def _build_hostname_correlation_query(self, asset_field: FieldIntelligence,
                                        activity_field: FieldIntelligence,
                                        time_field: FieldIntelligence) -> str:
        
        time_filter = self._generate_intelligent_time_filter(time_field)
        activity_filter = self._generate_intelligent_activity_filter(activity_field)
        
        asset_hostname_transform = self._generate_hostname_transform(asset_field)
        activity_hostname_transform = self._generate_hostname_transform(activity_field)
        
        return f"""
WITH asset_inventory AS (
    SELECT DISTINCT {asset_hostname_transform} as hostname
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND {self._generate_quality_filter(asset_field)}
),
activity_evidence AS (
    SELECT DISTINCT {activity_hostname_transform} as hostname
    FROM {activity_field.table} af
    {'JOIN ' + time_field.table + ' tf ON af.' + self._find_join_key(activity_field.table, time_field.table) + ' = tf.' + self._find_join_key(time_field.table, activity_field.table) if activity_field.table != time_field.table else ''}
    WHERE {activity_field.name} IS NOT NULL
      AND {activity_filter}
      AND {time_filter if activity_field.table == time_field.table else 'tf.' + time_field.name + ' IS NOT NULL'}
),
visibility_calculation AS (
    SELECT 
        COUNT(ai.hostname) as total_assets,
        COUNT(ae.hostname) as visible_assets
    FROM asset_inventory ai
    LEFT JOIN activity_evidence ae ON ai.hostname = ae.hostname
)
SELECT 
    total_assets,
    visible_assets,
    total_assets - visible_assets as silent_assets,
    CASE 
        WHEN total_assets = 0 THEN 0.0
        ELSE ROUND(100.0 * visible_assets / total_assets, 2)
    END as visibility_percentage,
    'HOSTNAME_CORRELATION' as calculation_method,
    '{asset_field.get_key()}' as asset_source,
    '{activity_field.get_key()}' as activity_source,
    '{time_field.get_key()}' as time_source
FROM visibility_calculation;
        """
        
    def _build_two_field_visibility_query(self, combination: FieldCombination) -> IntelligentQuery:
        field1, field2 = combination.fields
        
        if combination.combination_type == "ASSET_ACTIVITY":
            sql = self._build_asset_activity_query(field1, field2)
            business_logic = "Asset-to-activity correlation analysis"
        elif combination.combination_type == "ASSET_TIME":
            sql = self._build_asset_time_query(field1, field2)
            business_logic = "Temporal asset activity analysis"
        else:
            sql = self._build_generic_two_field_query(field1, field2)
            business_logic = "Generic two-field correlation"
            
        return IntelligentQuery(
            name="AO1 Global Visibility Score (Two-Field)",
            description="Intelligent two-field visibility calculation",
            sql=sql,
            field_combination=combination.fields,
            intelligence_reasoning=combination.intelligence_basis,
            confidence_score=combination.expected_accuracy,
            business_logic=business_logic,
            semantic_coherence=self._calculate_query_semantic_coherence(combination)
        )
        
    def _build_single_field_visibility_query(self, combination: FieldCombination) -> IntelligentQuery:
        field = combination.fields[0]
        
        sql = f"""
WITH asset_analysis AS (
    SELECT 
        COUNT(DISTINCT {field.name}) as total_unique_assets,
        COUNT({field.name}) as total_records,
        COUNT(DISTINCT CASE WHEN {self._generate_quality_filter(field)} THEN {field.name} END) as high_quality_assets
    FROM {field.table}
    WHERE {field.name} IS NOT NULL
),
statistical_visibility AS (
    SELECT 
        total_unique_assets,
        high_quality_assets,
        total_unique_assets - high_quality_assets as lower_quality_assets,
        CASE 
            WHEN total_unique_assets = 0 THEN 0.0
            ELSE ROUND(100.0 * high_quality_assets / total_unique_assets, 2)
        END as estimated_visibility_percentage,
        ROUND(100.0 * high_quality_assets / total_records, 2) as data_quality_percentage
    FROM asset_analysis
)
SELECT 
    total_unique_assets as total_assets,
    high_quality_assets as visible_assets,
    lower_quality_assets as silent_assets,
    estimated_visibility_percentage as visibility_percentage,
    'SINGLE_FIELD_STATISTICAL' as calculation_method,
    '{field.get_key()}' as asset_source,
    data_quality_percentage
FROM statistical_visibility;
        """
        
        return IntelligentQuery(
            name="AO1 Global Visibility Score (Statistical)",
            description="Single-field statistical visibility estimate",
            sql=sql,
            field_combination=combination.fields,
            intelligence_reasoning=combination.intelligence_basis,
            confidence_score=combination.expected_accuracy,
            business_logic="Statistical analysis of single high-confidence asset field",
            semantic_coherence=self._calculate_query_semantic_coherence(combination)
        )
        
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
        
        if any(re.search(r'^\d+$', v) for v in string_values):
            return f"{activity_field.name} > 0"
        elif any('error' in v or 'fail' in v for v in string_values):
            return f"LOWER({activity_field.name}) NOT LIKE '%error%' AND LOWER({activity_field.name}) NOT LIKE '%fail%'"
        else:
            return f"{activity_field.name} IS NOT NULL"
            
    def _generate_quality_filter(self, field: FieldIntelligence) -> str:
        if not field.sample_values:
            return "1=1"
            
        string_values = [str(v) for v in field.sample_values[:20]]
        avg_length = statistics.mean([len(v) for v in string_values]) if string_values else 0
        
        if avg_length > 2:
            return f"LENGTH({field.name}) >= 3"
        else:
            return f"{field.name} != '' AND {field.name} != '0'"
            
    def _generate_hostname_transform(self, field: FieldIntelligence) -> str:
        if not field.sample_values:
            return field.name
            
        sample_value = str(field.sample_values[0]).lower()
        
        if '.' in sample_value and len(sample_value.split('.')) > 2:
            return f"SUBSTR({field.name}, 1, INSTR({field.name}, '.') - 1)"
        else:
            return field.name
            
    def _determine_intelligent_join_strategy(self, fields: List[FieldIntelligence]) -> str:
        table_count = len(set(f.table for f in fields))
        
        if table_count == 1:
            return "SAME_TABLE"
            
        hostname_like_count = sum(1 for f in fields if self._is_hostname_like(f))
        
        if hostname_like_count >= 2:
            return "INTELLIGENT_HOSTNAME_JOIN"
            
        value_overlap = self._calculate_value_overlap_potential(fields)
        if value_overlap > 0.3:
            return "SEMANTIC_VALUE_JOIN"
            
        return "STATISTICAL_CORRELATION"
        
    def _is_hostname_like(self, field: FieldIntelligence) -> bool:
        name_lower = field.name.lower()
        hostname_indicators = ['host', 'hostname', 'device', 'server', 'computer', 'node', 'machine']
        
        if any(indicator in name_lower for indicator in hostname_indicators):
            return True
            
        if field.semantic_profile and field.semantic_profile.content_intelligence:
            hostname_intel = field.semantic_profile.content_intelligence.get('hostname')
            asset_intel = field.semantic_profile.content_intelligence.get('asset_identifier')
            return (hostname_intel and hostname_intel.confidence > 0.5) or (asset_intel and asset_intel.confidence > 0.6)
            
        return False
        
    def _calculate_value_overlap_potential(self, fields: List[FieldIntelligence]) -> float:
        if len(fields) < 2:
            return 0.0
            
        value_sets = []
        for field in fields:
            if field.sample_values:
                value_set = set(str(v).lower() for v in field.sample_values[:50])
                value_sets.append(value_set)
                
        if len(value_sets) < 2:
            return 0.0
            
        max_overlap = 0.0
        for i in range(len(value_sets)):
            for j in range(i + 1, len(value_sets)):
                intersection = len(value_sets[i] & value_sets[j])
                union = len(value_sets[i] | value_sets[j])
                overlap = intersection / union if union > 0 else 0.0
                max_overlap = max(max_overlap, overlap)
                
        return max_overlap
        
    def _build_semantic_value_join_query(self, asset_field: FieldIntelligence,
                                       activity_field: FieldIntelligence,
                                       time_field: FieldIntelligence) -> str:
        
        value_transform1 = self._generate_value_normalization(asset_field)
        value_transform2 = self._generate_value_normalization(activity_field)
        time_filter = self._generate_intelligent_time_filter(time_field)
        
        return f"""
WITH normalized_assets AS (
    SELECT DISTINCT {value_transform1} as normalized_value
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND {self._generate_quality_filter(asset_field)}
),
normalized_activity AS (
    SELECT DISTINCT {value_transform2} as normalized_value
    FROM {activity_field.table} af
    {'JOIN ' + time_field.table + ' tf ON af.' + self._find_join_key(activity_field.table, time_field.table) + ' = tf.' + self._find_join_key(time_field.table, activity_field.table) if activity_field.table != time_field.table else ''}
    WHERE {activity_field.name} IS NOT NULL
      AND {time_filter if activity_field.table == time_field.table else 'tf.' + time_field.name + ' IS NOT NULL'}
),
value_correlation AS (
    SELECT 
        COUNT(na.normalized_value) as total_assets,
        COUNT(nac.normalized_value) as visible_assets
    FROM normalized_assets na
    LEFT JOIN normalized_activity nac ON na.normalized_value = nac.normalized_value
)
SELECT 
    total_assets,
    visible_assets,
    total_assets - visible_assets as silent_assets,
    CASE 
        WHEN total_assets = 0 THEN 0.0
        ELSE ROUND(100.0 * visible_assets / total_assets, 2)
    END as visibility_percentage,
    'SEMANTIC_VALUE_CORRELATION' as calculation_method
FROM value_correlation;
        """
        
    def _build_statistical_correlation_query(self, asset_field: FieldIntelligence,
                                           activity_field: FieldIntelligence,
                                           time_field: FieldIntelligence) -> str:
        return f"""
WITH asset_stats AS (
    SELECT COUNT(DISTINCT {asset_field.name}) as total_assets
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND {self._generate_quality_filter(asset_field)}
),
activity_stats AS (
    SELECT COUNT(DISTINCT {activity_field.name}) as unique_activity_sources
    FROM {activity_field.table} af
    {'JOIN ' + time_field.table + ' tf ON af.' + self._find_join_key(activity_field.table, time_field.table) + ' = tf.' + self._find_join_key(time_field.table, activity_field.table) if activity_field.table != time_field.table else ''}
    WHERE {activity_field.name} IS NOT NULL
      AND {self._generate_intelligent_time_filter(time_field) if activity_field.table == time_field.table else 'tf.' + time_field.name + ' IS NOT NULL'}
),
statistical_estimate AS (
    SELECT 
        ass.total_assets,
        CASE 
            WHEN acs.unique_activity_sources > ass.total_assets THEN ass.total_assets
            ELSE acs.unique_activity_sources
        END as estimated_visible_assets
    FROM asset_stats ass, activity_stats acs
)
SELECT 
    total_assets,
    estimated_visible_assets as visible_assets,
    total_assets - estimated_visible_assets as silent_assets,
    CASE 
        WHEN total_assets = 0 THEN 0.0
        ELSE ROUND(100.0 * estimated_visible_assets / total_assets, 2)
    END as visibility_percentage,
    'STATISTICAL_ESTIMATE' as calculation_method
FROM statistical_estimate;
        """
        
    def _generate_value_normalization(self, field: FieldIntelligence) -> str:
        if not field.sample_values:
            return f"LOWER(TRIM({field.name}))"
            
        sample_value = str(field.sample_values[0])
        
        if '.' in sample_value and re.match(r'^[a-z0-9\-\.]+$', sample_value.lower()):
            return f"LOWER(SUBSTR({field.name}, 1, CASE WHEN INSTR({field.name}, '.') > 0 THEN INSTR({field.name}, '.') - 1 ELSE LENGTH({field.name}) END))"
        elif re.match(r'^[A-Z0-9\-_]+$', sample_value):
            return f"UPPER(TRIM({field.name}))"
        else:
            return f"LOWER(TRIM({field.name}))"
            
    def _find_join_key(self, table1: str, table2: str) -> str:
        common_keys = ['id', 'key', 'identifier', 'uuid', 'guid']
        for key in common_keys:
            return key
        return 'id'
        
    def _build_asset_activity_query(self, asset_field: FieldIntelligence, activity_field: FieldIntelligence) -> str:
        if asset_field.table == activity_field.table:
            return f"""
WITH asset_activity_analysis AS (
    SELECT 
        COUNT(DISTINCT {asset_field.name}) as total_assets,
        COUNT(DISTINCT CASE WHEN {activity_field.name} IS NOT NULL AND {self._generate_intelligent_activity_filter(activity_field)} THEN {asset_field.name} END) as active_assets
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND {self._generate_quality_filter(asset_field)}
)
SELECT 
    total_assets,
    active_assets as visible_assets,
    total_assets - active_assets as silent_assets,
    CASE 
        WHEN total_assets = 0 THEN 0.0
        ELSE ROUND(100.0 * active_assets / total_assets, 2)
    END as visibility_percentage
FROM asset_activity_analysis;
            """
        else:
            join_strategy = self._determine_intelligent_join_strategy([asset_field, activity_field])
            
            if join_strategy == "INTELLIGENT_HOSTNAME_JOIN":
                asset_transform = self._generate_hostname_transform(asset_field)
                activity_transform = self._generate_hostname_transform(activity_field)
                
                return f"""
WITH asset_inventory AS (
    SELECT DISTINCT {asset_transform} as identifier
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND {self._generate_quality_filter(asset_field)}
),
activity_sources AS (
    SELECT DISTINCT {activity_transform} as identifier
    FROM {activity_field.table}
    WHERE {activity_field.name} IS NOT NULL
      AND {self._generate_intelligent_activity_filter(activity_field)}
)
SELECT 
    COUNT(ai.identifier) as total_assets,
    COUNT(acs.identifier) as visible_assets,
    COUNT(ai.identifier) - COUNT(acs.identifier) as silent_assets,
    CASE 
        WHEN COUNT(ai.identifier) = 0 THEN 0.0
        ELSE ROUND(100.0 * COUNT(acs.identifier) / COUNT(ai.identifier), 2)
    END as visibility_percentage
FROM asset_inventory ai
LEFT JOIN activity_sources acs ON ai.identifier = acs.identifier;
                """
            else:
                return self._build_statistical_correlation_query(asset_field, activity_field, activity_field)
                
    def _build_asset_time_query(self, asset_field: FieldIntelligence, time_field: FieldIntelligence) -> str:
        time_filter = self._generate_intelligent_time_filter(time_field)
        
        if asset_field.table == time_field.table:
            return f"""
WITH temporal_asset_analysis AS (
    SELECT 
        COUNT(DISTINCT {asset_field.name}) as total_assets,
        COUNT(DISTINCT CASE WHEN {time_filter} THEN {asset_field.name} END) as recent_assets
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND {self._generate_quality_filter(asset_field)}
)
SELECT 
    total_assets,
    recent_assets as visible_assets,
    total_assets - recent_assets as silent_assets,
    CASE 
        WHEN total_assets = 0 THEN 0.0
        ELSE ROUND(100.0 * recent_assets / total_assets, 2)
    END as visibility_percentage
FROM temporal_asset_analysis;
            """
        else:
            return f"""
WITH all_assets AS (
    SELECT COUNT(DISTINCT {asset_field.name}) as total_assets
    FROM {asset_field.table}
    WHERE {asset_field.name} IS NOT NULL
      AND {self._generate_quality_filter(asset_field)}
),
recent_activity AS (
    SELECT COUNT(DISTINCT {time_field.name}) as recent_records
    FROM {time_field.table}
    WHERE {time_filter}
)
SELECT 
    aa.total_assets,
    LEAST(aa.total_assets, ra.recent_records) as visible_assets,
    aa.total_assets - LEAST(aa.total_assets, ra.recent_records) as silent_assets,
    CASE 
        WHEN aa.total_assets = 0 THEN 0.0
        ELSE ROUND(100.0 * LEAST(aa.total_assets, ra.recent_records) / aa.total_assets, 2)
    END as visibility_percentage
FROM all_assets aa, recent_activity ra;
            """
            
    def _build_generic_two_field_query(self, field1: FieldIntelligence, field2: FieldIntelligence) -> str:
        return f"""
WITH field1_analysis AS (
    SELECT COUNT(DISTINCT {field1.name}) as field1_count
    FROM {field1.table}
    WHERE {field1.name} IS NOT NULL
),
field2_analysis AS (
    SELECT COUNT(DISTINCT {field2.name}) as field2_count
    FROM {field2.table}
    WHERE {field2.name} IS NOT NULL
)
SELECT 
    f1.field1_count as total_assets,
    LEAST(f1.field1_count, f2.field2_count) as visible_assets,
    ABS(f1.field1_count - f2.field2_count) as difference,
    CASE 
        WHEN f1.field1_count = 0 THEN 0.0
        ELSE ROUND(100.0 * LEAST(f1.field1_count, f2.field2_count) / f1.field1_count, 2)
    END as visibility_percentage
FROM field1_analysis f1, field2_analysis f2;
        """
        
    def _validate_query_intelligence(self, query: IntelligentQuery) -> bool:
        if query.confidence_score < 0.3:
            return False
            
        if not query.sql or len(query.sql.strip()) < 50:
            return False
            
        required_elements = ['SELECT', 'FROM', 'WHERE']
        sql_upper = query.sql.upper()
        if not all(element in sql_upper for element in required_elements):
            return False
            
        return True
        
    def _calculate_query_semantic_coherence(self, combination: FieldCombination) -> float:
        coherence_factors = []
        
        avg_field_intelligence = statistics.mean([
            self._get_cached_semantic_score(field) for field in combination.fields
        ])
        coherence_factors.append(avg_field_intelligence)
        
        coherence_factors.append(combination.expected_accuracy)
        
        if len(combination.fields) > 1:
            relationship_strength = combination.intelligence_basis.get('relationship_strength', 0.5)
            coherence_factors.append(relationship_strength)
            
        semantic_alignment = self._calculate_semantic_field_alignment(combination.fields)
        coherence_factors.append(semantic_alignment)
        
        return statistics.mean(coherence_factors)
        
    def _calculate_semantic_field_alignment(self, fields: List[FieldIntelligence]) -> float:
        if len(fields) < 2:
            return 1.0
            
        semantic_types = []
        for field in fields:
            if hasattr(field, 'get_semantic_meaning'):
                semantic_types.append(field.get_semantic_meaning())
            else:
                semantic_types.append('unknown')
                
        alignment_scores = []
        expected_combinations = [
            {'asset_identifier', 'logging_activity'},
            {'hostname', 'logging_activity'},
            {'asset_identifier', 'temporal'},
            {'hostname', 'temporal'},
            {'logging_activity', 'temporal'}
        ]
        
        field_types = set(semantic_types)
        for expected in expected_combinations:
            if field_types.issubset(expected) or expected.issubset(field_types):
                alignment_scores.append(0.9)
                break
        else:
            alignment_scores.append(0.5)
            
        return statistics.mean(alignment_scores) if alignment_scores else 0.5
        
    def _get_cached_semantic_score(self, field: FieldIntelligence) -> float:
        cache_key = field.get_key()
        if cache_key in self.field_semantic_cache:
            return self.field_semantic_cache[cache_key]
            
        if field.intelligence_score > 0:
            return field.intelligence_score
            
        semantic_score = max([
            self._calculate_asset_semantic_score(field),
            self._calculate_activity_semantic_score(field),
            self._calculate_time_semantic_score(field)
        ])
        
        self.field_semantic_cache[cache_key] = semantic_score
        return semantic_score
        
    def _assess_value_quality(self, sample_values: List[Any]) -> float:
        if not sample_values:
            return 0.0
            
        string_values = [str(v) for v in sample_values]
        
        quality_factors = []
        
        non_empty_ratio = sum(1 for v in string_values if v and v.strip()) / len(string_values)
        quality_factors.append(non_empty_ratio)
        
        avg_length = statistics.mean([len(v) for v in string_values])
        length_quality = min(1.0, avg_length / 10.0) if avg_length > 0 else 0.0
        quality_factors.append(length_quality)
        
        unique_ratio = len(set(string_values)) / len(string_values)
        quality_factors.append(unique_ratio)
        
        return statistics.mean(quality_factors)