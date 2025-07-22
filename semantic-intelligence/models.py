#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple

@dataclass
class ContentIntelligence:
    content_type: str
    confidence: float
    evidence: List[str]
    pattern_matches: Dict[str, float]
    semantic_classification: str

@dataclass
class SemanticProfile:
    field_name: str
    content_intelligence: Dict[str, ContentIntelligence] = field(default_factory=dict)
    entropy_metrics: Dict[str, float] = field(default_factory=dict)
    pattern_features: Dict[str, List[float]] = field(default_factory=dict)
    statistical_properties: Dict[str, float] = field(default_factory=dict)
    behavioral_indicators: Dict[str, float] = field(default_factory=dict)
    value_analysis: Dict[str, Any] = field(default_factory=dict)
    cross_field_patterns: Dict[str, float] = field(default_factory=dict)
    semantic_density: float = 0.0
    complexity_score: float = 0.0
    meaning_confidence: float = 0.0

@dataclass
class FieldIntelligence:
    name: str
    table: str
    data_type: str
    semantic_profile: Optional[SemanticProfile] = None
    intelligence_score: float = 0.0
    confidence_level: float = 0.0
    business_context: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, float] = field(default_factory=dict)
    sample_values: List[Any] = field(default_factory=list)
    value_patterns: Dict[str, Any] = field(default_factory=dict)
    meaning_indicators: Dict[str, float] = field(default_factory=dict)
    cross_references: Dict[str, List[str]] = field(default_factory=dict)
    
    def get_key(self) -> str:
        return f"{self.table}.{self.name}"
    
    def get_semantic_meaning(self) -> str:
        if self.semantic_profile and self.semantic_profile.content_intelligence:
            highest_confidence = 0.0
            best_meaning = "unknown"
            for content_type, intelligence in self.semantic_profile.content_intelligence.items():
                if intelligence.confidence > highest_confidence:
                    highest_confidence = intelligence.confidence
                    best_meaning = intelligence.semantic_classification
            return best_meaning
        return "unknown"

@dataclass
class IntelligentQuery:
    name: str
    description: str
    sql: str
    field_combination: List[FieldIntelligence]
    intelligence_reasoning: Dict[str, Any]
    confidence_score: float = 0.0
    complexity_rating: int = 0
    field_count: int = 0
    tables_used: List[str] = field(default_factory=list)
    estimated_performance: str = "UNKNOWN"
    business_logic: str = ""
    validation_rules: List[str] = field(default_factory=list)
    semantic_coherence: float = 0.0

@dataclass
class QueryResult:
    name: str
    description: str
    sql: str
    intelligence_score: float = 0.0
    complexity_rating: int = 0
    field_count: int = 0
    tables_used: List[str] = field(default_factory=list)
    estimated_performance: str = "UNKNOWN"

@dataclass
class MetricResult:
    metric_name: str
    query: IntelligentQuery
    results: List[Tuple]
    validation_confidence: float
    business_assessment: str
    extracted_values: Dict[str, Any] = field(default_factory=dict)
    alternative_queries: List[IntelligentQuery] = field(default_factory=list)

@dataclass
class AnalysisResults:
    total_fields: int = 0
    analyzed_fields: int = 0
    high_intelligence_fields: int = 0
    relationships_found: int = 0
    clusters_created: int = 0
    queries_generated: int = 0
    processing_time_seconds: float = 0.0
    success: bool = False
    error_message: str = ""
    semantic_coherence: float = 0.0

@dataclass
class FieldCombination:
    fields: List[FieldIntelligence]
    combination_type: str
    semantic_reasoning: str
    intelligence_basis: Dict[str, Any]
    expected_accuracy: float
    cross_field_validation: Dict[str, float] = field(default_factory=dict)

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
    semantic_coherence: float = 0.0