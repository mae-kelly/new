#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class SemanticProfile:
    field_name: str
    entropy_metrics: Dict[str, float] = field(default_factory=dict)
    pattern_features: Dict[str, List[float]] = field(default_factory=dict)
    statistical_properties: Dict[str, float] = field(default_factory=dict)
    behavioral_indicators: Dict[str, float] = field(default_factory=dict)
    semantic_density: float = 0.0
    complexity_score: float = 0.0

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
    
    def get_key(self) -> str:
        return f"{self.table}.{self.name}"

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