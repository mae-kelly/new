import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class SemanticVector:
    base_embedding: np.ndarray
    contextual_embedding: np.ndarray
    linguistic_features: Dict[str, Any]
    semantic_fingerprint: str
    confidence_distribution: Dict[str, float]
    reasoning_chain: List[str]

@dataclass
class FieldAnalysis:
    field_name: str
    field_type: str
    ao1_category: str
    confidence_score: float
    semantic_evidence: List[str]
    sample_values: List[str]
    value_patterns: List[str]
    semantic_vector: Optional[SemanticVector] = None
    reasoning_explanation: str = ""
    alternative_classifications: List[Tuple[str, float]] = None

@dataclass
class ConceptEmbedding:
    concept_name: str
    base_embedding: np.ndarray
    contextual_embedding: np.ndarray
    domain_embedding: np.ndarray
    linguistic_profile: Dict[str, Any]

@dataclass
class TableContext:
    field_names: List[str]
    semantic_themes: List[str]
    domain_indicators: List[str]
    structural_patterns: Dict[str, Any]
    data_characteristics: Dict[str, Any]
    confidence_signals: List[str]