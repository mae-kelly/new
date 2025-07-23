import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from fuzzywuzzy import fuzz
from ..nlp import SpecializedModels
from ..config import AO1_CONCEPTS
from .types import FieldAnalysis, TableContext

logger = logging.getLogger(__name__)

class ReasoningEngine:
    def __init__(self):
        self.specialized_models = SpecializedModels()
        self.concept_relationships = self._build_concept_relationships()
        self.reasoning_templates = self._load_reasoning_templates()
    
    def _build_concept_relationships(self):
        relationships = {
            'supports': defaultdict(list),
            'conflicts': defaultdict(list),
            'requires': defaultdict(list),
            'implies': defaultdict(list)
        }
        
        relationships['supports']['asset_identity'].extend(['network_identity', 'system_classification'])
        relationships['supports']['security_tools'].extend(['log_sources', 'coverage_metrics'])
        relationships['supports']['network_identity'].extend(['asset_identity', 'geographic_data'])
        
        relationships['conflicts']['asset_identity'].extend(['log_sources'])
        relationships['conflicts']['coverage_metrics'].extend(['business_context'])
        
        return relationships
    
    def _load_reasoning_templates(self):
        return {
            'high_confidence': "Strong semantic alignment with {concept} based on {evidence}",
            'medium_confidence': "Moderate alignment with {concept}, supported by {evidence}",
            'low_confidence': "Weak alignment with {concept}, limited evidence: {evidence}",
            'alternative': "Could also be {concept} with confidence {score:.2f}"
        }
    
    def perform_multi_step_reasoning(self, field_name: str, sample_values: List[str], 
                                   concept_scores: Dict[str, float], table_context: TableContext) -> Tuple[str, float, str]:
        reasoning_steps = []
        
        # Step 1: Initial concept ranking
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        reasoning_steps.append(f"Initial ranking: {', '.join([f'{c}({s:.2f})' for c, s in sorted_concepts[:3]])}")
        
        if not sorted_concepts:
            return 'unknown', 0.0, "No semantic matches found"
        
        best_concept, best_score = sorted_concepts[0]
        
        # Step 2: Contextual validation
        contextual_boost = self._validate_contextual_alignment(best_concept, table_context)
        if contextual_boost > 0:
            best_score *= (1 + contextual_boost)
            reasoning_steps.append(f"Contextual validation boosted confidence by {contextual_boost:.2f}")
        
        # Step 3: Cross-field relationship analysis
        relationship_score = self._analyze_field_relationships(field_name, best_concept, table_context)
        if relationship_score != 0:
            best_score *= (1 + relationship_score)
            reasoning_steps.append(f"Cross-field relationships {'support' if relationship_score > 0 else 'conflict'} with score {relationship_score:.2f}")
        
        # Step 4: Pattern consistency check
        pattern_consistency = self._check_pattern_consistency(sample_values, best_concept)
        best_score *= pattern_consistency
        reasoning_steps.append(f"Pattern consistency score: {pattern_consistency:.2f}")
        
        # Step 5: Domain expertise validation
        domain_validation = self._apply_domain_expertise(field_name, sample_values, best_concept)
        best_score *= domain_validation
        reasoning_steps.append(f"Domain expertise validation: {domain_validation:.2f}")
        
        # Step 6: Uncertainty estimation
        uncertainty = self._estimate_uncertainty(best_score, sorted_concepts)
        final_confidence = best_score * (1 - uncertainty)
        reasoning_steps.append(f"Uncertainty adjustment: -{uncertainty:.2f}")
        
        # Generate explanation
        explanation = self._generate_explanation(field_name, best_concept, final_confidence, reasoning_steps)
        
        return best_concept, min(final_confidence, 1.0), explanation
    
    def _validate_contextual_alignment(self, concept: str, table_context: TableContext) -> float:
        boost = 0.0
        
        concept_theme_map = {
            'asset_identity': ['asset', 'infrastructure', 'inventory', 'device'],
            'network_identity': ['network', 'connectivity', 'addressing'],
            'security_tools': ['security', 'monitoring', 'protection'],
            'log_sources': ['logging', 'audit', 'monitoring'],
            'geographic_data': ['location', 'geography', 'facility'],
            'coverage_metrics': ['metrics', 'measurement', 'analysis']
        }
        
        if concept in concept_theme_map:
            expected_themes = concept_theme_map[concept]
            actual_themes = table_context.semantic_themes
            
            theme_overlap = len(set(expected_themes) & set(actual_themes))
            if theme_overlap > 0:
                boost = min(theme_overlap * 0.15, 0.3)
        
        return boost
    
    def _analyze_field_relationships(self, field_name: str, concept: str, table_context: TableContext) -> float:
        score = 0.0
        
        # Analyze field name patterns in the table
        related_fields = [f for f in table_context.field_names if f != field_name]
        
        for related_field in related_fields:
            relationship_strength = self._compute_field_relationship_strength(field_name, related_field, concept)
            if relationship_strength > 0.5:
                score += 0.1
            elif relationship_strength < -0.5:
                score -= 0.1
        
        return max(-0.3, min(0.3, score))
    
    def _compute_field_relationship_strength(self, field1: str, field2: str, concept: str) -> float:
        # Simple relationship analysis based on field names
        field1_lower = field1.lower()
        field2_lower = field2.lower()
        
        supporting_patterns = {
            'asset_identity': ['host', 'device', 'machine', 'asset'],
            'network_identity': ['ip', 'network', 'dns', 'url'],
            'security_tools': ['agent', 'security', 'tool', 'sensor']
        }
        
        if concept in supporting_patterns:
            patterns = supporting_patterns[concept]
            field1_matches = sum(1 for p in patterns if p in field1_lower)
            field2_matches = sum(1 for p in patterns if p in field2_lower)
            
            if field1_matches > 0 and field2_matches > 0:
                return 0.7
            elif field1_matches > 0 or field2_matches > 0:
                return 0.3
        
        return 0.0
    
    def _check_pattern_consistency(self, sample_values: List[str], concept: str) -> float:
        if not sample_values:
            return 0.8
        
        concept_patterns = AO1_CONCEPTS.get(concept, {}).get('value_patterns', [])
        
        if not concept_patterns:
            return 0.8
        
        matches = 0
        total_values = len(sample_values)
        
        for value in sample_values:
            value_str = str(value)
            for pattern in concept_patterns:
                try:
                    import re
                    if re.search(pattern, value_str, re.IGNORECASE):
                        matches += 1
                        break
                except:
                    continue
        
        consistency_ratio = matches / total_values if total_values > 0 else 0
        return 0.5 + (consistency_ratio * 0.5)
    
    def _apply_domain_expertise(self, field_name: str, sample_values: List[str], concept: str) -> float:
        domain_rules = {
            'asset_identity': {
                'field_indicators': ['host', 'device', 'machine', 'asset', 'computer'],
                'value_indicators': ['hostname', 'server', 'workstation'],
                'anti_indicators': ['log', 'event', 'count']
            },
            'network_identity': {
                'field_indicators': ['ip', 'network', 'dns', 'url', 'address'],
                'value_indicators': ['http', 'https', '.com', '.local'],
                'anti_indicators': ['agent', 'tool']
            },
            'security_tools': {
                'field_indicators': ['agent', 'tool', 'security', 'sensor'],
                'value_indicators': ['crowdstrike', 'splunk', 'agent'],
                'anti_indicators': ['hostname', 'address']
            }
        }
        
        if concept not in domain_rules:
            return 1.0
        
        rules = domain_rules[concept]
        score = 1.0
        
        field_lower = field_name.lower()
        
        # Positive indicators
        field_matches = sum(1 for indicator in rules['field_indicators'] if indicator in field_lower)
        if field_matches > 0:
            score *= 1.2
        
        # Value indicators
        if sample_values:
            value_text = ' '.join(str(v).lower() for v in sample_values[:5])
            value_matches = sum(1 for indicator in rules['value_indicators'] if indicator in value_text)
            if value_matches > 0:
                score *= 1.1
        
        # Anti-indicators
        anti_matches = sum(1 for indicator in rules['anti_indicators'] if indicator in field_lower)
        if anti_matches > 0:
            score *= 0.8
        
        return max(0.5, min(1.5, score))
    
    def _estimate_uncertainty(self, best_score: float, sorted_concepts: List[Tuple[str, float]]) -> float:
        if len(sorted_concepts) < 2:
            return 0.1
        
        # Calculate uncertainty based on score distribution
        scores = [score for _, score in sorted_concepts]
        
        if len(scores) > 1:
            score_gap = scores[0] - scores[1]
            if score_gap < 0.1:
                return 0.3  # High uncertainty
            elif score_gap < 0.2:
                return 0.2  # Medium uncertainty
            else:
                return 0.1  # Low uncertainty
        
        return 0.1
    
    def _generate_explanation(self, field_name: str, concept: str, confidence: float, reasoning_steps: List[str]) -> str:
        concept_descriptions = {
            'asset_identity': "identifies computing assets and devices",
            'network_identity': "represents network addressing and connectivity",
            'security_tools': "relates to cybersecurity monitoring and protection",
            'log_sources': "contains logging and audit information",
            'geographic_data': "indicates location or geographic information",
            'business_context': "provides organizational or business context",
            'infrastructure_type': "describes infrastructure and platform types",
            'system_classification': "categorizes systems and device types",
            'coverage_metrics': "measures coverage or performance metrics"
        }
        
        description = concept_descriptions.get(concept, f"relates to {concept}")
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        
        explanation = f"Field '{field_name}' most likely {description} with {confidence_level} confidence ({confidence:.3f}). "
        
        if reasoning_steps:
            key_steps = [step for step in reasoning_steps if any(indicator in step.lower() 
                        for indicator in ['boost', 'support', 'validation'])]
            if key_steps:
                explanation += f"Key reasoning: {'; '.join(key_steps[:2])}"
        
        return explanation