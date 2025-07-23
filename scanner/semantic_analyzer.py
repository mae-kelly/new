import logging
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from .nlp import FoundationModels, SpecializedModels, LinguisticResources
from .semantic import EmbeddingManager, ReasoningEngine, SemanticMemory, FieldAnalysis, TableContext
from .config import AO1_CONCEPTS, MIN_CONFIDENCE_SCORE

logger = logging.getLogger(__name__)

class AdvancedSemanticAnalyzer:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.reasoning_engine = ReasoningEngine()
        self.semantic_memory = SemanticMemory()
        self.specialized_models = SpecializedModels()
        self.linguistic_resources = LinguisticResources()
        
        self.ao1_concepts = AO1_CONCEPTS
        self.analysis_cache = {}
    
    def analyze_batch_fields(self, table_data: List[Dict], schema_fields: List) -> List[FieldAnalysis]:
        field_analyses = []
        
        # Extract field samples
        field_samples = self._extract_field_samples(table_data, schema_fields)
        
        # Analyze table context
        table_context = self._analyze_table_context(field_samples, schema_fields)
        
        # Process each field
        for field in schema_fields:
            field_name = field.name
            field_type = field.field_type
            sample_values = field_samples.get(field_name, [])
            
            analysis = self._analyze_single_field(field_name, field_type, sample_values, table_context)
            
            if analysis.confidence_score > MIN_CONFIDENCE_SCORE:
                field_analyses.append(analysis)
                self._update_learning(analysis, table_context)
        
        # Apply cross-field reasoning
        field_analyses = self._apply_cross_field_reasoning(field_analyses, table_context)
        
        return field_analyses
    
    def _extract_field_samples(self, table_data: List[Dict], schema_fields: List) -> Dict[str, List[str]]:
        field_samples = {}
        
        for field in schema_fields:
            field_name = field.name
            field_samples[field_name] = []
            
            for row in table_data:
                try:
                    value = None
                    if hasattr(row, field_name):
                        value = getattr(row, field_name)
                    elif isinstance(row, dict) and field_name in row:
                        value = row[field_name]
                    
                    if value is not None and str(value).strip():
                        field_samples[field_name].append(str(value).strip())
                except Exception:
                    continue
        
        return field_samples
    
    def _analyze_table_context(self, field_samples: Dict[str, List[str]], schema_fields: List) -> TableContext:
        field_names = list(field_samples.keys())
        
        # Detect semantic themes
        semantic_themes = self._detect_semantic_themes(field_names, field_samples)
        
        # Detect domain indicators
        domain_indicators = self._detect_domain_indicators(field_names, field_samples)
        
        # Analyze structural patterns
        structural_patterns = self._analyze_structural_patterns(field_samples)
        
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(field_samples)
        
        return TableContext(
            field_names=field_names,
            semantic_themes=semantic_themes,
            domain_indicators=domain_indicators,
            structural_patterns=structural_patterns,
            data_characteristics=data_characteristics,
            confidence_signals=[]
        )
    
    def _detect_semantic_themes(self, field_names: List[str], field_samples: Dict[str, List[str]]) -> List[str]:
        themes = []
        
        all_field_text = ' '.join(field_names).lower()
        
        theme_indicators = {
            'asset_management': ['host', 'asset', 'device', 'machine', 'computer', 'server'],
            'network_infrastructure': ['ip', 'network', 'dns', 'url', 'address', 'domain'],
            'security_monitoring': ['agent', 'security', 'crowdstrike', 'edr', 'sensor'],
            'logging_audit': ['log', 'event', 'audit', 'source', 'siem'],
            'geographic_location': ['country', 'region', 'location', 'site', 'datacenter'],
            'business_organization': ['business', 'unit', 'department', 'owner', 'application']
        }
        
        for theme, indicators in theme_indicators.items():
            if any(indicator in all_field_text for indicator in indicators):
                themes.append(theme)
        
        return themes
    
    def _detect_domain_indicators(self, field_names: List[str], field_samples: Dict[str, List[str]]) -> List[str]:
        indicators = []
        
        domain_patterns = {
            'cybersecurity': ['security', 'agent', 'edr', 'siem', 'threat'],
            'it_infrastructure': ['server', 'network', 'infrastructure', 'system'],
            'data_management': ['log', 'data', 'source', 'collection', 'ingest'],
            'identity_management': ['user', 'identity', 'authentication', 'access'],
            'compliance_audit': ['audit', 'compliance', 'policy', 'governance']
        }
        
        all_text = ' '.join(field_names + [v for values in field_samples.values() for v in values[:3]]).lower()
        
        for domain, patterns in domain_patterns.items():
            if any(pattern in all_text for pattern in patterns):
                indicators.append(domain)
        
        return indicators
    
    def _analyze_structural_patterns(self, field_samples: Dict[str, List[str]]) -> Dict[str, Any]:
        patterns = {
            'field_count': len(field_samples),
            'avg_values_per_field': sum(len(values) for values in field_samples.values()) / max(len(field_samples), 1),
            'id_fields': [],
            'categorical_fields': [],
            'numeric_fields': []
        }
        
        for field_name, values in field_samples.items():
            if not values:
                continue
            
            # Detect ID fields
            if any(term in field_name.lower() for term in ['id', 'identifier', 'key']):
                patterns['id_fields'].append(field_name)
            
            # Detect categorical fields
            unique_ratio = len(set(values)) / len(values) if values else 0
            if unique_ratio < 0.5 and len(set(values)) < 20:
                patterns['categorical_fields'].append(field_name)
            
            # Detect numeric fields
            numeric_count = sum(1 for v in values if str(v).replace('.', '').replace('-', '').isdigit())
            if numeric_count / len(values) > 0.8:
                patterns['numeric_fields'].append(field_name)
        
        return patterns
    
    def _analyze_data_characteristics(self, field_samples: Dict[str, List[str]]) -> Dict[str, Any]:
        characteristics = {
            'data_density': 0.0,
            'completeness': 0.0,
            'value_diversity': 0.0,
            'pattern_consistency': 0.0
        }
        
        total_fields = len(field_samples)
        if total_fields == 0:
            return characteristics
        
        # Calculate data density
        non_empty_fields = sum(1 for values in field_samples.values() if values)
        characteristics['data_density'] = non_empty_fields / total_fields
        
        # Calculate completeness
        total_values = sum(len(values) for values in field_samples.values())
        characteristics['completeness'] = total_values / (total_fields * 20)  # Assuming max 20 samples per field
        
        # Calculate value diversity
        diversities = []
        for values in field_samples.values():
            if values:
                unique_ratio = len(set(values)) / len(values)
                diversities.append(unique_ratio)
        characteristics['value_diversity'] = sum(diversities) / len(diversities) if diversities else 0
        
        return characteristics
    
    def _analyze_single_field(self, field_name: str, field_type: str, sample_values: List[str], 
                             table_context: TableContext) -> FieldAnalysis:
        # Get field embedding
        field_embedding = self.embedding_manager.get_field_embedding(field_name, sample_values, table_context.__dict__)
        
        # Compute semantic scores for each concept
        concept_scores = {}
        for concept in self.ao1_concepts.keys():
            # Base semantic similarity
            semantic_score = self.embedding_manager.compute_semantic_similarity(field_embedding, concept)
            
            # Add learned patterns score
            learned_score = self.semantic_memory.get_learned_score(field_name, concept, table_context.__dict__)
            
            # Add linguistic analysis score
            linguistic_score = self._compute_linguistic_score(field_name, sample_values, concept)
            
            # Add pattern matching score
            pattern_score = self._compute_pattern_score(sample_values, concept)
            
            # Combine scores
            combined_score = (
                semantic_score * 0.4 +
                learned_score * 0.25 +
                linguistic_score * 0.2 +
                pattern_score * 0.15
            ) * self.ao1_concepts[concept]['weight']
            
            if combined_score > 0.1:
                concept_scores[concept] = combined_score
        
        # Apply multi-step reasoning
        if concept_scores:
            best_concept, final_confidence, explanation = self.reasoning_engine.perform_multi_step_reasoning(
                field_name, sample_values, concept_scores, table_context
            )
        else:
            best_concept, final_confidence, explanation = 'unknown', 0.0, "No semantic matches found"
        
        # Calibrate confidence
        calibrated_confidence = self.semantic_memory.get_calibrated_confidence(best_concept, final_confidence)
        
        # Extract value patterns
        value_patterns = self._extract_value_patterns(sample_values, best_concept)
        
        # Generate alternatives
        alternatives = [(concept, score) for concept, score in concept_scores.items() 
                       if concept != best_concept and score > calibrated_confidence * 0.3]
        alternatives = sorted(alternatives, key=lambda x: x[1], reverse=True)[:3]
        
        return FieldAnalysis(
            field_name=field_name,
            field_type=field_type,
            ao1_category=best_concept,
            confidence_score=calibrated_confidence,
            semantic_evidence=[f"reasoning: {explanation}"],
            sample_values=sample_values[:5],
            value_patterns=value_patterns,
            reasoning_explanation=explanation,
            alternative_classifications=alternatives
        )
    
    def _compute_linguistic_score(self, field_name: str, sample_values: List[str], concept: str) -> float:
        score = 0.0
        
        # Analyze field name linguistically
        field_features = self.linguistic_resources.analyze_linguistic_features(field_name)
        
        # Check POS tag alignment
        concept_pos_preferences = {
            'asset_identity': ['NOUN'],
            'network_identity': ['NOUN', 'ADJ'],
            'security_tools': ['NOUN', 'VERB'],
            'log_sources': ['NOUN'],
            'geographic_data': ['NOUN', 'PROPN'],
            'coverage_metrics': ['NOUN', 'NUM']
        }
        
        if concept in concept_pos_preferences:
            pos_distribution = field_features.get('pos_distribution', {})
            preferred_pos = concept_pos_preferences[concept]
            
            matching_pos = sum(pos_distribution.get(pos, 0) for pos in preferred_pos)
            total_pos = sum(pos_distribution.values())
            
            if total_pos > 0:
                score += (matching_pos / total_pos) * 0.6
        
        # Check entity alignment
        entities = field_features.get('entities', [])
        concept_entity_preferences = {
            'asset_identity': ['ORG', 'MISC'],
            'network_identity': ['MISC'],
            'security_tools': ['ORG'],
            'geographic_data': ['LOC', 'GPE']
        }
        
        if concept in concept_entity_preferences and entities:
            preferred_entities = concept_entity_preferences[concept]
            matching_entities = sum(1 for _, entity_type in entities if entity_type in preferred_entities)
            score += (matching_entities / len(entities)) * 0.4
        
        return min(1.0, score)
    
    def _compute_pattern_score(self, sample_values: List[str], concept: str) -> float:
        if not sample_values:
            return 0.0
        
        concept_patterns = self.ao1_concepts.get(concept, {}).get('value_patterns', [])
        if not concept_patterns:
            return 0.5
        
        matches = 0
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
        
        return matches / len(sample_values) if sample_values else 0.0
    
    def _extract_value_patterns(self, sample_values: List[str], concept: str) -> List[str]:
        patterns = []
        
        if not sample_values:
            return patterns
        
        # Common patterns
        pattern_detectors = {
            'ipv4_address': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            'fqdn': r'^[a-zA-Z0-9\-\.]+\.(com|local|corp|internal|net|org)$',
            'uuid': r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://[^\s]+$',
            'numeric_id': r'^\d+$',
            'alphanumeric_id': r'^[a-zA-Z0-9\-]+\d+$',
            'crowdstrike_ref': r'.*crowdstrike.*|.*falcon.*',
            'chronicle_ref': r'.*chronicle.*',
            'splunk_ref': r'.*splunk.*'
        }
        
        import re
        for pattern_name, pattern_regex in pattern_detectors.items():
            matches = sum(1 for value in sample_values 
                         if re.search(pattern_regex, str(value), re.IGNORECASE))
            if matches > len(sample_values) * 0.2:
                patterns.append(pattern_name)
        
        return patterns
    
    def _apply_cross_field_reasoning(self, field_analyses: List[FieldAnalysis], 
                                   table_context: TableContext) -> List[FieldAnalysis]:
        # Build concept distribution
        concept_counts = defaultdict(int)
        for analysis in field_analyses:
            concept_counts[analysis.ao1_category] += 1
        
        # Apply consistency boost/penalty
        for analysis in field_analyses:
            concept = analysis.ao1_category
            
            # Boost confidence for concepts that appear multiple times
            if concept_counts[concept] > 1:
                boost = min(0.1, concept_counts[concept] * 0.03)
                analysis.confidence_score = min(1.0, analysis.confidence_score + boost)
                analysis.semantic_evidence.append(f"cross_field_consistency_boost({boost:.3f})")
        
        return field_analyses
    
    def _update_learning(self, analysis: FieldAnalysis, table_context: TableContext):
        self.semantic_memory.learn_from_analysis(
            analysis.field_name,
            analysis.ao1_category,
            analysis.confidence_score,
            analysis.sample_values,
            table_context.__dict__
        )
    
    def find_relationships(self, tables_analysis: Dict[str, List[FieldAnalysis]]) -> List[Dict]:
        relationships = []
        
        # Find tables with same concepts
        concept_tables = defaultdict(list)
        for table_name, analyses in tables_analysis.items():
            for analysis in analyses:
                if analysis.confidence_score > 0.5:
                    concept_tables[analysis.ao1_category].append((table_name, analysis))
        
        # Build relationships
        for concept, table_analyses in concept_tables.items():
            if len(table_analyses) > 1:
                for i, (table1, analysis1) in enumerate(table_analyses):
                    for table2, analysis2 in table_analyses[i+1:]:
                        if table1 != table2:
                            similarity = self._compute_field_similarity(analysis1, analysis2)
                            if similarity > 0.6:
                                relationships.append({
                                    'table1': table1,
                                    'table2': table2,
                                    'field1': analysis1.field_name,
                                    'field2': analysis2.field_name,
                                    'concept': concept,
                                    'similarity': similarity,
                                    'confidence': min(analysis1.confidence_score, analysis2.confidence_score)
                                })
        
        return relationships
    
    def _compute_field_similarity(self, analysis1: FieldAnalysis, analysis2: FieldAnalysis) -> float:
        from fuzzywuzzy import fuzz
        
        # Name similarity
        name_sim = fuzz.token_sort_ratio(analysis1.field_name, analysis2.field_name) / 100.0
        
        # Pattern similarity
        patterns1 = set(analysis1.value_patterns)
        patterns2 = set(analysis2.value_patterns)
        pattern_sim = len(patterns1 & patterns2) / max(len(patterns1 | patterns2), 1)
        
        # Confidence similarity
        conf_sim = min(analysis1.confidence_score, analysis2.confidence_score)
        
        return (name_sim * 0.4 + pattern_sim * 0.4 + conf_sim * 0.2)

# Alias for backward compatibility
SemanticAnalyzer = AdvancedSemanticAnalyzer