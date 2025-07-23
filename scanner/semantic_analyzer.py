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
        try:
            self.embedding_manager = EmbeddingManager()
            self.reasoning_engine = ReasoningEngine()
            self.semantic_memory = SemanticMemory()
            self.specialized_models = SpecializedModels()
            self.linguistic_resources = LinguisticResources()
            
            self.ao1_concepts = AO1_CONCEPTS
            self.analysis_cache = {}
            
            logger.info("AdvancedSemanticAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AdvancedSemanticAnalyzer: {e}")
            # Initialize with minimal fallback functionality
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize with minimal functionality if main init fails"""
        logger.warning("Using fallback semantic analyzer")
        self.embedding_manager = None
        self.reasoning_engine = None
        self.semantic_memory = None
        self.specialized_models = None
        self.linguistic_resources = None
        self.ao1_concepts = AO1_CONCEPTS
        self.analysis_cache = {}
    
    def analyze_batch_fields(self, table_data: List[Dict], schema_fields: List) -> List[FieldAnalysis]:
        """Analyze fields with comprehensive error handling"""
        try:
            field_analyses = []
            
            # Extract field samples with error handling
            field_samples = self._extract_field_samples_safe(table_data, schema_fields)
            
            if not field_samples:
                logger.warning("No field samples extracted")
                return []
            
            # Analyze table context
            table_context = self._analyze_table_context_safe(field_samples, schema_fields)
            
            # Process each field
            for field in schema_fields:
                try:
                    field_name = field.name
                    field_type = field.field_type
                    sample_values = field_samples.get(field_name, [])
                    
                    if not sample_values:
                        continue
                    
                    analysis = self._analyze_single_field_safe(field_name, field_type, sample_values, table_context)
                    
                    if analysis and analysis.confidence_score > MIN_CONFIDENCE_SCORE:
                        field_analyses.append(analysis)
                        
                        # Update learning if available
                        if self.semantic_memory:
                            try:
                                self._update_learning(analysis, table_context)
                            except Exception as e:
                                logger.debug(f"Learning update failed: {e}")
                
                except Exception as e:
                    logger.warning(f"Failed to analyze field {field.name}: {e}")
                    continue
            
            # Apply cross-field reasoning if possible
            try:
                if field_analyses and self.reasoning_engine:
                    field_analyses = self._apply_cross_field_reasoning(field_analyses, table_context)
            except Exception as e:
                logger.debug(f"Cross-field reasoning failed: {e}")
            
            logger.info(f"Successfully analyzed {len(field_analyses)} fields")
            return field_analyses
            
        except Exception as e:
            logger.error(f"Batch field analysis failed: {e}")
            return []
    
    def _extract_field_samples_safe(self, table_data: List[Dict], schema_fields: List) -> Dict[str, List[str]]:
        """Extract field samples with comprehensive error handling"""
        field_samples = {}
        
        try:
            for field in schema_fields:
                field_name = field.name
                field_samples[field_name] = []
                
                for row in table_data:
                    try:
                        value = None
                        
                        # Try different access methods
                        if hasattr(row, field_name):
                            value = getattr(row, field_name)
                        elif isinstance(row, dict) and field_name in row:
                            value = row[field_name]
                        elif hasattr(row, '_fields') and field_name in row._fields:
                            value = getattr(row, field_name)
                        
                        if value is not None:
                            value_str = str(value).strip()
                            if value_str and value_str.lower() not in ['null', 'none', '']:
                                field_samples[field_name].append(value_str)
                                
                                # Limit samples per field to avoid memory issues
                                if len(field_samples[field_name]) >= 10:
                                    break
                    
                    except Exception as e:
                        logger.debug(f"Error extracting value for {field_name}: {e}")
                        continue
                
                # Remove empty sample lists
                if not field_samples[field_name]:
                    del field_samples[field_name]
        
        except Exception as e:
            logger.error(f"Field sample extraction failed: {e}")
        
        return field_samples
    
    def _analyze_table_context_safe(self, field_samples: Dict[str, List[str]], schema_fields: List) -> TableContext:
        """Analyze table context with error handling"""
        try:
            field_names = list(field_samples.keys())
            
            # Simple theme detection
            semantic_themes = self._detect_semantic_themes_simple(field_names, field_samples)
            domain_indicators = self._detect_domain_indicators_simple(field_names, field_samples)
            
            return TableContext(
                field_names=field_names,
                semantic_themes=semantic_themes,
                domain_indicators=domain_indicators,
                structural_patterns={'field_count': len(field_names)},
                data_characteristics={'completeness': 0.8},
                confidence_signals=[]
            )
        
        except Exception as e:
            logger.warning(f"Table context analysis failed: {e}")
            return TableContext(
                field_names=list(field_samples.keys()),
                semantic_themes=[],
                domain_indicators=[],
                structural_patterns={},
                data_characteristics={},
                confidence_signals=[]
            )
    
    def _detect_semantic_themes_simple(self, field_names: List[str], field_samples: Dict[str, List[str]]) -> List[str]:
        """Simple theme detection"""
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
    
    def _detect_domain_indicators_simple(self, field_names: List[str], field_samples: Dict[str, List[str]]) -> List[str]:
        """Simple domain indicator detection"""
        indicators = []
        all_text = ' '.join(field_names).lower()
        
        domain_patterns = {
            'cybersecurity': ['security', 'agent', 'edr', 'siem', 'threat'],
            'it_infrastructure': ['server', 'network', 'infrastructure', 'system'],
            'data_management': ['log', 'data', 'source', 'collection', 'ingest']
        }
        
        for domain, patterns in domain_patterns.items():
            if any(pattern in all_text for pattern in patterns):
                indicators.append(domain)
        
        return indicators
    
    def _analyze_single_field_safe(self, field_name: str, field_type: str, sample_values: List[str], 
                                  table_context: TableContext) -> FieldAnalysis:
        """Analyze single field with comprehensive error handling"""
        try:
            # Use embedding manager if available
            if self.embedding_manager:
                try:
                    field_embedding = self.embedding_manager.get_field_embedding(field_name, sample_values, table_context.__dict__)
                    
                    # Compute semantic scores
                    concept_scores = {}
                    for concept in self.ao1_concepts.keys():
                        try:
                            semantic_score = self.embedding_manager.compute_semantic_similarity(field_embedding, concept)
                            
                            # Add simple pattern matching
                            pattern_score = self._compute_pattern_score_simple(sample_values, concept)
                            
                            # Simple combination
                            combined_score = (semantic_score * 0.7 + pattern_score * 0.3) * self.ao1_concepts[concept]['weight']
                            
                            if combined_score > 0.1:
                                concept_scores[concept] = combined_score
                        
                        except Exception as e:
                            logger.debug(f"Failed to compute score for {concept}: {e}")
                            continue
                    
                    if concept_scores:
                        best_concept = max(concept_scores.items(), key=lambda x: x[1])
                        final_confidence = best_concept[1]
                        explanation = f"Semantic analysis: {best_concept[0]} with confidence {final_confidence:.3f}"
                    else:
                        best_concept = ('unknown', 0.0)
                        final_confidence = 0.0
                        explanation = "No semantic matches found"
                
                except Exception as e:
                    logger.warning(f"Embedding analysis failed for {field_name}: {e}")
                    # Fall back to simple analysis
                    best_concept, final_confidence, explanation = self._simple_field_analysis(field_name, sample_values)
            else:
                # Use simple analysis
                best_concept, final_confidence, explanation = self._simple_field_analysis(field_name, sample_values)
            
            # Extract value patterns
            value_patterns = self._extract_value_patterns_simple(sample_values)
            
            return FieldAnalysis(
                field_name=field_name,
                field_type=field_type,
                ao1_category=best_concept[0] if isinstance(best_concept, tuple) else best_concept,
                confidence_score=final_confidence,
                semantic_evidence=[explanation],
                sample_values=sample_values[:5],
                value_patterns=value_patterns,
                reasoning_explanation=explanation,
                alternative_classifications=[]
            )
        
        except Exception as e:
            logger.error(f"Single field analysis failed for {field_name}: {e}")
            return None
    
    def _simple_field_analysis(self, field_name: str, sample_values: List[str]) -> tuple:
        """Simple field analysis based on field name and patterns"""
        field_lower = field_name.lower()
        
        # Simple rule-based matching
        if any(term in field_lower for term in ['host', 'device', 'machine', 'asset']):
            return 'asset_identity', 0.8, 'Field name suggests asset identity'
        elif any(term in field_lower for term in ['ip', 'network', 'dns', 'url']):
            return 'network_identity', 0.8, 'Field name suggests network identity'
        elif any(term in field_lower for term in ['agent', 'security', 'tool']):
            return 'security_tools', 0.8, 'Field name suggests security tools'
        elif any(term in field_lower for term in ['log', 'event', 'audit']):
            return 'log_sources', 0.8, 'Field name suggests log sources'
        elif any(term in field_lower for term in ['country', 'region', 'location']):
            return 'geographic_data', 0.8, 'Field name suggests geographic data'
        else:
            return 'unknown', 0.3, 'No clear pattern match'
    
    def _compute_pattern_score_simple(self, sample_values: List[str], concept: str) -> float:
        """Simple pattern score computation"""
        if not sample_values:
            return 0.0
        
        concept_patterns = self.ao1_concepts.get(concept, {}).get('value_patterns', [])
        if not concept_patterns:
            return 0.5
        
        matches = 0
        for value in sample_values:
            value_str = str(value).lower()
            for pattern in concept_patterns:
                try:
                    import re
                    if re.search(pattern, value_str, re.IGNORECASE):
                        matches += 1
                        break
                except:
                    continue
        
        return matches / len(sample_values) if sample_values else 0.0
    
    def _extract_value_patterns_simple(self, sample_values: List[str]) -> List[str]:
        """Simple value pattern extraction"""
        patterns = []
        
        if not sample_values:
            return patterns
        
        # Simple pattern detection
        for value in sample_values[:5]:
            value_str = str(value).lower()
            if '.' in value_str and len(value_str.split('.')) == 4:
                patterns.append('ipv4_pattern')
            elif '@' in value_str:
                patterns.append('email_pattern')
            elif 'http' in value_str:
                patterns.append('url_pattern')
            elif value_str.isdigit():
                patterns.append('numeric_pattern')
        
        return list(set(patterns))
    
    def _apply_cross_field_reasoning(self, field_analyses: List[FieldAnalysis], 
                                   table_context: TableContext) -> List[FieldAnalysis]:
        """Apply simple cross-field reasoning"""
        try:
            # Simple consistency boost
            concept_counts = defaultdict(int)
            for analysis in field_analyses:
                concept_counts[analysis.ao1_category] += 1
            
            # Boost confidence for concepts that appear multiple times
            for analysis in field_analyses:
                concept = analysis.ao1_category
                if concept_counts[concept] > 1:
                    boost = min(0.1, concept_counts[concept] * 0.03)
                    analysis.confidence_score = min(1.0, analysis.confidence_score + boost)
                    analysis.semantic_evidence.append(f"cross_field_consistency_boost({boost:.3f})")
            
            return field_analyses
        
        except Exception as e:
            logger.warning(f"Cross-field reasoning failed: {e}")
            return field_analyses
    
    def _update_learning(self, analysis: FieldAnalysis, table_context: TableContext):
        """Update learning with error handling"""
        try:
            if self.semantic_memory:
                self.semantic_memory.learn_from_analysis(
                    analysis.field_name,
                    analysis.ao1_category,
                    analysis.confidence_score,
                    analysis.sample_values,
                    table_context.__dict__
                )
        except Exception as e:
            logger.debug(f"Learning update failed: {e}")
    
    def find_relationships(self, tables_analysis: Dict[str, List[FieldAnalysis]]) -> List[Dict]:
        """Find relationships with error handling"""
        try:
            relationships = []
            
            # Simple relationship finding
            concept_tables = defaultdict(list)
            for table_name, analyses in tables_analysis.items():
                for analysis in analyses:
                    if analysis.confidence_score > 0.5:
                        concept_tables[analysis.ao1_category].append((table_name, analysis))
            
            # Build simple relationships
            for concept, table_analyses in concept_tables.items():
                if len(table_analyses) > 1:
                    for i, (table1, analysis1) in enumerate(table_analyses):
                        for table2, analysis2 in table_analyses[i+1:]:
                            if table1 != table2:
                                similarity = 0.8  # Simple fixed similarity
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
        
        except Exception as e:
            logger.error(f"Relationship finding failed: {e}")
            return []

# Alias for backward compatibility
SemanticAnalyzer = AdvancedSemanticAnalyzer