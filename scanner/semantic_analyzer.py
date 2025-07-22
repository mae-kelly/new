import re
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

from .config import AO1_CONCEPTS, FUZZY_MATCH_THRESHOLD, SEMANTIC_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

@dataclass
class FieldAnalysis:
    field_name: str
    field_type: str
    ao1_category: str
    confidence_score: float
    semantic_evidence: List[str]
    sample_values: List[str]
    value_patterns: List[str]

class SemanticAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.ao1_concepts = AO1_CONCEPTS
        self._prepare_concept_vectors()
    
    def _prepare_concept_vectors(self):
        all_texts = []
        self.concept_mappings = {}
        
        for concept, data in self.ao1_concepts.items():
            combined_text = ' '.join(data['content_indicators'] + [p.replace('.*', '').replace(r'\.', '') for p in data['patterns']])
            all_texts.append(combined_text)
            self.concept_mappings[len(all_texts) - 1] = concept
        
        try:
            self.concept_vectors = self.vectorizer.fit_transform(all_texts)
        except Exception as e:
            logger.warning(f"Failed to create concept vectors: {e}")
            self.concept_vectors = None
    
    def analyze_field(self, field_name: str, field_type: str, sample_values: List[str]) -> FieldAnalysis:
        ao1_category, confidence, evidence = self._classify_field(field_name, field_type, sample_values)
        value_patterns = self._extract_value_patterns(sample_values)
        
        return FieldAnalysis(
            field_name=field_name,
            field_type=field_type,
            ao1_category=ao1_category,
            confidence_score=confidence,
            semantic_evidence=evidence,
            sample_values=sample_values[:10],
            value_patterns=value_patterns
        )
    
    def _classify_field(self, field_name: str, field_type: str, sample_values: List[str]) -> Tuple[str, float, List[str]]:
        scores = {}
        evidence = defaultdict(list)
        
        for concept, config in self.ao1_concepts.items():
            score = 0.0
            concept_evidence = []
            
            name_score = self._analyze_field_name(field_name, config)
            if name_score > 0:
                score += name_score * 0.4
                concept_evidence.append(f"field_name_match({name_score:.2f})")
            
            content_score = self._analyze_field_content(sample_values, config)
            if content_score > 0:
                score += content_score * 0.6
                concept_evidence.append(f"content_match({content_score:.2f})")
            
            if score > 0:
                scores[concept] = score * config['weight']
                evidence[concept] = concept_evidence
        
        if not scores:
            return 'unknown', 0.0, ['no_matches']
        
        best_concept = max(scores.keys(), key=lambda k: scores[k])
        return best_concept, scores[best_concept], evidence[best_concept]
    
    def _analyze_field_name(self, field_name: str, config: Dict) -> float:
        cleaned_name = self._clean_field_name(field_name)
        max_score = 0.0
        
        for pattern in config['patterns']:
            try:
                if re.search(pattern, cleaned_name, re.IGNORECASE):
                    max_score = max(max_score, 0.9)
            except re.error:
                continue
        
        for indicator in config['content_indicators']:
            fuzzy_score = fuzz.partial_ratio(cleaned_name.lower(), indicator.lower()) / 100.0
            if fuzzy_score >= FUZZY_MATCH_THRESHOLD:
                max_score = max(max_score, fuzzy_score)
        
        if self.concept_vectors is not None:
            semantic_score = self._semantic_similarity(cleaned_name, config['content_indicators'])
            max_score = max(max_score, semantic_score)
        
        return max_score
    
    def _analyze_field_content(self, sample_values: List[str], config: Dict) -> float:
        if not sample_values:
            return 0.0
        
        cleaned_values = [self._clean_value(v) for v in sample_values if v is not None]
        if not cleaned_values:
            return 0.0
        
        pattern_matches = 0
        total_patterns = len(config['value_patterns'])
        
        for pattern in config['value_patterns']:
            try:
                pattern_obj = re.compile(pattern, re.IGNORECASE)
                matches = sum(1 for value in cleaned_values if pattern_obj.search(str(value)))
                if matches > 0:
                    pattern_matches += matches / len(cleaned_values)
            except re.error:
                continue
        
        if total_patterns == 0:
            return 0.0
        
        content_indicators_score = 0.0
        for indicator in config['content_indicators']:
            indicator_matches = sum(1 for value in cleaned_values 
                                 if fuzz.partial_ratio(str(value).lower(), indicator.lower()) >= FUZZY_MATCH_THRESHOLD * 100)
            if indicator_matches > 0:
                content_indicators_score = max(content_indicators_score, indicator_matches / len(cleaned_values))
        
        return max(pattern_matches / total_patterns, content_indicators_score)
    
    def _semantic_similarity(self, text: str, indicators: List[str]) -> float:
        try:
            combined_indicators = ' '.join(indicators)
            texts = [text, combined_indicators]
            vectors = self.vectorizer.transform(texts)
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity if similarity >= SEMANTIC_SIMILARITY_THRESHOLD else 0.0
        except Exception:
            return 0.0
    
    def _clean_field_name(self, field_name: str) -> str:
        cleaned = re.sub(r'[_\-\.]', ' ', field_name)
        cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)
        return cleaned.strip()
    
    def _clean_value(self, value: Any) -> str:
        if value is None:
            return ''
        
        str_value = str(value).strip()
        if not str_value or str_value.lower() in ['null', 'none', 'n/a', '']:
            return ''
        
        return str_value
    
    def _extract_value_patterns(self, sample_values: List[str]) -> List[str]:
        if not sample_values:
            return []
        
        patterns = []
        cleaned_values = [self._clean_value(v) for v in sample_values if v]
        
        if not cleaned_values:
            return []
        
        common_patterns = [
            (r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', 'ipv4_address'),
            (r'^[a-zA-Z0-9\-\.]+\.(com|local|corp|internal|net|org)$', 'fqdn'),
            (r'^[a-zA-Z0-9\-]+\d+$', 'alphanumeric_with_number'),
            (r'^[A-Z]{2,3}$', 'country_code'),
            (r'.*[Aa]gent.*', 'agent_reference'),
            (r'.*[Ll]og.*', 'log_reference'),
            (r'.*[Hh]ost.*', 'host_reference')
        ]
        
        for pattern, name in common_patterns:
            try:
                pattern_obj = re.compile(pattern, re.IGNORECASE)
                matches = sum(1 for value in cleaned_values if pattern_obj.search(value))
                if matches > len(cleaned_values) * 0.3:
                    patterns.append(name)
            except re.error:
                continue
        
        value_lengths = [len(v) for v in cleaned_values if v]
        if value_lengths:
            avg_length = sum(value_lengths) / len(value_lengths)
            if avg_length > 20:
                patterns.append('long_text')
            elif avg_length < 5:
                patterns.append('short_code')
        
        return patterns
    
    def find_relationships(self, tables_analysis: Dict[str, List[FieldAnalysis]]) -> List[Dict]:
        relationships = []
        
        asset_tables = self._get_tables_by_category(tables_analysis, 'asset_identity')
        security_tables = self._get_tables_by_category(tables_analysis, 'security_tools')
        network_tables = self._get_tables_by_category(tables_analysis, 'network_identity')
        
        for asset_table in asset_tables:
            for security_table in security_tables:
                relationship = self._analyze_table_relationship(asset_table, security_table, tables_analysis)
                if relationship:
                    relationships.append(relationship)
        
        return relationships
    
    def _get_tables_by_category(self, tables_analysis: Dict[str, List[FieldAnalysis]], category: str) -> List[str]:
        matching_tables = []
        for table_name, fields in tables_analysis.items():
            for field in fields:
                if field.ao1_category == category and field.confidence_score > 0.7:
                    matching_tables.append(table_name)
                    break
        return matching_tables
    
    def _analyze_table_relationship(self, table1: str, table2: str, tables_analysis: Dict) -> Optional[Dict]:
        table1_fields = tables_analysis.get(table1, [])
        table2_fields = tables_analysis.get(table2, [])
        
        potential_joins = []
        
        for field1 in table1_fields:
            for field2 in table2_fields:
                if (field1.ao1_category == field2.ao1_category and 
                    field1.confidence_score > 0.6 and field2.confidence_score > 0.6):
                    
                    similarity = self._field_similarity(field1, field2)
                    if similarity > 0.7:
                        potential_joins.append({
                            'field1': field1.field_name,
                            'field2': field2.field_name,
                            'category': field1.ao1_category,
                            'similarity': similarity
                        })
        
        if potential_joins:
            return {
                'table1': table1,
                'table2': table2,
                'join_candidates': potential_joins,
                'relationship_strength': max(j['similarity'] for j in potential_joins)
            }
        
        return None
    
    def _field_similarity(self, field1: FieldAnalysis, field2: FieldAnalysis) -> float:
        name_similarity = fuzz.ratio(field1.field_name, field2.field_name) / 100.0
        
        pattern_overlap = len(set(field1.value_patterns) & set(field2.value_patterns))
        max_patterns = max(len(field1.value_patterns), len(field2.value_patterns), 1)
        pattern_similarity = pattern_overlap / max_patterns
        
        confidence_similarity = min(field1.confidence_score, field2.confidence_score)
        
        return (name_similarity * 0.3 + pattern_similarity * 0.4 + confidence_similarity * 0.3)