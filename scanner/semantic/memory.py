import logging
import pickle
import os
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime
from ...config import SEMANTIC_MODELS_CACHE

logger = logging.getLogger(__name__)

class SemanticMemory:
    def __init__(self):
        self.memory = self._load_memory()
        self.learning_rate = 0.1
        self.confidence_threshold = 0.6
    
    def _load_memory(self):
        memory_file = os.path.join(SEMANTIC_MODELS_CACHE, 'semantic_memory.pkl')
        
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load semantic memory: {e}")
        
        return {
            'field_patterns': defaultdict(set),
            'value_signatures': defaultdict(list),
            'concept_associations': defaultdict(lambda: defaultdict(float)),
            'contextual_patterns': defaultdict(list),
            'confidence_history': defaultdict(list),
            'learning_feedback': [],
            'cross_table_patterns': defaultdict(dict),
            'error_corrections': defaultdict(list),
            'temporal_patterns': defaultdict(list)
        }
    
    def learn_from_analysis(self, field_name: str, concept: str, confidence: float, 
                           sample_values: List[str], table_context: Dict[str, Any]):
        if confidence < self.confidence_threshold:
            return
        
        # Learn field patterns
        self._learn_field_pattern(field_name, concept, confidence)
        
        # Learn value signatures
        self._learn_value_signatures(sample_values, concept, confidence)
        
        # Learn contextual associations
        self._learn_contextual_patterns(table_context, concept, confidence)
        
        # Update confidence history
        self._update_confidence_history(concept, confidence)
        
        # Learn cross-table patterns
        self._learn_cross_table_patterns(field_name, concept, table_context)
    
    def _learn_field_pattern(self, field_name: str, concept: str, confidence: float):
        normalized_name = self._normalize_field_name(field_name)
        self.memory['field_patterns'][concept].add(normalized_name)
        
        # Update association strength
        current_strength = self.memory['concept_associations'][concept][normalized_name]
        new_strength = current_strength + (confidence * self.learning_rate)
        self.memory['concept_associations'][concept][normalized_name] = min(1.0, new_strength)
    
    def _learn_value_signatures(self, sample_values: List[str], concept: str, confidence: float):
        for value in sample_values[:5]:
            if value and len(str(value)) > 2:
                signature = self._generate_value_signature(str(value))
                
                # Check if signature already exists
                existing_signatures = self.memory['value_signatures'][concept]
                if not any(self._signature_similarity(signature, existing) > 0.8 
                          for existing in existing_signatures):
                    self.memory['value_signatures'][concept].append(signature)
                    
                    # Keep only top N signatures
                    if len(self.memory['value_signatures'][concept]) > 50:
                        self.memory['value_signatures'][concept] = \
                            self.memory['value_signatures'][concept][-50:]
    
    def _learn_contextual_patterns(self, table_context: Dict[str, Any], concept: str, confidence: float):
        themes = table_context.get('semantic_themes', [])
        domain_indicators = table_context.get('domain_indicators', [])
        
        for theme in themes:
            current_strength = self.memory['concept_associations'][concept][f"theme_{theme}"]
            new_strength = current_strength + (confidence * self.learning_rate * 0.5)
            self.memory['concept_associations'][concept][f"theme_{theme}"] = min(1.0, new_strength)
        
        for indicator in domain_indicators:
            current_strength = self.memory['concept_associations'][concept][f"domain_{indicator}"]
            new_strength = current_strength + (confidence * self.learning_rate * 0.3)
            self.memory['concept_associations'][concept][f"domain_{indicator}"] = min(1.0, new_strength)
    
    def _update_confidence_history(self, concept: str, confidence: float):
        history = self.memory['confidence_history'][concept]
        history.append({
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(history) > 100:
            self.memory['confidence_history'][concept] = history[-100:]
    
    def _learn_cross_table_patterns(self, field_name: str, concept: str, table_context: Dict[str, Any]):
        table_signature = self._generate_table_signature(table_context)
        
        if concept not in self.memory['cross_table_patterns']:
            self.memory['cross_table_patterns'][concept] = {}
        
        patterns = self.memory['cross_table_patterns'][concept]
        if table_signature not in patterns:
            patterns[table_signature] = []
        
        patterns[table_signature].append(self._normalize_field_name(field_name))
        
        # Keep unique patterns only
        patterns[table_signature] = list(set(patterns[table_signature]))
    
    def get_learned_score(self, field_name: str, concept: str, table_context: Dict[str, Any]) -> float:
        score = 0.0
        
        # Check field pattern similarity
        normalized_name = self._normalize_field_name(field_name)
        if normalized_name in self.memory['field_patterns'][concept]:
            score += 0.8
        else:
            # Check for similar patterns
            similar_score = self._find_similar_field_pattern(normalized_name, concept)
            score += similar_score * 0.6
        
        # Check contextual patterns
        contextual_score = self._check_contextual_patterns(table_context, concept)
        score += contextual_score * 0.4
        
        # Check cross-table patterns
        cross_table_score = self._check_cross_table_patterns(field_name, concept, table_context)
        score += cross_table_score * 0.3
        
        return min(1.0, score)
    
    def _normalize_field_name(self, field_name: str) -> str:
        import re
        normalized = re.sub(r'[_\-\.]', ' ', field_name.lower())
        normalized = re.sub(r'([a-z])([A-Z])', r'\1 \2', normalized)
        return normalized.strip()
    
    def _generate_value_signature(self, value: str) -> str:
        import re
        value_lower = value.lower().strip()
        
        # Generate pattern signature
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}, value_lower):
            return 'ipv4_pattern'
        elif re.match(r'^[a-zA-Z0-9\-\.]+\.(com|local|corp|internal|net|org), value_lower):
            return 'fqdn_pattern'
        elif re.match(r'^[a-zA-Z0-9\-]+\d+, value_lower):
            return 'alphanumeric_id_pattern'
        elif 'crowdstrike' in value_lower or 'falcon' in value_lower:
            return 'crowdstrike_reference'
        elif 'chronicle' in value_lower:
            return 'chronicle_reference'
        elif 'splunk' in value_lower:
            return 'splunk_reference'
        elif len(value) < 5:
            return 'short_code'
        elif len(value) > 50:
            return 'long_text'
        else:
            return f'text_pattern_{len(value)}'
    
    def _signature_similarity(self, sig1: str, sig2: str) -> float:
        if sig1 == sig2:
            return 1.0
        
        # Check pattern family similarity
        pattern_families = {
            'network': ['ipv4_pattern', 'fqdn_pattern'],
            'security': ['crowdstrike_reference', 'chronicle_reference', 'splunk_reference'],
            'text': ['short_code', 'long_text']
        }
        
        for family, patterns in pattern_families.items():
            if sig1 in patterns and sig2 in patterns:
                return 0.7
        
        return 0.0
    
    def _generate_table_signature(self, table_context: Dict[str, Any]) -> str:
        themes = sorted(table_context.get('semantic_themes', []))
        domains = sorted(table_context.get('domain_indicators', []))
        return f"themes_{'-'.join(themes)}_domains_{'-'.join(domains)}"
    
    def _find_similar_field_pattern(self, field_name: str, concept: str) -> float:
        from fuzzywuzzy import fuzz
        
        max_similarity = 0.0
        learned_patterns = self.memory['field_patterns'][concept]
        
        for pattern in learned_patterns:
            similarity = fuzz.token_sort_ratio(field_name, pattern) / 100.0
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity if max_similarity > 0.8 else 0.0
    
    def _check_contextual_patterns(self, table_context: Dict[str, Any], concept: str) -> float:
        score = 0.0
        associations = self.memory['concept_associations'][concept]
        
        themes = table_context.get('semantic_themes', [])
        for theme in themes:
            theme_key = f"theme_{theme}"
            if theme_key in associations:
                score += associations[theme_key] * 0.5
        
        domains = table_context.get('domain_indicators', [])
        for domain in domains:
            domain_key = f"domain_{domain}"
            if domain_key in associations:
                score += associations[domain_key] * 0.3
        
        return min(1.0, score)
    
    def _check_cross_table_patterns(self, field_name: str, concept: str, table_context: Dict[str, Any]) -> float:
        table_signature = self._generate_table_signature(table_context)
        normalized_field = self._normalize_field_name(field_name)
        
        cross_patterns = self.memory['cross_table_patterns'].get(concept, {})
        
        if table_signature in cross_patterns:
            if normalized_field in cross_patterns[table_signature]:
                return 0.9
            else:
                # Check for similar fields in same table pattern
                from fuzzywuzzy import fuzz
                for pattern_field in cross_patterns[table_signature]:
                    similarity = fuzz.token_sort_ratio(normalized_field, pattern_field) / 100.0
                    if similarity > 0.8:
                        return similarity * 0.7
        
        return 0.0
    
    def record_feedback(self, field_name: str, predicted_concept: str, actual_concept: str, confidence: float):
        feedback = {
            'field_name': field_name,
            'predicted': predicted_concept,
            'actual': actual_concept,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'was_correct': predicted_concept == actual_concept
        }
        
        self.memory['learning_feedback'].append(feedback)
        
        if not feedback['was_correct']:
            self.memory['error_corrections'][predicted_concept].append({
                'field_name': field_name,
                'correct_concept': actual_concept,
                'timestamp': feedback['timestamp']
            })
        
        # Keep recent feedback only
        if len(self.memory['learning_feedback']) > 500:
            self.memory['learning_feedback'] = self.memory['learning_feedback'][-500:]
    
    def get_calibrated_confidence(self, concept: str, raw_confidence: float) -> float:
        history = self.memory['confidence_history'].get(concept, [])
        
        if not history:
            return raw_confidence
        
        # Calculate historical accuracy for similar confidence ranges
        similar_confidences = [h['confidence'] for h in history 
                             if abs(h['confidence'] - raw_confidence) < 0.2]
        
        if similar_confidences:
            avg_historical = sum(similar_confidences) / len(similar_confidences)
            # Blend historical and current
            calibrated = 0.7 * raw_confidence + 0.3 * avg_historical
        else:
            calibrated = raw_confidence
        
        return max(0.0, min(1.0, calibrated))
    
    def save_memory(self):
        memory_file = os.path.join(SEMANTIC_MODELS_CACHE, 'semantic_memory.pkl')
        try:
            # Convert sets to lists for serialization
            serializable_memory = {}
            for key, value in self.memory.items():
                if key == 'field_patterns':
                    serializable_memory[key] = {k: list(v) for k, v in value.items()}
                else:
                    serializable_memory[key] = dict(value) if hasattr(value, 'items') else value
            
            with open(memory_file, 'wb') as f:
                pickle.dump(serializable_memory, f)
        except Exception as e:
            logger.warning(f"Failed to save semantic memory: {e}")
    
    def __del__(self):
        self.save_memory()