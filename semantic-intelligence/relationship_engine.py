#!/usr/bin/env python3

import math
import statistics
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from models import FieldIntelligence

class RelationshipEngine:
    def __init__(self):
        self.relationship_cache = {}
        
    def analyze_relationships(self, fields: List[FieldIntelligence]) -> Dict[str, Dict[str, float]]:
        relationships = defaultdict(dict)
        
        for i, field1 in enumerate(fields):
            for j, field2 in enumerate(fields[i+1:], i+1):
                strength = self._calculate_relationship_strength(field1, field2)
                
                if strength > 0.3:  # Only store meaningful relationships
                    key1, key2 = field1.get_key(), field2.get_key()
                    relationships[key1][key2] = strength
                    relationships[key2][key1] = strength
                    
        return dict(relationships)
        
    def _calculate_relationship_strength(self, field1: FieldIntelligence, field2: FieldIntelligence) -> float:
        cache_key = f"{field1.get_key()}|{field2.get_key()}"
        if cache_key in self.relationship_cache:
            return self.relationship_cache[cache_key]
            
        if not field1.semantic_profile or not field2.semantic_profile:
            strength = 0.0
        else:
            # Multiple similarity dimensions
            semantic_sim = self._semantic_similarity(field1, field2)
            structural_sim = self._structural_similarity(field1, field2)
            statistical_sim = self._statistical_similarity(field1, field2)
            name_sim = self._name_similarity(field1.name, field2.name)
            
            # Weighted combination
            strength = (
                semantic_sim * 0.35 +
                structural_sim * 0.25 +
                statistical_sim * 0.25 +
                name_sim * 0.15
            )
            
            # Table bonus
            if field1.table == field2.table:
                strength += 0.1
                
            strength = min(1.0, strength)
            
        self.relationship_cache[cache_key] = strength
        return strength
        
    def _semantic_similarity(self, field1: FieldIntelligence, field2: FieldIntelligence) -> float:
        profile1, profile2 = field1.semantic_profile, field2.semantic_profile
        
        if not profile1.pattern_features or not profile2.pattern_features:
            return 0.0
            
        similarities = []
        
        # Compare pattern features
        for pattern_type in profile1.pattern_features.keys():
            if pattern_type in profile2.pattern_features:
                vec1 = profile1.pattern_features[pattern_type]
                vec2 = profile2.pattern_features[pattern_type]
                
                if vec1 and vec2 and len(vec1) == len(vec2):
                    sim = self._cosine_similarity(vec1, vec2)
                    similarities.append(sim)
                    
        # Compare entropy metrics
        entropy_sim = self._entropy_similarity(profile1.entropy_metrics, profile2.entropy_metrics)
        similarities.append(entropy_sim)
        
        return statistics.mean(similarities) if similarities else 0.0
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    def _entropy_similarity(self, entropy1: Dict[str, float], entropy2: Dict[str, float]) -> float:
        if not entropy1 or not entropy2:
            return 0.0
            
        common_keys = set(entropy1.keys()) & set(entropy2.keys())
        if not common_keys:
            return 0.0
            
        similarities = []
        for key in common_keys:
            val1, val2 = entropy1[key], entropy2[key]
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            else:
                max_val = max(val1, val2)
                if max_val > 0:
                    sim = 1.0 - abs(val1 - val2) / max_val
                    similarities.append(max(0.0, sim))
                    
        return statistics.mean(similarities)
        
    def _structural_similarity(self, field1: FieldIntelligence, field2: FieldIntelligence) -> float:
        if not field1.sample_values or not field2.sample_values:
            return 0.0
            
        # Length distribution similarity
        lengths1 = [len(str(v)) for v in field1.sample_values if v is not None]
        lengths2 = [len(str(v)) for v in field2.sample_values if v is not None]
        
        if not lengths1 or not lengths2:
            return 0.0
            
        length_sim = self._distribution_similarity(lengths1, lengths2)
        
        # Format similarity
        format_sim = self._format_similarity(field1.sample_values, field2.sample_values)
        
        # Character distribution similarity
        char_sim = self._character_similarity(field1.sample_values, field2.sample_values)
        
        return statistics.mean([length_sim, format_sim, char_sim])
        
    def _distribution_similarity(self, values1: List[int], values2: List[int]) -> float:
        dist1 = Counter(values1)
        dist2 = Counter(values2)
        
        all_values = set(dist1.keys()) | set(dist2.keys())
        
        similarities = []
        total1, total2 = len(values1), len(values2)
        
        for value in all_values:
            freq1 = dist1.get(value, 0) / total1
            freq2 = dist2.get(value, 0) / total2
            sim = 1.0 - abs(freq1 - freq2)
            similarities.append(sim)
            
        return statistics.mean(similarities)
        
    def _format_similarity(self, values1: List, values2: List) -> float:
        import re
        
        def extract_format(values):
            formats = Counter()
            for v in values:
                if v is not None:
                    pattern = re.sub(r'\d', 'N', re.sub(r'[a-zA-Z]', 'A', re.sub(r'[^a-zA-Z0-9]', 'S', str(v))))
                    formats[pattern] += 1
            return formats
            
        formats1 = extract_format(values1)
        formats2 = extract_format(values2)
        
        if not formats1 or not formats2:
            return 0.0
            
        common_formats = set(formats1.keys()) & set(formats2.keys())
        if not common_formats:
            return 0.0
            
        similarities = []
        total1, total2 = sum(formats1.values()), sum(formats2.values())
        
        for fmt in common_formats:
            freq1 = formats1[fmt] / total1
            freq2 = formats2[fmt] / total2
            sim = 1.0 - abs(freq1 - freq2)
            similarities.append(sim)
            
        return statistics.mean(similarities)
        
    def _character_similarity(self, values1: List, values2: List) -> float:
        chars1 = Counter(''.join(str(v) for v in values1 if v is not None).lower())
        chars2 = Counter(''.join(str(v) for v in values2 if v is not None).lower())
        
        if not chars1 or not chars2:
            return 0.0
            
        all_chars = set(chars1.keys()) | set(chars2.keys())
        
        similarities = []
        total1, total2 = sum(chars1.values()), sum(chars2.values())
        
        for char in all_chars:
            freq1 = chars1.get(char, 0) / total1 if total1 > 0 else 0
            freq2 = chars2.get(char, 0) / total2 if total2 > 0 else 0
            sim = 1.0 - abs(freq1 - freq2)
            similarities.append(sim)
            
        return statistics.mean(similarities)
        
    def _statistical_similarity(self, field1: FieldIntelligence, field2: FieldIntelligence) -> float:
        stats1 = field1.semantic_profile.statistical_properties
        stats2 = field2.semantic_profile.statistical_properties
        
        if not stats1 or not stats2:
            return 0.0
            
        common_stats = set(stats1.keys()) & set(stats2.keys())
        
        similarities = []
        for stat in common_stats:
            val1, val2 = stats1[stat], stats2[stat]
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            elif val1 == 0 or val2 == 0:
                similarities.append(0.0)
            else:
                max_val = max(abs(val1), abs(val2))
                sim = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(0.0, sim))
                
        return statistics.mean(similarities) if similarities else 0.0
        
    def _name_similarity(self, name1: str, name2: str) -> float:
        name1_lower, name2_lower = name1.lower(), name2.lower()
        
        # Keyword matching
        keywords = ['id', 'key', 'name', 'type', 'status', 'code', 'time', 'date', 'user']
        keyword_matches = sum(1 for kw in keywords if kw in name1_lower and kw in name2_lower)
        keyword_sim = keyword_matches / len(keywords)
        
        # Character overlap
        chars1, chars2 = set(name1_lower), set(name2_lower)
        char_overlap = len(chars1 & chars2)
        char_total = len(chars1 | chars2)
        char_sim = char_overlap / char_total if char_total > 0 else 0.0
        
        # Length similarity
        max_len = max(len(name1), len(name2))
        length_sim = 1.0 - abs(len(name1) - len(name2)) / max_len if max_len > 0 else 1.0
        
        return keyword_sim * 0.5 + char_sim * 0.3 + length_sim * 0.2
        
    def create_clusters(self, fields: List[FieldIntelligence], threshold: float = 0.7) -> Dict[str, List[str]]:
        if len(fields) < 2:
            return {}
            
        # Build similarity matrix
        field_keys = [f.get_key() for f in fields]
        field_map = {f.get_key(): f for f in fields}
        
        similarity_matrix = {}
        for i, key1 in enumerate(field_keys):
            similarity_matrix[key1] = {}
            for j, key2 in enumerate(field_keys):
                if i != j:
                    sim = self._calculate_relationship_strength(field_map[key1], field_map[key2])
                    similarity_matrix[key1][key2] = sim
                    
        # Perform agglomerative clustering
        clusters = self._cluster_fields(similarity_matrix, threshold)
        
        # Group by cluster ID
        cluster_groups = defaultdict(list)
        for field_key, cluster_id in clusters.items():
            cluster_groups[f"cluster_{cluster_id}"].append(field_key)
            
        return dict(cluster_groups)
        
    def _cluster_fields(self, similarity_matrix: Dict[str, Dict[str, float]], threshold: float) -> Dict[str, int]:
        field_keys = list(similarity_matrix.keys())
        clusters = {field: i for i, field in enumerate(field_keys)}
        
        changed = True
        while changed:
            changed = False
            max_similarity = 0.0
            merge_pair = None
            
            for field1 in field_keys:
                for field2 in field_keys:
                    if field1 != field2 and clusters[field1] != clusters[field2]:
                        sim = similarity_matrix[field1].get(field2, 0.0)
                        if sim > max_similarity and sim > threshold:
                            max_similarity = sim
                            merge_pair = (field1, field2)
                            
            if merge_pair:
                field1, field2 = merge_pair
                old_cluster = clusters[field2]
                new_cluster = clusters[field1]
                
                for field in field_keys:
                    if clusters[field] == old_cluster:
                        clusters[field] = new_cluster
                        
                changed = True
                
        # Renumber clusters sequentially
        unique_clusters = list(set(clusters.values()))
        cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}
        
        return {field: cluster_mapping[cluster_id] for field, cluster_id in clusters.items()}