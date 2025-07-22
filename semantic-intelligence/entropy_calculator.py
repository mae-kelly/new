#!/usr/bin/env python3

import math
from typing import List, Dict
from collections import Counter

class EntropyCalculator:
    def calculate_all_entropy_metrics(self, values: List[str]) -> Dict[str, float]:
        if not values:
            return {}
            
        return {
            'shannon': self._shannon_entropy(values),
            'length_distribution': self._length_entropy(values),
            'character_distribution': self._character_entropy(values),
            'normalized': self._normalized_entropy(values),
            'conditional': self._conditional_entropy(values),
            'compression_ratio': self._compression_entropy(values)
        }
        
    def _shannon_entropy(self, values: List[str]) -> float:
        if not values:
            return 0.0
        counts = Counter(values)
        total = len(values)
        return -sum((count/total) * math.log2(count/total) for count in counts.values())
        
    def _length_entropy(self, values: List[str]) -> float:
        lengths = [len(v) for v in values]
        return self._shannon_entropy([str(l) for l in lengths])
        
    def _character_entropy(self, values: List[str]) -> float:
        chars = Counter()
        for value in values:
            chars.update(value.lower())
        total = sum(chars.values())
        if total == 0:
            return 0.0
        return -sum((count/total) * math.log2(count/total) for count in chars.values())
        
    def _normalized_entropy(self, values: List[str]) -> float:
        unique_values = len(set(values))
        if unique_values <= 1:
            return 0.0
        shannon = self._shannon_entropy(values)
        max_entropy = math.log2(unique_values)
        return shannon / max_entropy
        
    def _conditional_entropy(self, values: List[str]) -> float:
        if len(values) < 2:
            return 0.0
            
        char_transitions = {}
        for value in values:
            for i in range(len(value) - 1):
                current_char = value[i]
                next_char = value[i + 1]
                
                if current_char not in char_transitions:
                    char_transitions[current_char] = Counter()
                char_transitions[current_char][next_char] += 1
                
        conditional_entropy = 0.0
        total_transitions = sum(sum(transitions.values()) for transitions in char_transitions.values())
        
        if total_transitions == 0:
            return 0.0
            
        for char, transitions in char_transitions.items():
            char_prob = sum(transitions.values()) / total_transitions
            char_entropy = 0.0
            total_char_transitions = sum(transitions.values())
            
            for next_char, count in transitions.items():
                prob = count / total_char_transitions
                char_entropy -= prob * math.log2(prob)
                
            conditional_entropy += char_prob * char_entropy
            
        return conditional_entropy
        
    def _compression_entropy(self, values: List[str]) -> float:
        if not values:
            return 0.0
        text = ''.join(values)
        if not text:
            return 0.0
            
        try:
            import zlib
            compressed = zlib.compress(text.encode())
            return len(compressed) / len(text.encode())
        except:
            # Fallback: estimate compression ratio using character diversity
            unique_chars = len(set(text))
            total_chars = len(text)
            return unique_chars / total_chars if total_chars > 0 else 0.0