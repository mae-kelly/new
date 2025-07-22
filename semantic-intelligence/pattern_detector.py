#!/usr/bin/env python3

import re
import statistics
from typing import List, Dict, Any
from collections import Counter

class PatternDetector:
    def __init__(self):
        self.domain_patterns = {
            'identity': ['user', 'person', 'account', 'name', 'id', 'login', 'email'],
            'network': ['ip', 'address', 'hostname', 'port', 'protocol', 'mac', 'url'],
            'security': ['alert', 'threat', 'event', 'auth', 'permission', 'token', 'key'],
            'temporal': ['time', 'date', 'timestamp', 'created', 'modified', 'start', 'end'],
            'business': ['customer', 'order', 'product', 'account', 'transaction', 'price'],
            'system': ['server', 'service', 'process', 'application', 'database', 'log'],
            'location': ['address', 'city', 'country', 'region', 'zone', 'latitude', 'longitude'],
            'status': ['status', 'state', 'flag', 'active', 'enabled', 'valid', 'error']
        }
        
        self.regex_patterns = {
            'ipv4': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'uuid': r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
            'iso_date': r'\b\d{4}-\d{2}-\d{2}\b',
            'iso_datetime': r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
            'unix_timestamp': r'\b\d{10,13}\b',
            'url': r'https?://[^\s]+',
            'mac_address': r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b'
        }
        
    def detect_format_patterns(self, values: List[Any]) -> List[float]:
        if not values:
            return [0.0] * 8
            
        string_values = [str(v) for v in values]
        patterns = []
        
        # Character type analysis
        all_text = ''.join(string_values).lower()
        if all_text:
            alpha_ratio = sum(c.isalpha() for c in all_text) / len(all_text)
            digit_ratio = sum(c.isdigit() for c in all_text) / len(all_text)
            special_ratio = sum(not c.isalnum() for c in all_text) / len(all_text)
            patterns.extend([alpha_ratio, digit_ratio, special_ratio])
        else:
            patterns.extend([0.0, 0.0, 0.0])
            
        # Format consistency
        format_patterns = []
        for value in string_values:
            pattern = re.sub(r'\d', 'N', re.sub(r'[a-zA-Z]', 'A', re.sub(r'[^a-zA-Z0-9]', 'S', value)))
            format_patterns.append(pattern)
            
        format_consistency = 0.0
        if format_patterns:
            most_common = Counter(format_patterns).most_common(1)[0][1]
            format_consistency = most_common / len(format_patterns)
        patterns.append(format_consistency)
        
        # Length consistency
        lengths = [len(v) for v in string_values]
        length_consistency = 0.0
        if lengths:
            if len(set(lengths)) == 1:
                length_consistency = 1.0
            else:
                mean_length = statistics.mean(lengths)
                if mean_length > 0:
                    std_length = statistics.stdev(lengths) if len(lengths) > 1 else 0
                    length_consistency = max(0.0, 1.0 - (std_length / mean_length))
        patterns.append(length_consistency)
        
        # Case pattern analysis
        case_patterns = []
        for value in string_values:
            if value:
                upper_ratio = sum(c.isupper() for c in value) / len(value)
                case_patterns.append(upper_ratio)
        case_consistency = 1.0 - statistics.stdev(case_patterns) if len(case_patterns) > 1 else 1.0
        patterns.append(case_consistency)
        
        # Punctuation patterns
        punct_chars = '.,;:!?-_()[]{}/'
        punct_ratios = []
        for value in string_values:
            if value:
                punct_ratio = sum(c in punct_chars for c in value) / len(value)
                punct_ratios.append(punct_ratio)
        punct_consistency = statistics.mean(punct_ratios) if punct_ratios else 0.0
        patterns.append(punct_consistency)
        
        # Structure complexity
        struct_complexity = self._calculate_structural_complexity(string_values)
        patterns.append(struct_complexity)
        
        return patterns[:8] + [0.0] * max(0, 8 - len(patterns))
        
    def detect_semantic_patterns(self, field_name: str, values: List[Any]) -> List[float]:
        patterns = []
        field_lower = field_name.lower()
        
        for domain, keywords in self.domain_patterns.items():
            domain_score = 0.0
            
            # Check field name
            name_matches = sum(1 for keyword in keywords if keyword in field_lower)
            domain_score += (name_matches / len(keywords)) * 0.7
            
            # Check sample values
            if values:
                sample_text = ' '.join(str(v).lower() for v in values[:20])
                value_matches = sum(1 for keyword in keywords if keyword in sample_text)
                domain_score += (value_matches / len(keywords)) * 0.3
                
            patterns.append(min(1.0, domain_score))
            
        return patterns
        
    def detect_regex_patterns(self, values: List[Any]) -> Dict[str, float]:
        if not values:
            return {pattern: 0.0 for pattern in self.regex_patterns.keys()}
            
        string_values = [str(v) for v in values]
        results = {}
        
        for pattern_name, regex in self.regex_patterns.items():
            matches = 0
            for value in string_values:
                if re.search(regex, value):
                    matches += 1
            results[pattern_name] = matches / len(string_values)
            
        return results
        
    def detect_structural_patterns(self, values: List[Any]) -> List[float]:
        if not values:
            return [0.0] * 6
            
        string_values = [str(v) for v in values]
        patterns = []
        
        # Delimiter analysis
        delimiters = [',', ';', '|', ':', '-', '_']
        for delimiter in delimiters:
            delimiter_count = sum(v.count(delimiter) for v in string_values)
            total_chars = sum(len(v) for v in string_values)
            delimiter_ratio = delimiter_count / total_chars if total_chars > 0 else 0
            patterns.append(delimiter_ratio)
            
        return patterns[:6] + [0.0] * max(0, 6 - len(patterns))
        
    def detect_temporal_patterns(self, values: List[Any]) -> List[float]:
        if not values:
            return [0.0] * 5
            
        string_values = [str(v) for v in values]
        patterns = []
        
        # Temporal format detection
        temporal_regexes = [
            r'\d{4}-\d{2}-\d{2}',  # ISO date
            r'\d{10,13}',          # Unix timestamp
            r'\d{2}/\d{2}/\d{4}',  # US date format
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO datetime
        ]
        
        for regex in temporal_regexes:
            matches = sum(1 for v in string_values if re.search(regex, v))
            ratio = matches / len(string_values)
            patterns.append(ratio)
            
        # Sequence detection
        sequence_score = self._detect_sequence_pattern(values)
        patterns.append(sequence_score)
        
        return patterns[:5]
        
    def _calculate_structural_complexity(self, values: List[str]) -> float:
        if not values:
            return 0.0
            
        complexity_indicators = 0
        total_chars = sum(len(v) for v in values)
        
        if total_chars == 0:
            return 0.0
            
        # Mixed case complexity
        mixed_case = sum(
            len([c for c in v if c.isupper()]) * len([c for c in v if c.islower()]) 
            for v in values
        )
        complexity_indicators += mixed_case / (total_chars ** 2)
        
        # Alphanumeric mixing
        alpha_numeric_mix = sum(
            len([c for c in v if c.isalpha()]) * len([c for c in v if c.isdigit()]) 
            for v in values
        )
        complexity_indicators += alpha_numeric_mix / (total_chars ** 2)
        
        # Special character integration
        special_chars = sum(len([c for c in v if not c.isalnum()]) for v in values)
        complexity_indicators += special_chars / total_chars
        
        return min(1.0, complexity_indicators)
        
    def _detect_sequence_pattern(self, values: List[Any]) -> float:
        try:
            # Try to convert to numeric sequence
            numeric_values = []
            for v in values:
                str_v = str(v)
                if str_v.replace('.', '').replace('-', '').isdigit():
                    numeric_values.append(float(str_v))
                    
            if len(numeric_values) < 3:
                return 0.0
                
            # Check for arithmetic progression
            differences = [numeric_values[i+1] - numeric_values[i] for i in range(len(numeric_values)-1)]
            
            if not differences:
                return 0.0
                
            # Calculate consistency of differences
            first_diff = differences[0]
            consistent_diffs = sum(1 for d in differences if abs(d - first_diff) < 0.001)
            
            return consistent_diffs / len(differences)
            
        except:
            return 0.0