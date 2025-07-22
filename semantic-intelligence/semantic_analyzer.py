#!/usr/bin/env python3

import statistics
from typing import List, Dict, Any
from models import SemanticProfile, FieldIntelligence
from entropy_calculator import EntropyCalculator
from pattern_detector import PatternDetector

class SemanticAnalyzer:
    def __init__(self):
        self.entropy_calculator = EntropyCalculator()
        self.pattern_detector = PatternDetector()
        
    def analyze_field(self, field_name: str, table_name: str, data_type: str, 
                     sample_values: List[Any]) -> FieldIntelligence:
        
        if not sample_values:
            return FieldIntelligence(field_name, table_name, data_type)
            
        clean_values = [v for v in sample_values if v is not None]
        if not clean_values:
            return FieldIntelligence(field_name, table_name, data_type)
            
        # Create semantic profile
        profile = self._create_semantic_profile(field_name, clean_values)
        
        # Calculate intelligence metrics
        intelligence_score = self._calculate_intelligence_score(profile)
        confidence_level = self._calculate_confidence_level(profile, clean_values)
        business_context = self._derive_business_context(field_name, profile)
        
        field_intelligence = FieldIntelligence(
            name=field_name,
            table=table_name,
            data_type=data_type,
            semantic_profile=profile,
            intelligence_score=intelligence_score,
            confidence_level=confidence_level,
            business_context=business_context,
            sample_values=clean_values[:50]  # Keep reasonable sample size
        )
        
        return field_intelligence
        
    def _create_semantic_profile(self, field_name: str, values: List[Any]) -> SemanticProfile:
        string_values = [str(v) for v in values]
        
        profile = SemanticProfile(field_name)
        
        # Calculate entropy metrics
        profile.entropy_metrics = self.entropy_calculator.calculate_all_entropy_metrics(string_values)
        
        # Extract pattern features
        profile.pattern_features = {
            'format_patterns': self.pattern_detector.detect_format_patterns(values),
            'semantic_patterns': self.pattern_detector.detect_semantic_patterns(field_name, values),
            'structural_patterns': self.pattern_detector.detect_structural_patterns(values),
            'temporal_patterns': self.pattern_detector.detect_temporal_patterns(values),
            'regex_matches': list(self.pattern_detector.detect_regex_patterns(values).values())
        }
        
        # Calculate statistical properties
        profile.statistical_properties = self._calculate_statistical_properties(values)
        
        # Analyze behavioral indicators
        profile.behavioral_indicators = self._analyze_behavioral_patterns(values)
        
        # Calculate derived metrics
        profile.semantic_density = self._calculate_semantic_density(profile)
        profile.complexity_score = self._calculate_complexity_score(profile)
        
        return profile
        
    def _calculate_statistical_properties(self, values: List[Any]) -> Dict[str, float]:
        properties = {}
        
        # String-based statistics
        string_values = [str(v) for v in values]
        if string_values:
            lengths = [len(v) for v in string_values]
            properties.update({
                'avg_length': statistics.mean(lengths),
                'length_std': statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
                'min_length': min(lengths),
                'max_length': max(lengths),
                'length_range': max(lengths) - min(lengths)
            })
            
            # Character diversity
            all_chars = ''.join(string_values)
            if all_chars:
                properties['character_diversity'] = len(set(all_chars)) / len(all_chars)
            else:
                properties['character_diversity'] = 0.0
                
        # Numeric statistics (if applicable)
        numeric_values = self._extract_numeric_values(values)
        if numeric_values:
            properties.update({
                'numeric_mean': statistics.mean(numeric_values),
                'numeric_median': statistics.median(numeric_values),
                'numeric_std': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
                'numeric_range': max(numeric_values) - min(numeric_values)
            })
            
        # Distribution properties
        from collections import Counter
        value_counts = Counter(string_values)
        properties.update({
            'unique_ratio': len(value_counts) / len(string_values),
            'singleton_ratio': sum(1 for count in value_counts.values() if count == 1) / len(value_counts) if value_counts else 0,
            'max_frequency': max(value_counts.values()) / len(string_values) if value_counts else 0
        })
        
        return properties
        
    def _analyze_behavioral_patterns(self, values: List[Any]) -> Dict[str, float]:
        patterns = {}
        
        patterns['consistency'] = self._calculate_consistency_score(values)
        patterns['predictability'] = self._calculate_predictability_score(values)
        patterns['variability'] = self._calculate_variability_score(values)
        patterns['anomaly_rate'] = self._calculate_anomaly_rate(values)
        patterns['growth_trend'] = self._calculate_growth_trend(values)
        
        return patterns
        
    def _calculate_consistency_score(self, values: List[Any]) -> float:
        if not values:
            return 0.0
            
        string_values = [str(v) for v in values]
        
        # Format consistency
        import re
        from collections import Counter
        
        formats = []
        for value in string_values:
            pattern = re.sub(r'\d', 'N', re.sub(r'[a-zA-Z]', 'A', re.sub(r'[^a-zA-Z0-9]', 'S', value)))
            formats.append(pattern)
            
        format_counts = Counter(formats)
        format_consistency = max(format_counts.values()) / len(formats) if formats else 0.0
        
        # Length consistency
        lengths = [len(v) for v in string_values]
        if len(set(lengths)) == 1:
            length_consistency = 1.0
        else:
            mean_length = statistics.mean(lengths)
            if mean_length > 0:
                std_length = statistics.stdev(lengths) if len(lengths) > 1 else 0
                length_consistency = max(0.0, 1.0 - (std_length / mean_length))
            else:
                length_consistency = 0.0
                
        return (format_consistency + length_consistency) / 2.0
        
    def _calculate_predictability_score(self, values: List[Any]) -> float:
        numeric_values = self._extract_numeric_values(values)
        if len(numeric_values) < 3:
            return 0.0
            
        try:
            differences = [numeric_values[i+1] - numeric_values[i] for i in range(len(numeric_values)-1)]
            if not differences:
                return 0.0
                
            # High predictability if differences are consistent
            diff_std = statistics.stdev(differences) if len(differences) > 1 else 0
            diff_mean = abs(statistics.mean(differences))
            
            if diff_mean == 0:
                return 1.0 if diff_std == 0 else 0.0
            else:
                return max(0.0, 1.0 - (diff_std / diff_mean))
        except:
            return 0.0
            
    def _calculate_variability_score(self, values: List[Any]) -> float:
        if not values:
            return 0.0
        string_values = [str(v) for v in values]
        unique_count = len(set(string_values))
        return unique_count / len(string_values)
        
    def _calculate_anomaly_rate(self, values: List[Any]) -> float:
        numeric_values = self._extract_numeric_values(values)
        if len(numeric_values) < 3:
            return 0.0
            
        try:
            mean_val = statistics.mean(numeric_values)
            std_val = statistics.stdev(numeric_values)
            
            if std_val == 0:
                return 0.0
                
            anomalies = sum(1 for v in numeric_values if abs(v - mean_val) > 2 * std_val)
            return anomalies / len(numeric_values)
        except:
            return 0.0
            
    def _calculate_growth_trend(self, values: List[Any]) -> float:
        numeric_values = self._extract_numeric_values(values)
        if len(numeric_values) < 3:
            return 0.0
            
        try:
            differences = [numeric_values[i+1] - numeric_values[i] for i in range(len(numeric_values)-1)]
            positive_diffs = sum(1 for d in differences if d > 0)
            negative_diffs = sum(1 for d in differences if d < 0)
            
            if positive_diffs > len(differences) * 0.8:
                return 1.0  # Strong positive trend
            elif negative_diffs > len(differences) * 0.8:
                return -1.0  # Strong negative trend
            else:
                return (positive_diffs - negative_diffs) / len(differences)
        except:
            return 0.0
            
    def _extract_numeric_values(self, values: List[Any]) -> List[float]:
        numeric_values = []
        for v in values:
            try:
                if isinstance(v, (int, float)):
                    numeric_values.append(float(v))
                else:
                    numeric_values.append(float(str(v)))
            except:
                pass
        return numeric_values
        
    def _calculate_semantic_density(self, profile: SemanticProfile) -> float:
        components = []
        
        # Entropy component
        if profile.entropy_metrics:
            entropy_avg = statistics.mean(profile.entropy_metrics.values())
            components.append(entropy_avg)
            
        # Pattern component
        if profile.pattern_features:
            pattern_scores = []
            for feature_list in profile.pattern_features.values():
                if feature_list:
                    pattern_scores.extend(feature_list)
            if pattern_scores:
                pattern_avg = statistics.mean(pattern_scores)
                components.append(pattern_avg)
                
        # Behavioral component
        if profile.behavioral_indicators:
            behavior_avg = statistics.mean(abs(v) for v in profile.behavioral_indicators.values())
            components.append(behavior_avg)
            
        return statistics.mean(components) if components else 0.0
        
    def _calculate_complexity_score(self, profile: SemanticProfile) -> float:
        complexity_factors = []
        
        # Entropy complexity
        shannon_entropy = profile.entropy_metrics.get('shannon', 0.0)
        complexity_factors.append(shannon_entropy)
        
        # Pattern complexity
        if profile.pattern_features:
            pattern_diversity = len([score for scores in profile.pattern_features.values() 
                                   for score in scores if score > 0.1])
            complexity_factors.append(min(1.0, pattern_diversity / 20.0))
            
        # Statistical complexity
        if profile.statistical_properties:
            std_metrics = [v for k, v in profile.statistical_properties.items() if 'std' in k.lower()]
            if std_metrics:
                avg_std = statistics.mean(std_metrics)
                complexity_factors.append(min(1.0, avg_std / 10.0))
                
        return statistics.mean(complexity_factors) if complexity_factors else 0.0
        
    def _calculate_intelligence_score(self, profile: SemanticProfile) -> float:
        components = []
        
        # Semantic density contributes 40%
        components.append(profile.semantic_density * 0.4)
        
        # Complexity score contributes 30%
        components.append(profile.complexity_score * 0.3)
        
        # Pattern recognition contributes 20%
        if profile.pattern_features:
            max_pattern_scores = []
            for feature_list in profile.pattern_features.values():
                if feature_list:
                    max_pattern_scores.append(max(feature_list))
            if max_pattern_scores:
                pattern_component = statistics.mean(max_pattern_scores) * 0.2
                components.append(pattern_component)
                
        # Behavioral insights contribute 10%
        if profile.behavioral_indicators:
            behavior_insight = max(abs(v) for v in profile.behavioral_indicators.values()) * 0.1
            components.append(behavior_insight)
            
        return sum(components)
        
    def _calculate_confidence_level(self, profile: SemanticProfile, values: List[Any]) -> float:
        confidence_factors = []
        
        # Sample size factor
        sample_factor = min(1.0, len(values) / 1000.0)
        confidence_factors.append(sample_factor)
        
        # Consistency factor
        consistency = profile.behavioral_indicators.get('consistency', 0.0)
        confidence_factors.append(consistency)
        
        # Pattern strength factor
        if profile.pattern_features:
            max_patterns = [max(scores) if scores else 0.0 for scores in profile.pattern_features.values()]
            pattern_strength = max(max_patterns) if max_patterns else 0.0
            confidence_factors.append(pattern_strength)
            
        return statistics.mean(confidence_factors) if confidence_factors else 0.0
        
    def _derive_business_context(self, field_name: str, profile: SemanticProfile) -> Dict[str, Any]:
        context = {
            'domain_classification': 'unknown',
            'business_importance': 'medium',
            'data_sensitivity': 'standard',
            'operational_role': 'data_field'
        }
        
        # Determine domain from semantic patterns
        if profile.pattern_features and 'semantic_patterns' in profile.pattern_features:
            semantic_scores = profile.pattern_features['semantic_patterns']
            domain_names = list(self.pattern_detector.domain_patterns.keys())
            
            if len(semantic_scores) >= len(domain_names):
                max_score_idx = semantic_scores.index(max(semantic_scores))
                if max_score_idx < len(domain_names):
                    context['domain_classification'] = domain_names[max_score_idx]
                    
        # Determine business importance
        field_lower = field_name.lower()
        if any(keyword in field_lower for keyword in ['id', 'key', 'primary']):
            context['business_importance'] = 'high'
        elif any(keyword in field_lower for keyword in ['name', 'email', 'user']):
            context['business_importance'] = 'high'
            context['data_sensitivity'] = 'sensitive'
        elif any(keyword in field_lower for keyword in ['temp', 'cache', 'log']):
            context['business_importance'] = 'low'
            
        return context