#!/usr/bin/env python3

import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    PERCENTAGE = "percentage"
    COUNT = "count" 
    RATIO = "ratio"
    SCORE = "score"

@dataclass
class ValidationRule:
    name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    expected_type: MetricType = MetricType.PERCENTAGE
    required_columns: List[str] = None
    business_logic_check: Optional[callable] = None

@dataclass 
class ValidationResult:
    is_valid: bool
    confidence_score: float
    issues: List[str]
    recommendations: List[str]
    extracted_metrics: Dict[str, Any]

class AO1MetricValidator:
    def __init__(self):
        self.validation_rules = {
            'global_visibility': ValidationRule(
                name='Global Visibility Score',
                min_value=0.0,
                max_value=100.0, 
                expected_type=MetricType.PERCENTAGE,
                business_logic_check=self._validate_global_visibility_logic
            ),
            'platform_coverage': ValidationRule(
                name='Platform Coverage',
                min_value=0.0,
                max_value=100.0,
                expected_type=MetricType.PERCENTAGE,
                business_logic_check=self._validate_platform_coverage_logic
            ),
            'infrastructure_visibility': ValidationRule(
                name='Infrastructure Type Visibility', 
                min_value=0.0,
                max_value=100.0,
                expected_type=MetricType.PERCENTAGE,
                business_logic_check=self._validate_infrastructure_logic
            ),
            'log_role_coverage': ValidationRule(
                name='Log Role Coverage',
                min_value=0.0,
                max_value=100.0,
                expected_type=MetricType.PERCENTAGE,
                business_logic_check=self._validate_log_role_logic
            )
        }
        
        # Business intelligence thresholds
        self.business_thresholds = {
            'excellent_visibility': 95.0,
            'good_visibility': 85.0, 
            'acceptable_visibility': 70.0,
            'poor_visibility': 50.0,
            'critical_visibility': 30.0
        }
        
    def validate_ao1_metric(self, metric_name: str, query_results: List[Tuple], 
                           field_context: List[str]) -> ValidationResult:
        """Validate AO1 metric results make business sense"""
        
        if metric_name not in self.validation_rules:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=[f"Unknown metric type: {metric_name}"],
                recommendations=["Define validation rules for this metric"],
                extracted_metrics={}
            )
            
        rule = self.validation_rules[metric_name]
        
        # Basic data validation
        basic_validation = self._validate_basic_structure(query_results, rule)
        if not basic_validation.is_valid:
            return basic_validation
            
        # Extract metrics from results
        extracted_metrics = self._extract_metrics_from_results(query_results, rule.expected_type)
        
        # Business logic validation
        business_validation = self._validate_business_logic(extracted_metrics, rule, field_context)
        
        # Combine validations
        overall_confidence = (basic_validation.confidence_score + business_validation.confidence_score) / 2
        
        all_issues = basic_validation.issues + business_validation.issues
        all_recommendations = basic_validation.recommendations + business_validation.recommendations
        
        return ValidationResult(
            is_valid=basic_validation.is_valid and business_validation.is_valid,
            confidence_score=overall_confidence,
            issues=all_issues,
            recommendations=all_recommendations,
            extracted_metrics={**extracted_metrics, **business_validation.extracted_metrics}
        )
        
    def _validate_basic_structure(self, results: List[Tuple], rule: ValidationRule) -> ValidationResult:
        """Validate basic result structure and data types"""
        
        issues = []
        recommendations = []
        confidence_components = []
        
        # Check if we have results
        if not results:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=["Query returned no results"],
                recommendations=["Check if data exists in source tables", "Verify field mappings"],
                extracted_metrics={}
            )
            
        # Check result structure
        if len(results) == 0:
            issues.append("Empty result set")
            confidence_components.append(0.0)
        elif len(results) > 1000:
            issues.append(f"Very large result set ({len(results)} rows) - may need aggregation")
            recommendations.append("Consider adding GROUP BY or summary aggregation")
            confidence_components.append(0.4)
        else:
            confidence_components.append(0.8)
            
        # Check column structure
        if results and len(results[0]) < 2:
            issues.append("Query returned single column - visibility metrics typically need comparison values")
            recommendations.append("Ensure query returns both numerator and denominator for percentage calculations")
            confidence_components.append(0.3)
        else:
            confidence_components.append(0.8)
            
        # Check for numeric values in expected ranges
        numeric_values = []
        for row in results:
            for value in row:
                if isinstance(value, (int, float)):
                    numeric_values.append(value)
                    
        if not numeric_values:
            issues.append("No numeric values found in results")
            confidence_components.append(0.0)
        else:
            # Check value ranges
            for value in numeric_values:
                if rule.min_value is not None and value < rule.min_value:
                    issues.append(f"Value {value} below expected minimum {rule.min_value}")
                    confidence_components.append(0.3)
                elif rule.max_value is not None and value > rule.max_value:
                    issues.append(f"Value {value} above expected maximum {rule.max_value}")
                    confidence_components.append(0.4)
                else:
                    confidence_components.append(0.9)
                    
        overall_confidence = sum(confidence_components) / len(confidence_components) if confidence_components else 0.0
        is_valid = overall_confidence > 0.5 and not any("returned no results" in issue for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=overall_confidence,
            issues=issues,
            recommendations=recommendations,
            extracted_metrics={}
        )
        
    def _extract_metrics_from_results(self, results: List[Tuple], expected_type: MetricType) -> Dict[str, Any]:
        """Extract key metrics from query results"""
        
        metrics = {}
        
        if not results:
            return metrics
            
        # Look for percentage values (0-100 range)
        percentages = []
        counts = []
        ratios = []
        
        for row in results:
            for i, value in enumerate(row):
                if isinstance(value, (int, float)):
                    if 0 <= value <= 100:
                        percentages.append(value)
                    elif value > 100 and value < 1000000:  # Reasonable count range
                        counts.append(value)
                    elif 0 <= value <= 1:  # Decimal ratio
                        ratios.append(value * 100)  # Convert to percentage
                        
        # Extract primary metrics based on expected type
        if expected_type == MetricType.PERCENTAGE:
            if percentages:
                metrics['primary_percentage'] = percentages[0]
                metrics['all_percentages'] = percentages
                metrics['percentage_range'] = [min(percentages), max(percentages)] if len(percentages) > 1 else [percentages[0]]
                
        if counts:
            metrics['total_count'] = max(counts) if counts else 0  # Assume largest is total
            metrics['counts'] = counts
            
        # Try to identify numerator/denominator pairs
        if len(results) == 1 and len(results[0]) >= 2:
            row = results[0]
            numerics = [v for v in row if isinstance(v, (int, float))]
            if len(numerics) >= 2:
                # Assume first two numeric values are numerator/denominator
                numerator, denominator = numerics[0], numerics[1]
                if denominator > 0:
                    calculated_percentage = (numerator / denominator) * 100
                    metrics['calculated_percentage'] = calculated_percentage
                    metrics['numerator'] = numerator
                    metrics['denominator'] = denominator
                    
        return metrics
        
    def _validate_business_logic(self, metrics: Dict[str, Any], rule: ValidationRule, 
                                field_context: List[str]) -> ValidationResult:
        """Apply business logic validation"""
        
        if rule.business_logic_check:
            return rule.business_logic_check(metrics, field_context)
        else:
            return ValidationResult(
                is_valid=True,
                confidence_score=0.7,
                issues=[],
                recommendations=[], 
                extracted_metrics={}
            )
            
    def _validate_global_visibility_logic(self, metrics: Dict[str, Any], field_context: List[str]) -> ValidationResult:
        """Business logic validation for global visibility scores"""
        
        issues = []
        recommendations = []
        confidence_score = 0.5
        business_metrics = {}
        
        # Check if we have a reasonable percentage
        primary_pct = metrics.get('primary_percentage') or metrics.get('calculated_percentage')
        
        if primary_pct is None:
            issues.append("Could not identify primary visibility percentage")
            confidence_score = 0.2
        else:
            business_metrics['visibility_percentage'] = primary_pct
            
            # Business logic checks
            if primary_pct == 100.0:
                issues.append("100% visibility is suspicious - may indicate data quality issue")
                recommendations.append("Verify that silent/offline assets are properly excluded")
                confidence_score = 0.6
            elif primary_pct > 95.0:
                business_metrics['visibility_grade'] = 'EXCELLENT'
                confidence_score = 0.9
            elif primary_pct > 85.0:
                business_metrics['visibility_grade'] = 'GOOD'
                confidence_score = 0.9
            elif primary_pct > 70.0:
                business_metrics['visibility_grade'] = 'NEEDS_IMPROVEMENT' 
                confidence_score = 0.8
            elif primary_pct > 50.0:
                business_metrics['visibility_grade'] = 'POOR'
                recommendations.append("Urgent action required to improve logging coverage")
                confidence_score = 0.8
            else:
                business_metrics['visibility_grade'] = 'CRITICAL'
                recommendations.append("Critical visibility gaps - immediate investigation required")
                confidence_score = 0.9  # High confidence in critical finding
                
            # Check for reasonable asset counts
            total_assets = metrics.get('denominator') or metrics.get('total_count')
            visible_assets = metrics.get('numerator')
            
            if total_assets:
                business_metrics['total_assets'] = total_assets
                
                if total_assets < 10:
                    issues.append(f"Very low total asset count ({total_assets}) - may indicate incomplete data")
                    confidence_score *= 0.7
                elif total_assets > 100000:
                    issues.append(f"Very high asset count ({total_assets}) - verify scope is correct")
                    recommendations.append("Consider segmenting analysis by business unit or region")
                    
                if visible_assets:
                    business_metrics['visible_assets'] = visible_assets
                    business_metrics['silent_assets'] = total_assets - visible_assets
                    
        # Field context validation
        has_cmdb_fields = any('cmdb' in field.lower() for field in field_context)
        has_logging_fields = any(any(kw in field.lower() for kw in ['log', 'event', 'source']) for field in field_context)
        has_asset_fields = any(any(kw in field.lower() for kw in ['asset', 'host', 'device']) for field in field_context)
        
        field_score_components = []
        if has_cmdb_fields:
            field_score_components.append(0.9)
            recommendations.append("CMDB-based calculation detected - good approach for comprehensive asset inventory")
        if has_logging_fields:
            field_score_components.append(0.8)
        if has_asset_fields:
            field_score_components.append(0.8)
            
        if field_score_components:
            field_confidence = sum(field_score_components) / len(field_score_components)
            confidence_score = (confidence_score + field_confidence) / 2
        else:
            issues.append("No clear asset or logging fields identified in query")
            confidence_score *= 0.5
            
        return ValidationResult(
            is_valid=confidence_score > 0.4,
            confidence_score=confidence_score,
            issues=issues,
            recommendations=recommendations,
            extracted_metrics=business_metrics
        )
        
    def _validate_platform_coverage_logic(self, metrics: Dict[str, Any], field_context: List[str]) -> ValidationResult:
        """Business logic validation for platform coverage"""
        
        issues = []
        recommendations = []
        confidence_score = 0.6
        business_metrics = {}
        
        # Platform coverage should show multiple platforms
        all_percentages = metrics.get('all_percentages', [])
        
        if len(all_percentages) < 2:
            issues.append("Expected multiple platform coverage percentages")
            recommendations.append("Verify query groups by platform/source")
            confidence_score = 0.4
        else:
            business_metrics['platform_count'] = len(all_percentages)
            business_metrics['coverage_range'] = [min(all_percentages), max(all_percentages)]
            
            # Check for platform coverage gaps
            low_coverage_platforms = [p for p in all_percentages if p < 50]
            if low_coverage_platforms:
                business_metrics['platforms_with_low_coverage'] = len(low_coverage_platforms)
                recommendations.append(f"{len(low_coverage_platforms)} platforms have <50% coverage - investigate ingestion issues")
                
            # Check for reasonable distribution
            coverage_variance = max(all_percentages) - min(all_percentages)
            if coverage_variance > 60:
                issues.append("Large variance in platform coverage - may indicate data quality issues")
                business_metrics['coverage_variance'] = coverage_variance
                
        return ValidationResult(
            is_valid=confidence_score > 0.4,
            confidence_score=confidence_score,
            issues=issues,
            recommendations=recommendations,
            extracted_metrics=business_metrics
        )
        
    def _validate_infrastructure_logic(self, metrics: Dict[str, Any], field_context: List[str]) -> ValidationResult:
        """Business logic validation for infrastructure type coverage"""
        
        issues = []
        recommendations = []
        confidence_score = 0.6
        business_metrics = {}
        
        # Should have multiple infrastructure types
        all_percentages = metrics.get('all_percentages', [])
        
        if not all_percentages:
            return ValidationResult(False, 0.2, ["No infrastructure coverage percentages found"], [], {})
            
        business_metrics['infrastructure_types'] = len(all_percentages)
        
        # Expected infrastructure types in modern environments
        if len(all_percentages) < 3:
            recommendations.append("Consider breaking down by Cloud, On-Prem, and SaaS infrastructure types")
        
        # Check for balanced coverage
        avg_coverage = sum(all_percentages) / len(all_percentages)
        business_metrics['average_infrastructure_coverage'] = avg_coverage
        
        if avg_coverage < 70:
            issues.append("Low average infrastructure coverage across types")
            recommendations.append("Focus on improving logging for underperforming infrastructure types")
        
        return ValidationResult(
            is_valid=True,
            confidence_score=confidence_score,
            issues=issues,
            recommendations=recommendations,
            extracted_metrics=business_metrics
        )
        
    def _validate_log_role_coverage_logic(self, metrics: Dict[str, Any], field_context: List[str]) -> ValidationResult:
        """Business logic validation for log role coverage"""
        
        issues = []
        recommendations = []
        confidence_score = 0.7
        business_metrics = {}
        
        all_percentages = metrics.get('all_percentages', [])
        
        if all_percentages:
            # Role coverage should show variation based on role complexity
            role_coverage_avg = sum(all_percentages) / len(all_percentages)
            business_metrics['average_role_coverage'] = role_coverage_avg
            
            # Identity and Network roles typically have higher coverage
            # Endpoint and Application roles often have gaps
            if role_coverage_avg > 80:
                business_metrics['role_coverage_grade'] = 'GOOD'
            elif role_coverage_avg > 60:
                business_metrics['role_coverage_grade'] = 'ACCEPTABLE'  
            else:
                business_metrics['role_coverage_grade'] = 'NEEDS_IMPROVEMENT'
                recommendations.append("Focus on improving log coverage for critical roles")
        
        return ValidationResult(
            is_valid=True,
            confidence_score=confidence_score,
            issues=issues,
            recommendations=recommendations,
            extracted_metrics=business_metrics
        )