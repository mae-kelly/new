#!/usr/bin/env python3

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    confidence_score: float
    business_assessment: str
    issues: List[str]
    recommendations: List[str]

class AO1ResultValidator:
    """
    Validates that AO1 query results make business sense.
    Knows what "good" vs "bad" visibility metrics look like.
    """
    
    def __init__(self):
        # Business intelligence thresholds based on industry standards
        self.visibility_thresholds = {
            'excellent': 90.0,
            'good': 75.0,
            'acceptable': 60.0,
            'poor': 40.0,
            'critical': 25.0
        }
        
    def validate_global_visibility(self, query_results: List[Tuple]) -> ValidationResult:
        """Validate global visibility results make business sense"""
        
        if not query_results:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                business_assessment="NO_DATA",
                issues=["Query returned no results"],
                recommendations=["Check data sources and field mappings"]
            )
        
        # Extract key metrics from first row
        row = query_results[0]
        total_assets = self._extract_number(row, ['total_assets', 'total'])
        visible_assets = self._extract_number(row, ['assets_with_logs', 'visible'])
        visibility_pct = self._extract_number(row, ['visibility_percentage', 'percentage'])
        
        issues = []
        recommendations = []
        
        # Validate asset counts make sense
        if total_assets and total_assets < 10:
            issues.append(f"Very low total asset count ({total_assets}) - may indicate incomplete inventory")
            
        if total_assets and total_assets > 100000:
            issues.append(f"Very high asset count ({total_assets}) - verify scope is correct")
            recommendations.append("Consider segmenting analysis by business unit")
            
        # Validate visibility percentage
        if visibility_pct is None:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.2,
                business_assessment="INVALID",
                issues=["Could not extract visibility percentage"],
                recommendations=["Check query output format"]
            )
            
        # Business assessment based on visibility percentage
        if visibility_pct >= self.visibility_thresholds['excellent']:
            assessment = "EXCELLENT"
            confidence = 0.9
        elif visibility_pct >= self.visibility_thresholds['good']:
            assessment = "GOOD"
            confidence = 0.9
            recommendations.append("Good visibility - monitor for any degradation")
        elif visibility_pct >= self.visibility_thresholds['acceptable']:
            assessment = "NEEDS_IMPROVEMENT"
            confidence = 0.8
            recommendations.append("Visibility gaps exist - prioritize improving logging coverage")
        elif visibility_pct >= self.visibility_thresholds['poor']:
            assessment = "POOR"
            confidence = 0.8
            issues.append("Significant visibility gaps detected")
            recommendations.append("URGENT: Investigate and remediate logging gaps")
        else:
            assessment = "CRITICAL"
            confidence = 0.9  # High confidence in critical finding
            issues.append("Critical visibility gaps - security risk")
            recommendations.append("IMMEDIATE ACTION REQUIRED: Review logging infrastructure")
            
        # Sanity checks
        if visibility_pct == 100.0:
            issues.append("100% visibility is suspicious - may indicate data quality issue")
            recommendations.append("Verify silent/offline assets are properly excluded")
            confidence *= 0.7
            
        if visibility_pct > 100.0:
            issues.append("Visibility percentage > 100% - data error")
            confidence = 0.2
            assessment = "INVALID"
            
        return ValidationResult(
            is_valid=confidence > 0.5,
            confidence_score=confidence,
            business_assessment=assessment,
            issues=issues,
            recommendations=recommendations
        )
    
    def validate_platform_coverage(self, query_results: List[Tuple]) -> ValidationResult:
        """Validate platform coverage results"""
        
        if not query_results:
            return ValidationResult(False, 0.0, "NO_DATA", ["No platform data"], [])
        
        platform_count = len(query_results)
        issues = []
        recommendations = []
        
        # Check for reasonable number of platforms
        if platform_count < 2:
            issues.append("Expected multiple logging platforms")
            recommendations.append("Verify data includes all logging systems")
            
        # Extract coverage percentages
        coverage_percentages = []
        for row in query_results:
            pct = self._extract_number(row, ['platform_coverage_pct', 'coverage_percentage'])
            if pct is not None:
                coverage_percentages.append(pct)
                
        if not coverage_percentages:
            return ValidationResult(False, 0.2, "INVALID", ["No coverage percentages found"], [])
        
        # Platform coverage analysis
        avg_coverage = sum(coverage_percentages) / len(coverage_percentages)
        max_coverage = max(coverage_percentages)
        min_coverage = min(coverage_percentages)
        
        low_coverage_platforms = [p for p in coverage_percentages if p < 30]
        
        if len(low_coverage_platforms) > 0:
            issues.append(f"{len(low_coverage_platforms)} platforms have <30% coverage")
            recommendations.append("Investigate ingestion issues for low-coverage platforms")
            
        # Assess platform distribution
        if max_coverage - min_coverage > 60:
            issues.append("Large variance in platform coverage - potential data quality issues")
            
        # Business assessment
        if avg_coverage >= 70:
            assessment = "GOOD_DISTRIBUTION"
            confidence = 0.8
        elif avg_coverage >= 50:
            assessment = "ADEQUATE_DISTRIBUTION"
            confidence = 0.7
        else:
            assessment = "POOR_DISTRIBUTION"
            confidence = 0.8
            recommendations.append("Review platform onboarding and data ingestion")
            
        return ValidationResult(
            is_valid=confidence > 0.5,
            confidence_score=confidence,
            business_assessment=assessment,
            issues=issues,
            recommendations=recommendations
        )
    
    def validate_silent_assets(self, query_results: List[Tuple]) -> ValidationResult:
        """Validate silent assets analysis"""
        
        silent_asset_count = len(query_results)
        
        # Business logic: Some silent assets expected, but not majority
        if silent_asset_count == 0:
            return ValidationResult(
                is_valid=True,
                confidence_score=0.9,
                business_assessment="OPTIMAL",
                issues=[],
                recommendations=["Excellent - no silent assets detected"]
            )
            
        # Estimate if this is reasonable (need total asset context)
        # For now, assess based on absolute count
        if silent_asset_count > 1000:
            assessment = "HIGH_RISK"
            issues = [f"Very large number of silent assets ({silent_asset_count})"]
            recommendations = ["URGENT: Mass logging deployment needed"]
            confidence = 0.9
        elif silent_asset_count > 100:
            assessment = "CONCERNING"
            issues = [f"Significant silent assets ({silent_asset_count})"]
            recommendations = ["Prioritize onboarding silent assets to logging"]
            confidence = 0.8
        else:
            assessment = "MANAGEABLE"
            issues = []
            recommendations = ["Review and onboard remaining silent assets"]
            confidence = 0.8
            
        return ValidationResult(
            is_valid=True,
            confidence_score=confidence,
            business_assessment=assessment,
            issues=issues,
            recommendations=recommendations
        )
    
    def validate_infrastructure_visibility(self, query_results: List[Tuple]) -> ValidationResult:
        """Validate infrastructure type visibility"""
        
        if not query_results:
            return ValidationResult(False, 0.0, "NO_DATA", ["No infrastructure data"], [])
        
        infra_types = len(query_results)
        coverage_data = []
        
        for row in query_results:
            coverage_pct = self._extract_number(row, ['coverage_percentage', 'coverage_pct'])
            if coverage_pct is not None:
                coverage_data.append(coverage_pct)
                
        if not coverage_data:
            return ValidationResult(False, 0.2, "INVALID", ["No coverage data"], [])
        
        avg_coverage = sum(coverage_data) / len(coverage_data)
        
        issues = []
        recommendations = []
        
        # Infrastructure-specific expectations
        if infra_types < 2:
            recommendations.append("Consider breaking down by Cloud/On-Prem/SaaS types")
            
        # Coverage assessment
        if avg_coverage >= 80:
            assessment = "GOOD_COVERAGE"
            confidence = 0.8
        elif avg_coverage >= 60:
            assessment = "ADEQUATE_COVERAGE"
            confidence = 0.7
        else:
            assessment = "POOR_COVERAGE"
            confidence = 0.8
            issues.append("Low infrastructure coverage across types")
            recommendations.append("Focus on improving logging for underperforming infrastructure")
            
        return ValidationResult(
            is_valid=True,
            confidence_score=confidence,
            business_assessment=assessment,
            issues=issues,
            recommendations=recommendations
        )
    
    def _extract_number(self, row: Tuple, possible_names: List[str]) -> float:
        """Extract numeric value from query result row"""
        
        # Try each value in the row
        for value in row:
            if isinstance(value, (int, float)):
                return float(value)
                
        # If no direct numeric values, return None
        return None
    
    def generate_executive_summary(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate executive summary for your boss"""
        
        summary = {
            'overall_status': 'UNKNOWN',
            'key_metrics': {},
            'critical_issues': [],
            'action_items': [],
            'confidence_level': 0.0
        }
        
        # Extract key metrics
        if 'global_visibility' in validation_results:
            global_val = validation_results['global_visibility']
            summary['key_metrics']['global_visibility_status'] = global_val.business_assessment
            summary['overall_status'] = global_val.business_assessment
            
        # Aggregate critical issues
        all_issues = []
        all_recommendations = []
        confidence_scores = []
        
        for metric_name, result in validation_results.items():
            all_issues.extend(result.issues)
            all_recommendations.extend(result.recommendations)
            confidence_scores.append(result.confidence_score)
            
        summary['critical_issues'] = list(set(all_issues))  # Remove duplicates
        summary['action_items'] = list(set(all_recommendations))
        summary['confidence_level'] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return summary