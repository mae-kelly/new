#!/usr/bin/env python3

import statistics
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from models import IntelligentQuery

@dataclass
class IntelligentValidationResult:
    is_valid: bool
    confidence_score: float
    business_assessment: str
    data_quality_score: float
    semantic_coherence: float
    issues: List[str]
    recommendations: List[str]
    extracted_metrics: Dict[str, Any]

class IntelligentValidator:
    def __init__(self):
        self.business_logic_validators = {
            'global_visibility': self._validate_global_visibility_logic,
            'platform_coverage': self._validate_platform_coverage_logic,
            'infrastructure_visibility': self._validate_infrastructure_logic,
            'silent_assets': self._validate_silent_assets_logic,
            'log_role_coverage': self._validate_log_role_logic
        }
        
        self.expected_ranges = {
            'visibility_percentage': (5.0, 99.0),
            'coverage_percentage': (1.0, 95.0),
            'asset_count': (10, 1000000),
            'silent_asset_ratio': (0.01, 0.50)
        }
        
    def validate_global_visibility_results(self, results: List[Tuple], query: IntelligentQuery) -> IntelligentValidationResult:
        if not results:
            return IntelligentValidationResult(
                is_valid=False,
                confidence_score=0.0,
                business_assessment="NO_DATA",
                data_quality_score=0.0,
                semantic_coherence=0.0,
                issues=["Query returned no results"],
                recommendations=["Verify data exists in source tables"],
                extracted_metrics={}
            )
            
        primary_result = results[0] if results else None
        if not primary_result:
            return self._create_invalid_result("Empty result set")
            
        extracted_metrics = self._extract_global_visibility_metrics(primary_result)
        data_quality = self._assess_data_quality(extracted_metrics, 'global_visibility')
        business_logic_validation = self._validate_business_logic(extracted_metrics, 'global_visibility')
        semantic_validation = self._validate_semantic_coherence(query, extracted_metrics)
        
        overall_confidence = statistics.mean([
            data_quality.confidence_score,
            business_logic_validation.confidence_score,
            semantic_validation.confidence_score
        ])
        
        is_valid = all([
            data_quality.is_valid,
            business_logic_validation.is_valid,
            overall_confidence > 0.4
        ])
        
        combined_issues = data_quality.issues + business_logic_validation.issues + semantic_validation.issues
        combined_recommendations = data_quality.recommendations + business_logic_validation.recommendations + semantic_validation.recommendations
        
        business_assessment = self._determine_business_assessment(extracted_metrics, 'global_visibility')
        
        return IntelligentValidationResult(
            is_valid=is_valid,
            confidence_score=overall_confidence,
            business_assessment=business_assessment,
            data_quality_score=data_quality.confidence_score,
            semantic_coherence=semantic_validation.confidence_score,
            issues=combined_issues,
            recommendations=combined_recommendations,
            extracted_metrics=extracted_metrics
        )
        
    def validate_platform_coverage_results(self, results: List[Tuple], query: IntelligentQuery) -> IntelligentValidationResult:
        if not results:
            return self._create_invalid_result("No platform coverage data")
            
        extracted_metrics = self._extract_platform_coverage_metrics(results)
        data_quality = self._assess_data_quality(extracted_metrics, 'platform_coverage')
        business_logic_validation = self._validate_business_logic(extracted_metrics, 'platform_coverage')
        semantic_validation = self._validate_semantic_coherence(query, extracted_metrics)
        
        overall_confidence = statistics.mean([
            data_quality.confidence_score,
            business_logic_validation.confidence_score,
            semantic_validation.confidence_score
        ])
        
        is_valid = data_quality.is_valid and business_logic_validation.is_valid
        
        return IntelligentValidationResult(
            is_valid=is_valid,
            confidence_score=overall_confidence,
            business_assessment=self._determine_business_assessment(extracted_metrics, 'platform_coverage'),
            data_quality_score=data_quality.confidence_score,
            semantic_coherence=semantic_validation.confidence_score,
            issues=data_quality.issues + business_logic_validation.issues,
            recommendations=data_quality.recommendations + business_logic_validation.recommendations,
            extracted_metrics=extracted_metrics
        )
        
    def validate_infrastructure_visibility_results(self, results: List[Tuple], query: IntelligentQuery) -> IntelligentValidationResult:
        if not results:
            return self._create_invalid_result("No infrastructure visibility data")
            
        extracted_metrics = self._extract_infrastructure_metrics(results)
        data_quality = self._assess_data_quality(extracted_metrics, 'infrastructure_visibility')
        business_logic_validation = self._validate_business_logic(extracted_metrics, 'infrastructure_visibility')
        semantic_validation = self._validate_semantic_coherence(query, extracted_metrics)
        
        overall_confidence = statistics.mean([
            data_quality.confidence_score,
            business_logic_validation.confidence_score,
            semantic_validation.confidence_score
        ])
        
        return IntelligentValidationResult(
            is_valid=data_quality.is_valid and business_logic_validation.is_valid,
            confidence_score=overall_confidence,
            business_assessment=self._determine_business_assessment(extracted_metrics, 'infrastructure_visibility'),
            data_quality_score=data_quality.confidence_score,
            semantic_coherence=semantic_validation.confidence_score,
            issues=data_quality.issues + business_logic_validation.issues,
            recommendations=data_quality.recommendations + business_logic_validation.recommendations,
            extracted_metrics=extracted_metrics
        )
        
    def validate_silent_assets_results(self, results: List[Tuple], query: IntelligentQuery) -> IntelligentValidationResult:
        extracted_metrics = self._extract_silent_assets_metrics(results)
        data_quality = self._assess_data_quality(extracted_metrics, 'silent_assets')
        business_logic_validation = self._validate_business_logic(extracted_metrics, 'silent_assets')
        semantic_validation = self._validate_semantic_coherence(query, extracted_metrics)
        
        overall_confidence = statistics.mean([
            data_quality.confidence_score,
            business_logic_validation.confidence_score,
            semantic_validation.confidence_score
        ])
        
        return IntelligentValidationResult(
            is_valid=data_quality.is_valid and business_logic_validation.is_valid,
            confidence_score=overall_confidence,
            business_assessment=self._determine_business_assessment(extracted_metrics, 'silent_assets'),
            data_quality_score=data_quality.confidence_score,
            semantic_coherence=semantic_validation.confidence_score,
            issues=data_quality.issues + business_logic_validation.issues,
            recommendations=data_quality.recommendations + business_logic_validation.recommendations,
            extracted_metrics=extracted_metrics
        )
        
    def validate_log_role_coverage_results(self, results: List[Tuple], query: IntelligentQuery) -> IntelligentValidationResult:
        if not results:
            return self._create_invalid_result("No log role coverage data")
            
        extracted_metrics = self._extract_log_role_metrics(results)
        data_quality = self._assess_data_quality(extracted_metrics, 'log_role_coverage')
        business_logic_validation = self._validate_business_logic(extracted_metrics, 'log_role_coverage')
        semantic_validation = self._validate_semantic_coherence(query, extracted_metrics)
        
        overall_confidence = statistics.mean([
            data_quality.confidence_score,
            business_logic_validation.confidence_score,
            semantic_validation.confidence_score
        ])
        
        return IntelligentValidationResult(
            is_valid=data_quality.is_valid and business_logic_validation.is_valid,
            confidence_score=overall_confidence,
            business_assessment=self._determine_business_assessment(extracted_metrics, 'log_role_coverage'),
            data_quality_score=data_quality.confidence_score,
            semantic_coherence=semantic_validation.confidence_score,
            issues=data_quality.issues + business_logic_validation.issues,
            recommendations=data_quality.recommendations + business_logic_validation.recommendations,
            extracted_metrics=extracted_metrics
        )
        
    def _extract_global_visibility_metrics(self, result_row: Tuple) -> Dict[str, Any]:
        metrics = {}
        
        for i, value in enumerate(result_row):
            if isinstance(value, (int, float)):
                if 0 <= value <= 100:
                    metrics['visibility_percentage'] = float(value)
                elif value > 100:
                    if 'total_assets' not in metrics:
                        metrics['total_assets'] = int(value)
                    elif 'visible_assets' not in metrics and value <= metrics.get('total_assets', float('inf')):
                        metrics['visible_assets'] = int(value)
                    elif 'silent_assets' not in metrics:
                        metrics['silent_assets'] = int(value)
            elif isinstance(value, str):
                if value.upper() in ['EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR', 'CRITICAL']:
                    metrics['visibility_grade'] = value.upper()
                elif 'method' in str(result_row):
                    metrics['calculation_method'] = value
                    
        if 'total_assets' in metrics and 'visible_assets' in metrics:
            if 'silent_assets' not in metrics:
                metrics['silent_assets'] = metrics['total_assets'] - metrics['visible_assets']
            if 'visibility_percentage' not in metrics:
                metrics['visibility_percentage'] = (metrics['visible_assets'] / metrics['total_assets']) * 100 if metrics['total_assets'] > 0 else 0.0
                
        return metrics
        
    def _extract_platform_coverage_metrics(self, results: List[Tuple]) -> Dict[str, Any]:
        metrics = {
            'platforms': [],
            'coverage_data': [],
            'total_platforms': len(results),
            'platform_names': [],
            'coverage_percentages': [],
            'total_assets_per_platform': []
        }
        
        for row in results:
            if len(row) >= 4:
                platform_name = str(row[0])
                assets_on_platform = int(row[1]) if isinstance(row[1], (int, float)) else 0
                total_assets = int(row[2]) if isinstance(row[2], (int, float)) else 0
                coverage_pct = float(row[3]) if isinstance(row[3], (int, float)) else 0.0
                
                metrics['platform_names'].append(platform_name)
                metrics['coverage_percentages'].append(coverage_pct)
                metrics['total_assets_per_platform'].append(assets_on_platform)
                
                platform_data = {
                    'platform': platform_name,
                    'assets': assets_on_platform,
                    'total_assets': total_assets,
                    'coverage_percentage': coverage_pct
                }
                metrics['coverage_data'].append(platform_data)
                
        if metrics['coverage_percentages']:
            metrics['average_coverage'] = statistics.mean(metrics['coverage_percentages'])
            metrics['max_coverage'] = max(metrics['coverage_percentages'])
            metrics['min_coverage'] = min(metrics['coverage_percentages'])
            metrics['coverage_variance'] = statistics.variance(metrics['coverage_percentages']) if len(metrics['coverage_percentages']) > 1 else 0.0
            
        return metrics
        
    def _extract_infrastructure_metrics(self, results: List[Tuple]) -> Dict[str, Any]:
        metrics = {
            'infrastructure_types': [],
            'visibility_data': [],
            'total_types': len(results),
            'type_names': [],
            'visibility_percentages': [],
            'asset_counts': []
        }
        
        for row in results:
            if len(row) >= 4:
                infra_type = str(row[0])
                total_assets = int(row[1]) if isinstance(row[1], (int, float)) else 0
                visible_assets = int(row[2]) if isinstance(row[2], (int, float)) else 0
                visibility_pct = float(row[3]) if isinstance(row[3], (int, float)) else 0.0
                
                metrics['type_names'].append(infra_type)
                metrics['visibility_percentages'].append(visibility_pct)
                metrics['asset_counts'].append(total_assets)
                
                infra_data = {
                    'type': infra_type,
                    'total_assets': total_assets,
                    'visible_assets': visible_assets,
                    'visibility_percentage': visibility_pct
                }
                metrics['visibility_data'].append(infra_data)
                
        if metrics['visibility_percentages']:
            metrics['average_visibility'] = statistics.mean(metrics['visibility_percentages'])
            metrics['total_assets_all_types'] = sum(metrics['asset_counts'])
            
        return metrics
        
    def _extract_silent_assets_metrics(self, results: List[Tuple]) -> Dict[str, Any]:
        metrics = {
            'silent_assets': [],
            'total_silent': len(results),
            'risk_levels': [],
            'asset_ids': []
        }
        
        risk_counts = {}
        for row in results:
            if len(row) >= 3:
                asset_id = str(row[0])
                status = str(row[1])
                risk_level = str(row[2])
                
                metrics['asset_ids'].append(asset_id)
                metrics['risk_levels'].append(risk_level)
                
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
                
                metrics['silent_assets'].append({
                    'asset_id': asset_id,
                    'status': status,
                    'risk_level': risk_level
                })
                
        metrics['risk_breakdown'] = risk_counts
        
        if metrics['total_silent'] > 0:
            metrics['high_risk_count'] = risk_counts.get('HIGH_RISK', 0)
            metrics['medium_risk_count'] = risk_counts.get('MEDIUM_RISK', 0)
            metrics['high_risk_percentage'] = (metrics['high_risk_count'] / metrics['total_silent']) * 100
            
        return metrics
        
    def _extract_log_role_metrics(self, results: List[Tuple]) -> Dict[str, Any]:
        metrics = {
            'roles': [],
            'coverage_data': [],
            'total_roles': len(results),
            'role_names': [],
            'coverage_percentages': [],
            'asset_counts': []
        }
        
        for row in results:
            if len(row) >= 4:
                role_name = str(row[0])
                total_assets = int(row[1]) if isinstance(row[1], (int, float)) else 0
                assets_with_logs = int(row[2]) if isinstance(row[2], (int, float)) else 0
                coverage_pct = float(row[3]) if isinstance(row[3], (int, float)) else 0.0
                
                metrics['role_names'].append(role_name)
                metrics['coverage_percentages'].append(coverage_pct)
                metrics['asset_counts'].append(total_assets)
                
                role_data = {
                    'role': role_name,
                    'total_assets': total_assets,
                    'assets_with_logs': assets_with_logs,
                    'coverage_percentage': coverage_pct
                }
                metrics['coverage_data'].append(role_data)
                
        if metrics['coverage_percentages']:
            metrics['average_role_coverage'] = statistics.mean(metrics['coverage_percentages'])
            metrics['best_covered_role'] = max(zip(metrics['role_names'], metrics['coverage_percentages']), key=lambda x: x[1])
            metrics['worst_covered_role'] = min(zip(metrics['role_names'], metrics['coverage_percentages']), key=lambda x: x[1])
            
        return metrics
        
    def _assess_data_quality(self, metrics: Dict[str, Any], metric_type: str) -> IntelligentValidationResult:
        quality_score = 0.0
        issues = []
        recommendations = []
        
        if metric_type == 'global_visibility':
            if 'visibility_percentage' in metrics:
                vis_pct = metrics['visibility_percentage']
                if 0 <= vis_pct <= 100:
                    quality_score += 0.4
                else:
                    issues.append(f"Invalid visibility percentage: {vis_pct}")
                    
                if 'total_assets' in metrics:
                    total = metrics['total_assets']
                    if 10 <= total <= 1000000:
                        quality_score += 0.3
                    else:
                        issues.append(f"Unusual total asset count: {total}")
                        
                if 'visible_assets' in metrics and 'total_assets' in metrics:
                    if metrics['visible_assets'] <= metrics['total_assets']:
                        quality_score += 0.3
                    else:
                        issues.append("Visible assets exceed total assets")
                        
        elif metric_type == 'platform_coverage':
            if 'total_platforms' in metrics:
                if metrics['total_platforms'] >= 1:
                    quality_score += 0.3
                    if metrics['total_platforms'] >= 2:
                        quality_score += 0.2
                else:
                    issues.append("No platforms found")
                    
            if 'coverage_percentages' in metrics and metrics['coverage_percentages']:
                valid_percentages = [p for p in metrics['coverage_percentages'] if 0 <= p <= 100]
                if len(valid_percentages) == len(metrics['coverage_percentages']):
                    quality_score += 0.3
                else:
                    issues.append("Invalid coverage percentages detected")
                    
                if valid_percentages:
                    avg_coverage = statistics.mean(valid_percentages)
                    if avg_coverage > 1.0:
                        quality_score += 0.2
                        
        elif metric_type == 'silent_assets':
            if 'total_silent' in metrics:
                total_silent = metrics['total_silent']
                if total_silent >= 0:
                    quality_score += 0.4
                    if total_silent == 0:
                        quality_score += 0.3
                        recommendations.append("Excellent - no silent assets detected")
                    elif total_silent < 100:
                        quality_score += 0.2
                        recommendations.append("Manageable number of silent assets")
                    else:
                        issues.append(f"Large number of silent assets: {total_silent}")
                        recommendations.append("Investigate widespread logging gaps")
                        
            if 'risk_breakdown' in metrics:
                high_risk = metrics['risk_breakdown'].get('HIGH_RISK', 0)
                if high_risk > 0:
                    if high_risk > 50:
                        issues.append(f"Many high-risk silent assets: {high_risk}")
                        recommendations.append("Prioritize high-risk asset logging deployment")
                    else:
                        quality_score += 0.1
                        
        is_valid = quality_score > 0.4 and len(issues) == 0
        
        return IntelligentValidationResult(
            is_valid=is_valid,
            confidence_score=quality_score,
            business_assessment="",
            data_quality_score=quality_score,
            semantic_coherence=0.0,
            issues=issues,
            recommendations=recommendations,
            extracted_metrics={}
        )
        
    def _validate_business_logic(self, metrics: Dict[str, Any], metric_type: str) -> IntelligentValidationResult:
        validator = self.business_logic_validators.get(metric_type)
        if validator:
            return validator(metrics)
        else:
            return IntelligentValidationResult(
                is_valid=True,
                confidence_score=0.5,
                business_assessment="UNKNOWN",
                data_quality_score=0.5,
                semantic_coherence=0.0,
                issues=[],
                recommendations=[],
                extracted_metrics={}
            )
            
    def _validate_global_visibility_logic(self, metrics: Dict[str, Any]) -> IntelligentValidationResult:
        issues = []
        recommendations = []
        confidence_score = 0.5
        
        visibility_pct = metrics.get('visibility_percentage', 0)
        total_assets = metrics.get('total_assets', 0)
        
        if visibility_pct == 100.0:
            issues.append("100% visibility is suspicious - may indicate data quality issue")
            recommendations.append("Verify silent/offline assets are properly excluded")
            confidence_score = 0.6
        elif visibility_pct >= 95.0:
            confidence_score = 0.95
            recommendations.append("Excellent visibility - maintain current monitoring")
        elif visibility_pct >= 85.0:
            confidence_score = 0.9
            recommendations.append("Good visibility with minor gaps to address")
        elif visibility_pct >= 70.0:
            confidence_score = 0.8
            recommendations.append("Acceptable visibility - focus on gap reduction")
        elif visibility_pct >= 50.0:
            confidence_score = 0.7
            issues.append("Significant visibility gaps detected")
            recommendations.append("PRIORITY: Investigate and remediate logging gaps")
        else:
            confidence_score = 0.6
            issues.append("Critical visibility gaps - major security risk")
            recommendations.append("URGENT: Comprehensive logging infrastructure review required")
            
        if total_assets > 0:
            if total_assets < 50:
                issues.append("Very small asset inventory - may be incomplete")
                confidence_score *= 0.8
            elif total_assets > 100000:
                recommendations.append("Large environment - consider regional segmentation")
                
        return IntelligentValidationResult(
            is_valid=len(issues) == 0 or (len(issues) == 1 and "suspicious" not in issues[0]),
            confidence_score=confidence_score,
            business_assessment="",
            data_quality_score=0.0,
            semantic_coherence=0.0,
            issues=issues,
            recommendations=recommendations,
            extracted_metrics={}
        )
        
    def _validate_platform_coverage_logic(self, metrics: Dict[str, Any]) -> IntelligentValidationResult:
        issues = []
        recommendations = []
        confidence_score = 0.7
        
        total_platforms = metrics.get('total_platforms', 0)
        coverage_percentages = metrics.get('coverage_percentages', [])
        
        if total_platforms < 2:
            issues.append("Expected multiple logging platforms")
            recommendations.append("Verify all logging systems are included in analysis")
            confidence_score = 0.5
        elif total_platforms > 10:
            recommendations.append("Many platforms detected - consider consolidation strategy")
            
        if coverage_percentages:
            avg_coverage = statistics.mean(coverage_percentages)
            variance = statistics.variance(coverage_percentages) if len(coverage_percentages) > 1 else 0
            
            low_coverage_count = len([p for p in coverage_percentages if p < 30])
            
            if low_coverage_count > 0:
                issues.append(f"{low_coverage_count} platforms have low coverage (<30%)")
                recommendations.append("Investigate ingestion issues for underperforming platforms")
                
            if variance > 900:
                issues.append("High variance in platform coverage suggests data quality issues")
                recommendations.append("Review data collection consistency across platforms")
                confidence_score *= 0.8
                
            if avg_coverage >= 70:
                confidence_score = 0.9
            elif avg_coverage >= 50:
                confidence_score = 0.75
            else:
                confidence_score = 0.6
                issues.append("Overall platform coverage is low")
                
        return IntelligentValidationResult(
            is_valid=len(issues) <= 2,
            confidence_score=confidence_score,
            business_assessment="",
            data_quality_score=0.0,
            semantic_coherence=0.0,
            issues=issues,
            recommendations=recommendations,
            extracted_metrics={}
        )
        
    def _validate_infrastructure_logic(self, metrics: Dict[str, Any]) -> IntelligentValidationResult:
        issues = []
        recommendations = []
        confidence_score = 0.7
        
        total_types = metrics.get('total_types', 0)
        visibility_percentages = metrics.get('visibility_percentages', [])
        
        if total_types < 2:
            recommendations.append("Consider breaking down by Cloud/On-Prem/SaaS types")
            confidence_score = 0.6
            
        if visibility_percentages:
            avg_visibility = statistics.mean(visibility_percentages)
            
            if avg_visibility >= 85:
                confidence_score = 0.9
                recommendations.append("Excellent infrastructure visibility")
            elif avg_visibility >= 70:
                confidence_score = 0.8
                recommendations.append("Good infrastructure visibility")
            elif avg_visibility >= 50:
                confidence_score = 0.7
                recommendations.append("Moderate infrastructure visibility - room for improvement")
            else:
                confidence_score = 0.6
                issues.append("Low infrastructure visibility across types")
                recommendations.append("Focus on improving logging for underperforming infrastructure")
                
        return IntelligentValidationResult(
            is_valid=True,
            confidence_score=confidence_score,
            business_assessment="",
            data_quality_score=0.0,
            semantic_coherence=0.0,
            issues=issues,
            recommendations=recommendations,
            extracted_metrics={}
        )
        
    def _validate_silent_assets_logic(self, metrics: Dict[str, Any]) -> IntelligentValidationResult:
        issues = []
        recommendations = []
        confidence_score = 0.8
        
        total_silent = metrics.get('total_silent', 0)
        high_risk_count = metrics.get('high_risk_count', 0)
        
        if total_silent == 0:
            confidence_score = 0.95
            recommendations.append("Optimal - no silent assets detected")
        elif total_silent <= 10:
            confidence_score = 0.9
            recommendations.append("Very few silent assets - excellent monitoring")
        elif total_silent <= 50:
            confidence_score = 0.8
            recommendations.append("Manageable number of silent assets")
        elif total_silent <= 200:
            confidence_score = 0.7
            issues.append("Moderate number of silent assets detected")
            recommendations.append("Plan systematic onboarding of silent assets")
        else:
            confidence_score = 0.6
            issues.append(f"High number of silent assets: {total_silent}")
            recommendations.append("URGENT: Address widespread logging gaps")
            
        if high_risk_count > 0:
            if high_risk_count > total_silent * 0.5:
                issues.append("Majority of silent assets are high-risk")
                recommendations.append("Prioritize high-risk silent asset remediation")
                confidence_score *= 0.8
            else:
                recommendations.append("Focus on high-risk silent assets first")
                
        return IntelligentValidationResult(
            is_valid=total_silent < 1000,
            confidence_score=confidence_score,
            business_assessment="",
            data_quality_score=0.0,
            semantic_coherence=0.0,
            issues=issues,
            recommendations=recommendations,
            extracted_metrics={}
        )
        
    def _validate_log_role_logic(self, metrics: Dict[str, Any]) -> IntelligentValidationResult:
        issues = []
        recommendations = []
        confidence_score = 0.7
        
        total_roles = metrics.get('total_roles', 0)
        coverage_percentages = metrics.get('coverage_percentages', [])
        
        if total_roles < 2:
            issues.append("Expected multiple asset roles")
            recommendations.append("Verify role classification coverage")
            confidence_score = 0.6
            
        if coverage_percentages:
            avg_coverage = statistics.mean(coverage_percentages)
            min_coverage = min(coverage_percentages)
            
            poorly_covered_roles = len([p for p in coverage_percentages if p < 50])
            
            if poorly_covered_roles > 0:
                issues.append(f"{poorly_covered_roles} roles have poor log coverage")
                recommendations.append("Focus on improving logging for underperforming roles")
                
            if avg_coverage >= 85:
                confidence_score = 0.9
                recommendations.append("Excellent role-based log coverage")
            elif avg_coverage >= 70:
                confidence_score = 0.8
                recommendations.append("Good role-based log coverage")
            elif avg_coverage >= 50:
                confidence_score = 0.7
                recommendations.append("Moderate log coverage - target improvement")
            else:
                confidence_score = 0.6
                issues.append("Low overall role-based log coverage")
                recommendations.append("Comprehensive role-based logging strategy needed")
                
        return IntelligentValidationResult(
            is_valid=len(issues) <= 1,
            confidence_score=confidence_score,
            business_assessment="",
            data_quality_score=0.0,
            semantic_coherence=0.0,
            issues=issues,
            recommendations=recommendations,
            extracted_metrics={}
        )
        
    def _validate_semantic_coherence(self, query: IntelligentQuery, metrics: Dict[str, Any]) -> IntelligentValidationResult:
        coherence_score = 0.5
        issues = []
        recommendations = []
        
        if hasattr(query, 'semantic_coherence'):
            coherence_score = query.semantic_coherence
            
        if hasattr(query, 'field_combination') and query.field_combination:
            field_count = len(query.field_combination)
            
            if field_count >= 3:
                coherence_score += 0.2
                recommendations.append("Multi-field analysis provides high confidence")
            elif field_count >= 2:
                coherence_score += 0.1
                recommendations.append("Two-field correlation analysis")
            else:
                coherence_score -= 0.1
                issues.append("Single-field analysis - lower confidence")
                
        if hasattr(query, 'intelligence_reasoning') and query.intelligence_reasoning:
            reasoning_quality = self._assess_reasoning_quality(query.intelligence_reasoning)
            coherence_score += reasoning_quality * 0.3
            
        if hasattr(query, 'business_logic') and query.business_logic:
            if len(query.business_logic) > 50:
                coherence_score += 0.1
            else:
                issues.append("Limited business logic documentation")
                
        coherence_score = min(1.0, max(0.0, coherence_score))
        
        return IntelligentValidationResult(
            is_valid=coherence_score > 0.4,
            confidence_score=coherence_score,
            business_assessment="",
            data_quality_score=0.0,
            semantic_coherence=coherence_score,
            issues=issues,
            recommendations=recommendations,
            extracted_metrics={}
        )
        
    def _assess_reasoning_quality(self, reasoning: Dict[str, Any]) -> float:
        quality_score = 0.0
        
        if isinstance(reasoning, dict):
            if 'field1_semantic_score' in reasoning or 'asset_semantic_score' in reasoning:
                quality_score += 0.3
                
            if 'relationship_strength' in reasoning:
                relationship_strength = reasoning.get('relationship_strength', 0)
                if relationship_strength > 0.7:
                    quality_score += 0.4
                elif relationship_strength > 0.5:
                    quality_score += 0.2
                    
            if 'table_alignment' in reasoning:
                alignment = reasoning.get('table_alignment', 0)
                if alignment >= 1.0:
                    quality_score += 0.2
                elif alignment >= 0.7:
                    quality_score += 0.1
                    
            evidence_count = len([k for k in reasoning.keys() if 'score' in k or 'confidence' in k])
            quality_score += min(0.3, evidence_count * 0.1)
            
        return min(1.0, quality_score)
        
    def _determine_business_assessment(self, metrics: Dict[str, Any], metric_type: str) -> str:
        if metric_type == 'global_visibility':
            visibility_pct = metrics.get('visibility_percentage', 0)
            
            if visibility_pct >= 95:
                return "EXCELLENT_VISIBILITY"
            elif visibility_pct >= 85:
                return "GOOD_VISIBILITY"
            elif visibility_pct >= 70:
                return "ACCEPTABLE_VISIBILITY"
            elif visibility_pct >= 50:
                return "POOR_VISIBILITY"
            else:
                return "CRITICAL_VISIBILITY_GAPS"
                
        elif metric_type == 'platform_coverage':
            avg_coverage = metrics.get('average_coverage', 0)
            total_platforms = metrics.get('total_platforms', 0)
            
            if avg_coverage >= 80 and total_platforms >= 3:
                return "EXCELLENT_PLATFORM_DISTRIBUTION"
            elif avg_coverage >= 60:
                return "GOOD_PLATFORM_COVERAGE"
            elif avg_coverage >= 40:
                return "MODERATE_PLATFORM_COVERAGE"
            else:
                return "POOR_PLATFORM_COVERAGE"
                
        elif metric_type == 'infrastructure_visibility':
            avg_visibility = metrics.get('average_visibility', 0)
            
            if avg_visibility >= 85:
                return "EXCELLENT_INFRASTRUCTURE_COVERAGE"
            elif avg_visibility >= 70:
                return "GOOD_INFRASTRUCTURE_COVERAGE"
            elif avg_visibility >= 50:
                return "MODERATE_INFRASTRUCTURE_COVERAGE"
            else:
                return "POOR_INFRASTRUCTURE_COVERAGE"
                
        elif metric_type == 'silent_assets':
            total_silent = metrics.get('total_silent', 0)
            high_risk_percentage = metrics.get('high_risk_percentage', 0)
            
            if total_silent == 0:
                return "OPTIMAL_NO_SILENT_ASSETS"
            elif total_silent <= 20:
                return "MINIMAL_SILENT_ASSETS"
            elif total_silent <= 100:
                return "MANAGEABLE_SILENT_ASSETS"
            elif high_risk_percentage > 70:
                return "HIGH_RISK_SILENT_ASSETS"
            else:
                return "CONCERNING_SILENT_ASSETS"
                
        elif metric_type == 'log_role_coverage':
            avg_coverage = metrics.get('average_role_coverage', 0)
            
            if avg_coverage >= 85:
                return "EXCELLENT_ROLE_COVERAGE"
            elif avg_coverage >= 70:
                return "GOOD_ROLE_COVERAGE"
            elif avg_coverage >= 50:
                return "MODERATE_ROLE_COVERAGE"
            else:
                return "POOR_ROLE_COVERAGE"
                
        return "UNKNOWN_ASSESSMENT"
        
    def _create_invalid_result(self, reason: str) -> IntelligentValidationResult:
        return IntelligentValidationResult(
            is_valid=False,
            confidence_score=0.0,
            business_assessment="INVALID",
            data_quality_score=0.0,
            semantic_coherence=0.0,
            issues=[reason],
            recommendations=["Verify query and data sources"],
            extracted_metrics={}
        )