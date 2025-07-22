#!/usr/bin/env python3

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from models import FieldIntelligence
from database_connector import DatabaseConnector
from ao1_query_builder import AO1QueryBuilder, AO1Query
from ao1_result_validator import AO1ResultValidator, ValidationResult

logger = logging.getLogger(__name__)

@dataclass
class AO1Metric:
    name: str
    query: AO1Query
    results: List[tuple]
    validation: ValidationResult
    business_value: str

class AO1Engine:
    """
    Main engine that uses your semantic intelligence to build actual working AO1 queries.
    Focuses ONLY on the specific visibility metrics your documentation requires.
    """
    
    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        self.query_builder = AO1QueryBuilder()
        self.validator = AO1ResultValidator()
        
    def generate_ao1_dashboard(self, high_intelligence_fields: List[FieldIntelligence]) -> Dict[str, AO1Metric]:
        """Generate complete AO1 dashboard using semantically intelligent fields"""
        
        logger.info(f"🎯 Building AO1 Dashboard from {len(high_intelligence_fields)} intelligent fields")
        
        dashboard = {}
        
        # 1. Global Visibility Score - THE #1 METRIC
        logger.info("📊 Building Global Visibility Score...")
        global_metric = self._build_metric('global_visibility', high_intelligence_fields)
        if global_metric:
            dashboard['global_visibility'] = global_metric
            logger.info(f"✅ Global Visibility: {global_metric.validation.business_assessment}")
        
        # 2. Platform Coverage
        logger.info("🔧 Building Platform Coverage...")
        platform_metric = self._build_metric('platform_coverage', high_intelligence_fields)
        if platform_metric:
            dashboard['platform_coverage'] = platform_metric
            logger.info(f"✅ Platform Coverage: {platform_metric.validation.business_assessment}")
        
        # 3. Silent Assets (Critical for Security)
        logger.info("🔇 Building Silent Assets Analysis...")
        silent_metric = self._build_metric('silent_assets', high_intelligence_fields)
        if silent_metric:
            dashboard['silent_assets'] = silent_metric
            logger.info(f"✅ Silent Assets: {silent_metric.validation.business_assessment}")
        
        # 4. Infrastructure Visibility
        logger.info("🏗️ Building Infrastructure Visibility...")
        infra_metric = self._build_metric('infrastructure_visibility', high_intelligence_fields)
        if infra_metric:
            dashboard['infrastructure_visibility'] = infra_metric
            logger.info(f"✅ Infrastructure Visibility: {infra_metric.validation.business_assessment}")
        
        success_count = len(dashboard)
        total_metrics = 4
        
        logger.info(f"🎯 AO1 Dashboard Complete: {success_count}/{total_metrics} metrics generated")
        
        return dashboard
    
    def _build_metric(self, metric_type: str, fields: List[FieldIntelligence]) -> Optional[AO1Metric]:
        """Build a specific AO1 metric using semantic intelligence"""
        
        # Use query builder to generate appropriate query
        query = None
        if metric_type == 'global_visibility':
            query = self.query_builder.build_global_visibility_query(fields)
        elif metric_type == 'platform_coverage':
            query = self.query_builder.build_platform_coverage_query(fields)
        elif metric_type == 'silent_assets':
            query = self.query_builder.build_silent_assets_query(fields)
        elif metric_type == 'infrastructure_visibility':
            query = self.query_builder.build_infrastructure_visibility_query(fields)
        
        if not query:
            logger.warning(f"Could not generate {metric_type} query - missing required fields")
            return None
        
        # Execute query
        try:
            logger.debug(f"Executing {metric_type} query...")
            results = self.db_connector.execute_query(query.sql)
            
            if not results:
                logger.warning(f"{metric_type} query returned no results")
                return None
                
            logger.debug(f"{metric_type} query returned {len(results)} rows")
            
        except Exception as e:
            logger.error(f"Failed to execute {metric_type} query: {e}")
            return None
        
        # Validate results with business logic
        try:
            if metric_type == 'global_visibility':
                validation = self.validator.validate_global_visibility(results)
            elif metric_type == 'platform_coverage':
                validation = self.validator.validate_platform_coverage(results)
            elif metric_type == 'silent_assets':
                validation = self.validator.validate_silent_assets(results)
            elif metric_type == 'infrastructure_visibility':
                validation = self.validator.validate_infrastructure_visibility(results)
            else:
                validation = ValidationResult(True, 0.5, "UNKNOWN", [], [])
                
        except Exception as e:
            logger.error(f"Validation failed for {metric_type}: {e}")
            validation = ValidationResult(False, 0.2, "ERROR", [f"Validation error: {e}"], [])
        
        # Determine business value
        business_value = self._assess_business_value(metric_type, validation)
        
        return AO1Metric(
            name=query.name,
            query=query,
            results=results,
            validation=validation,
            business_value=business_value
        )
    
    def _assess_business_value(self, metric_type: str, validation: ValidationResult) -> str:
        """Assess business value of this metric for executives"""
        
        if not validation.is_valid:
            return "LOW - Data quality issues"
        
        if metric_type == 'global_visibility':
            if validation.business_assessment in ['EXCELLENT', 'GOOD']:
                return "HIGH - Strong overall security posture"
            elif validation.business_assessment == 'NEEDS_IMPROVEMENT':
                return "MEDIUM - Visibility gaps to address"
            else:
                return "CRITICAL - Significant security blind spots"
                
        elif metric_type == 'platform_coverage':
            if validation.business_assessment == 'GOOD_DISTRIBUTION':
                return "HIGH - Well-distributed logging coverage"
            else:
                return "MEDIUM - Platform optimization opportunities"
                
        elif metric_type == 'silent_assets':
            if validation.business_assessment == 'OPTIMAL':
                return "HIGH - No security blind spots"
            elif validation.business_assessment == 'MANAGEABLE':
                return "MEDIUM - Manageable security gaps"
            else:
                return "CRITICAL - Major security visibility gaps"
                
        elif metric_type == 'infrastructure_visibility':
            if validation.business_assessment == 'GOOD_COVERAGE':
                return "HIGH - Comprehensive infrastructure monitoring"
            else:
                return "MEDIUM - Infrastructure monitoring improvements needed"
        
        return "MEDIUM - Standard business value"
    
    def generate_executive_report(self, dashboard: Dict[str, AO1Metric]) -> Dict[str, any]:
        """Generate executive report for your boss"""
        
        # Extract validation results for summary
        validations = {name: metric.validation for name, metric in dashboard.items()}
        summary = self.validator.generate_executive_summary(validations)
        
        # Add dashboard-specific insights
        executive_report = {
            'executive_summary': summary,
            'key_findings': [],
            'business_impact': {},
            'recommended_actions': []
        }
        
        # Process each metric for executive insights
        for metric_name, metric in dashboard.items():
            
            # Extract key business findings
            if metric_name == 'global_visibility' and metric.results:
                row = metric.results[0]
                total_assets = next((v for v in row if isinstance(v, int) and v > 100), 0)
                visibility_pct = next((v for v in row if isinstance(v, (int, float)) and 0 <= v <= 100), 0)
                
                executive_report['key_findings'].append(
                    f"🎯 Global Visibility: {visibility_pct:.1f}% of {total_assets:,} assets have logging visibility"
                )
                
                executive_report['business_impact']['security_coverage'] = f"{visibility_pct:.1f}%"
                
            elif metric_name == 'silent_assets' and metric.results:
                silent_count = len(metric.results)
                if silent_count > 0:
                    executive_report['key_findings'].append(
                        f"🔇 Security Risk: {silent_count:,} assets have zero logging (blind spots)"
                    )
                    executive_report['business_impact']['security_blind_spots'] = silent_count
                    
                    if silent_count > 50:
                        executive_report['recommended_actions'].append(
                            "URGENT: Deploy logging to silent assets - security vulnerability"
                        )
                        
            elif metric_name == 'platform_coverage' and metric.results:
                platform_count = len(metric.results)
                executive_report['key_findings'].append(
                    f"🔧 Platform Coverage: Monitoring across {platform_count} logging platforms"
                )
                
            # Add metric-specific recommendations
            executive_report['recommended_actions'].extend(metric.validation.recommendations)
        
        # Overall business assessment
        if summary['overall_status'] in ['EXCELLENT', 'GOOD']:
            executive_report['overall_recommendation'] = "MAINTAIN - Strong security monitoring posture"
        elif summary['overall_status'] == 'NEEDS_IMPROVEMENT':
            executive_report['overall_recommendation'] = "IMPROVE - Address identified visibility gaps"
        else:
            executive_report['overall_recommendation'] = "URGENT - Critical security monitoring deficiencies"
        
        return executive_report
    
    def save_sql_queries(self, dashboard: Dict[str, AO1Metric], output_file: str = "ao1_queries.sql"):
        """Save all working SQL queries to file"""
        
        with open(output_file, 'w') as f:
            f.write("-- AO1 Log Visibility Measurement Queries\n")
            f.write("-- Generated from Semantic Intelligence Analysis\n")
            f.write("-- These queries produce actual business metrics\n\n")
            
            for metric_name, metric in dashboard.items():
                f.write(f"-- {metric.name}\n")
                f.write(f"-- Business Logic: {metric.query.business_logic}\n")
                f.write(f"-- Expected Result: {metric.query.expected_result}\n")
                f.write(f"-- Validation: {metric.query.validation_check}\n")
                f.write(f"-- Business Value: {metric.business_value}\n")
                f.write(f"-- Assessment: {metric.validation.business_assessment}\n")
                f.write(metric.query.sql)
                f.write("\n\n" + "="*80 + "\n\n")
        
        logger.info(f"💾 Saved working AO1 queries to {output_file}")
        
        return output_file