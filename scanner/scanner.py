import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .connection import BigQueryConnection
from .semantic_analyzer import AdvancedSemanticAnalyzer, FieldAnalysis
from .query_generator import QueryGenerator, GeneratedQuery
from .data_validator import DataValidator, ValidationResult, TableQualityMetrics
from .config import RESULTS_OUTPUT_DIR, MIN_CONFIDENCE_SCORE

logger = logging.getLogger(__name__)

class AO1Scanner:
    def __init__(self, service_account_path: str = None, output_dir: str = None):
        self.output_dir = Path(output_dir or RESULTS_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        
        try:
            self.connection = BigQueryConnection(service_account_path)
            self.semantic_analyzer = AdvancedSemanticAnalyzer()
            self.query_generator = QueryGenerator()
            self.data_validator = DataValidator(self.connection)
            
            self.scan_results = {}
            self.discovered_queries = []
            self.validation_results = []
            
            logger.info("AO1Scanner initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AO1Scanner: {e}")
            raise
    
    def scan_all_datasets(self, max_workers: int = 3, quick_scan: bool = False) -> Dict[str, Any]:
        """Scan all datasets with improved error handling and resource management"""
        logger.info("Starting AO1 semantic scan of all BigQuery datasets")
        start_time = time.time()
        
        try:
            datasets = self.connection.list_datasets()
            logger.info(f"Found {len(datasets)} datasets to analyze")
            
            if not datasets:
                logger.warning("No datasets found")
                return self._create_empty_results(start_time)
            
            semantic_results = {}
            
            # Reduce workers to prevent resource exhaustion
            actual_workers = min(max_workers, 3, len(datasets))
            
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                future_to_dataset = {}
                
                for dataset in datasets:
                    try:
                        future = executor.submit(
                            self._analyze_dataset_safe, 
                            dataset.dataset_id, 
                            quick_scan
                        )
                        future_to_dataset[future] = dataset.dataset_id
                    except Exception as e:
                        logger.warning(f"Failed to submit task for {dataset.dataset_id}: {e}")
                
                completed = 0
                for future in as_completed(future_to_dataset, timeout=300):  # 5 minute timeout
                    dataset_id = future_to_dataset[future]
                    try:
                        dataset_analysis = future.result(timeout=60)  # 1 minute per dataset
                        if dataset_analysis:
                            semantic_results.update(dataset_analysis)
                        completed += 1
                        logger.info(f"Completed {completed}/{len(future_to_dataset)}: {dataset_id}")
                    except Exception as e:
                        logger.error(f"Failed to analyze dataset {dataset_id}: {e}")
                        completed += 1
            
            logger.info(f"Dataset analysis complete. Found {len(semantic_results)} tables with semantic data")
            
            # Generate queries
            try:
                logger.info("Generating AO1-optimized queries")
                generated_queries = self.query_generator.generate_ao1_queries(semantic_results)
                self.discovered_queries = generated_queries
                logger.info(f"Generated {len(generated_queries)} queries")
            except Exception as e:
                logger.error(f"Query generation failed: {e}")
                generated_queries = []
                self.discovered_queries = []
            
            # Validate queries (limited subset)
            if not quick_scan and generated_queries:
                try:
                    logger.info("Validating sample of generated queries")
                    validation_subset = generated_queries[:10]  # Only validate first 10
                    validation_results = self.data_validator.validate_queries(validation_subset)
                    self.validation_results = validation_results
                except Exception as e:
                    logger.error(f"Query validation failed: {e}")
                    self.validation_results = []
            else:
                self.validation_results = []
            
            # Find relationships
            try:
                relationships = self.semantic_analyzer.find_relationships(semantic_results)
            except Exception as e:
                logger.warning(f"Relationship analysis failed: {e}")
                relationships = []
            
            # Generate summary
            scan_summary = self._generate_scan_summary(semantic_results, generated_queries, self.validation_results)
            
            total_time = time.time() - start_time
            logger.info(f"AO1 scan completed in {total_time:.2f} seconds")
            
            results = {
                'scan_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_datasets': len(datasets),
                    'analyzed_tables': len(semantic_results),
                    'execution_time_seconds': round(total_time, 2),
                    'quick_scan': quick_scan
                },
                'semantic_analysis': self._serialize_semantic_results(semantic_results),
                'generated_queries': [self._serialize_query(q) for q in generated_queries],
                'validation_results': [self._serialize_validation(v) for v in self.validation_results],
                'relationships': relationships,
                'summary': scan_summary
            }
            
            self.scan_results = results
            self._save_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return self._create_error_results(start_time, str(e))
    
    def _analyze_dataset_safe(self, dataset_id: str, quick_scan: bool) -> Optional[Dict[str, List[FieldAnalysis]]]:
        """Safely analyze a single dataset with comprehensive error handling"""
        try:
            logger.info(f"Analyzing dataset: {dataset_id}")
            
            tables = self.connection.list_tables(dataset_id)
            if not tables:
                logger.debug(f"No tables found in dataset {dataset_id}")
                return None
            
            dataset_results = {}
            table_limit = 5 if quick_scan else 15  # Further reduced limits
            table_count = 0
            
            for table in tables:
                if table_count >= table_limit:
                    break
                
                try:
                    # Skip non-table objects
                    if table.table_type != 'TABLE':
                        continue
                    
                    table_key = f"{dataset_id}.{table.table_id}"
                    logger.debug(f"Analyzing table: {table_key}")
                    
                    # Get schema
                    schema = self.connection.get_table_schema(dataset_id, table.table_id)
                    if not schema or len(schema) == 0:
                        logger.debug(f"No schema found for {table_key}")
                        continue
                    
                    # Limit fields to analyze
                    limited_schema = schema[:10]  # Only analyze first 10 fields
                    
                    # Get sample data
                    try:
                        table_data = self.connection.batch_sample_table_data(
                            dataset_id, 
                            table.table_id, 
                            fields=[f.name for f in limited_schema], 
                            limit=10 if quick_scan else 15
                        )
                    except Exception as e:
                        logger.debug(f"Failed to sample data from {table_key}: {e}")
                        continue
                    
                    if not table_data:
                        logger.debug(f"No sample data for {table_key}")
                        continue
                    
                    # Analyze fields
                    try:
                        field_analyses = self.semantic_analyzer.analyze_batch_fields(table_data, limited_schema)
                        
                        if field_analyses:
                            dataset_results[table_key] = field_analyses
                            logger.debug(f"Successfully analyzed {len(field_analyses)} fields in {table_key}")
                        else:
                            logger.debug(f"No semantic fields found in {table_key}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to analyze fields in {table_key}: {e}")
                        continue
                    
                    table_count += 1
                    
                    # Add small delay to prevent overwhelming the system
                    time.sleep(0.1)
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze table {dataset_id}.{table.table_id}: {e}")
                    continue
            
            if dataset_results:
                logger.info(f"Dataset {dataset_id}: Found semantic data in {len(dataset_results)} tables")
                return dataset_results
            else:
                logger.debug(f"Dataset {dataset_id}: No semantic data found")
                return None
            
        except Exception as e:
            logger.error(f"Failed to analyze dataset {dataset_id}: {e}")
            return None
    
    def _create_empty_results(self, start_time: float) -> Dict[str, Any]:
        """Create empty results structure"""
        total_time = time.time() - start_time
        return {
            'scan_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_datasets': 0,
                'analyzed_tables': 0,
                'execution_time_seconds': round(total_time, 2),
                'quick_scan': True
            },
            'semantic_analysis': {},
            'generated_queries': [],
            'validation_results': [],
            'relationships': [],
            'summary': {
                'discovery_metrics': {
                    'total_tables_analyzed': 0,
                    'total_fields_analyzed': 0,
                    'high_confidence_fields': 0,
                    'ao1_relevant_tables': 0
                },
                'category_distribution': {},
                'ao1_table_types': {
                    'asset_identity_tables': 0,
                    'security_tool_tables': 0,
                    'log_source_tables': 0
                },
                'query_performance': {
                    'total_queries_generated': 0,
                    'successful_validations': 0,
                    'average_query_time_ms': 0,
                    'query_success_rate': 0
                },
                'recommendations': ['No datasets found to analyze']
            }
        }
    
    def _create_error_results(self, start_time: float, error_msg: str) -> Dict[str, Any]:
        """Create error results structure"""
        results = self._create_empty_results(start_time)
        results['summary']['recommendations'] = [f'Scan failed: {error_msg}']
        return results
    
    def _serialize_semantic_results(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> Dict:
        """Serialize semantic results safely"""
        serialized = {}
        for table_name, analyses in semantic_results.items():
            serialized[table_name] = []
            for analysis in analyses:
                try:
                    serialized[table_name].append({
                        'field_name': analysis.field_name,
                        'field_type': analysis.field_type,
                        'ao1_category': analysis.ao1_category,
                        'confidence_score': analysis.confidence_score,
                        'semantic_evidence': analysis.semantic_evidence,
                        'sample_values': analysis.sample_values[:3],  # Limit sample values
                        'value_patterns': analysis.value_patterns,
                        'reasoning_explanation': analysis.reasoning_explanation,
                        'alternative_classifications': analysis.alternative_classifications or []
                    })
                except Exception as e:
                    logger.warning(f"Failed to serialize analysis for {analysis.field_name}: {e}")
        return serialized
    
    def _generate_scan_summary(self, semantic_results: Dict, queries: List[GeneratedQuery], 
                              validations: List[ValidationResult]) -> Dict[str, Any]:
        """Generate scan summary with error handling"""
        try:
            total_tables = len(semantic_results)
            total_fields = sum(len(fields) for fields in semantic_results.values())
            
            category_counts = {}
            high_confidence_fields = 0
            
            for table_fields in semantic_results.values():
                for field in table_fields:
                    category = field.ao1_category
                    category_counts[category] = category_counts.get(category, 0) + 1
                    
                    if field.confidence_score >= 0.7:
                        high_confidence_fields += 1
            
            successful_queries = sum(1 for v in validations if v.success)
            avg_query_time = sum(v.execution_time_ms for v in validations) / len(validations) if validations else 0
            
            asset_tables = len([t for t, fields in semantic_results.items() 
                               if any(f.ao1_category == 'asset_identity' for f in fields)])
            
            tool_tables = len([t for t, fields in semantic_results.items() 
                              if any(f.ao1_category == 'security_tools' for f in fields)])
            
            log_tables = len([t for t, fields in semantic_results.items() 
                             if any(f.ao1_category == 'log_sources' for f in fields)])
            
            return {
                'discovery_metrics': {
                    'total_tables_analyzed': total_tables,
                    'total_fields_analyzed': total_fields,
                    'high_confidence_fields': high_confidence_fields,
                    'ao1_relevant_tables': len([t for t in semantic_results.keys() if semantic_results[t]])
                },
                'category_distribution': category_counts,
                'ao1_table_types': {
                    'asset_identity_tables': asset_tables,
                    'security_tool_tables': tool_tables,
                    'log_source_tables': log_tables
                },
                'query_performance': {
                    'total_queries_generated': len(queries),
                    'successful_validations': successful_queries,
                    'average_query_time_ms': round(avg_query_time, 2),
                    'query_success_rate': round(successful_queries / len(queries) * 100, 2) if queries else 0
                },
                'recommendations': self._generate_recommendations(semantic_results, validations)
            }
        except Exception as e:
            logger.error(f"Failed to generate scan summary: {e}")
            return {
                'discovery_metrics': {'total_tables_analyzed': 0, 'total_fields_analyzed': 0, 'high_confidence_fields': 0, 'ao1_relevant_tables': 0},
                'category_distribution': {},
                'ao1_table_types': {'asset_identity_tables': 0, 'security_tool_tables': 0, 'log_source_tables': 0},
                'query_performance': {'total_queries_generated': 0, 'successful_validations': 0, 'average_query_time_ms': 0, 'query_success_rate': 0},
                'recommendations': ['Summary generation failed']
            }
    
    def _generate_recommendations(self, semantic_results: Dict, validations: List[ValidationResult]) -> List[str]:
        """Generate recommendations safely"""
        try:
            recommendations = []
            
            asset_tables = [t for t, fields in semantic_results.items() 
                           if any(f.ao1_category == 'asset_identity' and f.confidence_score > 0.7 for f in fields)]
            
            if len(asset_tables) > 1:
                recommendations.append(f"Found {len(asset_tables)} high-quality asset tables - consider using the largest as baseline")
            elif len(asset_tables) == 0:
                recommendations.append("No high-confidence asset identity tables found - manual review needed")
            else:
                recommendations.append(f"Single asset table found: {asset_tables[0]} - good baseline candidate")
            
            # Tool coverage analysis
            tool_indicators = {}
            for table, fields in semantic_results.items():
                for field in fields:
                    if field.ao1_category == 'security_tools' and field.confidence_score > 0.6:
                        field_lower = field.field_name.lower()
                        if 'crowdstrike' in field_lower or 'falcon' in field_lower:
                            tool_indicators['CrowdStrike'] = tool_indicators.get('CrowdStrike', 0) + 1
                        elif 'chronicle' in field_lower:
                            tool_indicators['Chronicle'] = tool_indicators.get('Chronicle', 0) + 1
                        elif 'splunk' in field_lower:
                            tool_indicators['Splunk'] = tool_indicators.get('Splunk', 0) + 1
            
            if tool_indicators:
                tools_found = ', '.join([f"{tool} ({count} refs)" for tool, count in tool_indicators.items()])
                recommendations.append(f"Security tools detected: {tools_found}")
            else:
                recommendations.append("No clear security tool references found")
            
            # Data quality assessment
            if validations:
                failed_queries = sum(1 for v in validations if not v.success)
                if failed_queries > len(validations) * 0.5:
                    recommendations.append("High query failure rate detected - check data access permissions")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
            return ["Recommendation generation failed - manual analysis required"]
    
    def _serialize_query(self, query: GeneratedQuery) -> Dict:
        """Serialize query safely"""
        try:
            return {
                'purpose': query.purpose,
                'query': query.query,
                'source_tables': query.source_tables,
                'key_fields': query.key_fields,
                'expected_result_type': query.expected_result_type,
                'confidence_score': query.confidence_score
            }
        except Exception as e:
            logger.warning(f"Failed to serialize query: {e}")
            return {'purpose': 'Serialization failed', 'query': '', 'source_tables': [], 'key_fields': [], 'expected_result_type': 'unknown', 'confidence_score': 0.0}
    
    def _serialize_validation(self, validation: ValidationResult) -> Dict:
        """Serialize validation safely"""
        try:
            return {
                'query_id': validation.query_id,
                'success': validation.success,
                'execution_time_ms': validation.execution_time_ms,
                'result_count': validation.result_count,
                'sample_results': validation.sample_results[:3],  # Limit sample results
                'data_quality_score': validation.data_quality_score,
                'validation_errors': validation.validation_errors,
                'performance_rating': validation.performance_rating
            }
        except Exception as e:
            logger.warning(f"Failed to serialize validation: {e}")
            return {'query_id': 'unknown', 'success': False, 'execution_time_ms': 0, 'result_count': 0, 'sample_results': [], 'data_quality_score': 0.0, 'validation_errors': ['Serialization failed'], 'performance_rating': 'FAILED'}
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results with error handling"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON results
            results_file = self.output_dir / f"ao1_scan_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Scan results saved to: {results_file}")
            
            # Save SQL queries
            if self.discovered_queries:
                queries_file = self.output_dir / f"ao1_queries_{timestamp}.sql"
                with open(queries_file, 'w') as f:
                    f.write("-- AO1 Generated Queries\n")
                    f.write(f"-- Generated on: {datetime.now().isoformat()}\n\n")
                    
                    for i, query in enumerate(self.discovered_queries):
                        f.write(f"-- Query {i+1}: {query.purpose}\n")
                        f.write(f"-- Confidence: {query.confidence_score:.2f}\n")
                        f.write(f"-- Tables: {', '.join(query.source_tables)}\n")
                        f.write(f"{query.query};\n\n")
                logger.info(f"Generated queries saved to: {queries_file}")
            
            # Save summary
            summary_file = self.output_dir / f"ao1_summary_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                summary = results['summary']
                f.write("AO1 BigQuery Semantic Scan Summary\n")
                f.write("=" * 40 + "\n\n")
                
                f.write("Discovery Metrics:\n")
                for key, value in summary['discovery_metrics'].items():
                    f.write(f"  {key}: {value}\n")
                
                f.write("\nAO1 Category Distribution:\n")
                for category, count in summary['category_distribution'].items():
                    f.write(f"  {category}: {count} fields\n")
                
                f.write("\nRecommendations:\n")
                for rec in summary['recommendations']:
                    f.write(f"  - {rec}\n")
            logger.info(f"Summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def get_best_queries(self, min_confidence: float = 0.7) -> List[GeneratedQuery]:
        """Get best queries safely"""
        try:
            return [q for q in self.discovered_queries if q.confidence_score >= min_confidence]
        except Exception as e:
            logger.warning(f"Failed to get best queries: {e}")
            return []
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary safely"""
        try:
            if not self.validation_results:
                return {}
            
            successful = sum(1 for v in self.validation_results if v.success)
            total = len(self.validation_results)
            
            avg_time = sum(v.execution_time_ms for v in self.validation_results) / total
            avg_quality = sum(v.data_quality_score for v in self.validation_results if v.success) / max(successful, 1)
            
            return {
                'success_rate': round(successful / total * 100, 2),
                'average_execution_time_ms': round(avg_time, 2),
                'average_data_quality_score': round(avg_quality, 2),
                'performance_distribution': {
                    rating: sum(1 for v in self.validation_results if v.performance_rating == rating)
                    for rating in ['EXCELLENT', 'GOOD', 'ACCEPTABLE', 'SLOW', 'VERY_SLOW', 'FAILED']
                }
            }
        except Exception as e:
            logger.warning(f"Failed to get validation summary: {e}")
            return {}