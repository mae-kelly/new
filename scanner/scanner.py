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
        
        self.connection = BigQueryConnection(service_account_path)
        self.semantic_analyzer = AdvancedSemanticAnalyzer()
        self.query_generator = QueryGenerator()
        self.data_validator = DataValidator(self.connection)
        
        self.scan_results = {}
        self.discovered_queries = []
        self.validation_results = []
        
    def scan_all_datasets(self, max_workers: int = 6, quick_scan: bool = False) -> Dict[str, Any]:
        logger.info("Starting AO1 semantic scan of all BigQuery datasets")
        start_time = time.time()
        
        datasets = self.connection.list_datasets()
        logger.info(f"Found {len(datasets)} datasets to analyze")
        
        semantic_results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_dataset = {
                executor.submit(self._analyze_dataset_optimized, dataset.dataset_id, quick_scan): dataset.dataset_id
                for dataset in datasets
            }
            
            completed = 0
            for future in as_completed(future_to_dataset):
                dataset_id = future_to_dataset[future]
                try:
                    dataset_analysis = future.result()
                    if dataset_analysis:
                        semantic_results.update(dataset_analysis)
                        completed += 1
                        logger.info(f"Completed {completed}/{len(datasets)}: {dataset_id}")
                except Exception as e:
                    logger.error(f"Failed to analyze dataset {dataset_id}: {e}")
        
        logger.info("Generating AO1-optimized queries")
        generated_queries = self.query_generator.generate_ao1_queries(semantic_results)
        self.discovered_queries = generated_queries
        
        logger.info(f"Generated {len(generated_queries)} queries for validation")
        
        if not quick_scan and generated_queries:
            logger.info("Validating generated queries")
            validation_results = self.data_validator.validate_queries(generated_queries[:20])
            self.validation_results = validation_results
        else:
            self.validation_results = []
        
        relationships = self.semantic_analyzer.find_relationships(semantic_results)
        
        scan_summary = self._generate_scan_summary(semantic_results, generated_queries, self.validation_results)
        
        total_time = time.time() - start_time
        logger.info(f"AO1 scan completed in {total_time:.2f} seconds")
        
        results = {
            'scan_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_datasets': len(datasets),
                'analyzed_tables': sum(len(tables) for tables in semantic_results.values()),
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
    
    def _analyze_dataset_optimized(self, dataset_id: str, quick_scan: bool) -> Optional[Dict[str, List[FieldAnalysis]]]:
        try:
            tables = self.connection.list_tables(dataset_id)
            if not tables:
                return None
            
            dataset_results = {}
            table_limit = 10 if quick_scan else 50
            
            for i, table in enumerate(tables):
                if i >= table_limit:
                    break
                
                try:
                    if table.table_type != 'TABLE':
                        continue
                    
                    table_key = f"{dataset_id}.{table.table_id}"
                    
                    schema = self.connection.get_table_schema(dataset_id, table.table_id)
                    if not schema:
                        continue
                    
                    table_data = self.connection.batch_sample_table_data(
                        dataset_id, table.table_id, 
                        fields=[f.name for f in schema[:20]], 
                        limit=15 if quick_scan else 25
                    )
                    
                    if not table_data:
                        continue
                    
                    field_analyses = self.semantic_analyzer.analyze_batch_fields(table_data, schema)
                    
                    if field_analyses:
                        dataset_results[table_key] = field_analyses
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze table {dataset_id}.{table.table_id}: {e}")
                    continue
            
            return dataset_results if dataset_results else None
            
        except Exception as e:
            logger.error(f"Failed to analyze dataset {dataset_id}: {e}")
            return None
    
    def _serialize_semantic_results(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> Dict:
        serialized = {}
        for table_name, analyses in semantic_results.items():
            serialized[table_name] = []
            for analysis in analyses:
                serialized[table_name].append({
                    'field_name': analysis.field_name,
                    'field_type': analysis.field_type,
                    'ao1_category': analysis.ao1_category,
                    'confidence_score': analysis.confidence_score,
                    'semantic_evidence': analysis.semantic_evidence,
                    'sample_values': analysis.sample_values,
                    'value_patterns': analysis.value_patterns,
                    'reasoning_explanation': analysis.reasoning_explanation,
                    'alternative_classifications': analysis.alternative_classifications or []
                })
        return serialized
    
    def _generate_scan_summary(self, semantic_results: Dict, queries: List[GeneratedQuery], 
                              validations: List[ValidationResult]) -> Dict[str, Any]:
        
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
    
    def _generate_recommendations(self, semantic_results: Dict, validations: List[ValidationResult]) -> List[str]:
        recommendations = []
        
        asset_tables = [t for t, fields in semantic_results.items() 
                       if any(f.ao1_category == 'asset_identity' and f.confidence_score > 0.7 for f in fields)]
        
        if len(asset_tables) > 1:
            recommendations.append(f"Found {len(asset_tables)} high-quality asset tables - consider using the largest as your baseline")
        elif len(asset_tables) == 0:
            recommendations.append("No high-confidence asset identity tables found - manual review required")
        
        tool_coverage = {}
        for table, fields in semantic_results.items():
            for field in fields:
                if field.ao1_category == 'security_tools' and field.confidence_score > 0.6:
                    if 'crowdstrike' in field.field_name.lower():
                        tool_coverage['crowdstrike'] = tool_coverage.get('crowdstrike', 0) + 1
                    if 'chronicle' in field.field_name.lower():
                        tool_coverage['chronicle'] = tool_coverage.get('chronicle', 0) + 1
                    if 'splunk' in field.field_name.lower():
                        tool_coverage['splunk'] = tool_coverage.get('splunk', 0) + 1
        
        for tool, count in tool_coverage.items():
            if count >= 2:
                recommendations.append(f"Multiple {tool.title()} data sources found ({count} tables) - consolidation opportunity")
        
        failed_queries = [v for v in validations if not v.success]
        if len(failed_queries) > len(validations) * 0.3:
            recommendations.append("High query failure rate - data quality issues detected")
        
        geographic_tables = [t for t, fields in semantic_results.items() 
                           if any(f.ao1_category == 'geographic_data' for f in fields)]
        
        if not geographic_tables:
            recommendations.append("No geographic data found - regional visibility analysis may be limited")
        
        return recommendations
    
    def _serialize_query(self, query: GeneratedQuery) -> Dict:
        return {
            'purpose': query.purpose,
            'query': query.query,
            'source_tables': query.source_tables,
            'key_fields': query.key_fields,
            'expected_result_type': query.expected_result_type,
            'confidence_score': query.confidence_score
        }
    
    def _serialize_validation(self, validation: ValidationResult) -> Dict:
        return {
            'query_id': validation.query_id,
            'success': validation.success,
            'execution_time_ms': validation.execution_time_ms,
            'result_count': validation.result_count,
            'sample_results': validation.sample_results,
            'data_quality_score': validation.data_quality_score,
            'validation_errors': validation.validation_errors,
            'performance_rating': validation.performance_rating
        }
    
    def _save_results(self, results: Dict[str, Any]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_file = self.output_dir / f"ao1_scan_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Scan results saved to: {results_file}")
        
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
    
    def get_best_queries(self, min_confidence: float = 0.7) -> List[GeneratedQuery]:
        return [q for q in self.discovered_queries if q.confidence_score >= min_confidence]
    
    def get_validation_summary(self) -> Dict[str, Any]:
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