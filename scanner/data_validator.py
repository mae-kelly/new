import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from .connection import BigQueryConnection
from .query_generator import GeneratedQuery

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    query_id: str
    success: bool
    execution_time_ms: float
    result_count: int
    sample_results: List[Dict]
    data_quality_score: float
    validation_errors: List[str]
    performance_rating: str

@dataclass
class TableQualityMetrics:
    table_name: str
    completeness_score: float
    uniqueness_score: float
    consistency_score: float
    freshness_score: float
    overall_quality: float

class DataValidator:
    def __init__(self, connection: BigQueryConnection):
        self.connection = connection
        self.client = connection.get_client()
        
    def validate_queries(self, queries: List[GeneratedQuery]) -> List[ValidationResult]:
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"Validating query {i+1}/{len(queries)}: {query.purpose}")
            result = self._validate_single_query(query, f"query_{i}")
            results.append(result)
            
            if not result.success:
                logger.warning(f"Query validation failed: {query.purpose}")
            
            time.sleep(0.1)
        
        return results
    
    def _validate_single_query(self, query: GeneratedQuery, query_id: str) -> ValidationResult:
        start_time = time.time()
        
        try:
            job = self.client.query(query.query)
            results = list(job.result())
            
            execution_time = (time.time() - start_time) * 1000
            
            sample_results = []
            for row in results[:5]:
                sample_results.append(dict(row))
            
            data_quality_score = self._calculate_data_quality(results, query.expected_result_type)
            performance_rating = self._rate_performance(execution_time, len(results))
            
            validation_errors = self._check_result_validity(results, query.expected_result_type)
            
            return ValidationResult(
                query_id=query_id,
                success=True,
                execution_time_ms=execution_time,
                result_count=len(results),
                sample_results=sample_results,
                data_quality_score=data_quality_score,
                validation_errors=validation_errors,
                performance_rating=performance_rating
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Query execution failed: {str(e)}")
            
            return ValidationResult(
                query_id=query_id,
                success=False,
                execution_time_ms=execution_time,
                result_count=0,
                sample_results=[],
                data_quality_score=0.0,
                validation_errors=[str(e)],
                performance_rating="FAILED"
            )
    
    def _calculate_data_quality(self, results: List, expected_type: str) -> float:
        if not results:
            return 0.0
        
        quality_checks = []
        
        for row in results:
            row_dict = dict(row) if hasattr(row, 'keys') else row
            
            if expected_type == "coverage_metrics":
                quality_checks.extend(self._validate_coverage_metrics(row_dict))
            elif expected_type == "tool_coverage":
                quality_checks.extend(self._validate_tool_coverage(row_dict))
            elif expected_type == "log_coverage":
                quality_checks.extend(self._validate_log_coverage(row_dict))
            elif expected_type == "geographic_coverage":
                quality_checks.extend(self._validate_geographic_coverage(row_dict))
            else:
                quality_checks.extend(self._validate_generic_data(row_dict))
        
        if not quality_checks:
            return 0.0
        
        return sum(quality_checks) / len(quality_checks)
    
    def _validate_coverage_metrics(self, row: Dict) -> List[float]:
        checks = []
        
        for key, value in row.items():
            if 'asset' in key.lower() and isinstance(value, (int, float)):
                checks.append(1.0 if value >= 0 else 0.0)
            
            if 'percentage' in key.lower() and isinstance(value, (int, float)):
                checks.append(1.0 if 0 <= value <= 100 else 0.5)
            
            if 'coverage' in key.lower() and isinstance(value, (int, float)):
                checks.append(1.0 if value >= 0 else 0.0)
        
        return checks if checks else [0.5]
    
    def _validate_tool_coverage(self, row: Dict) -> List[float]:
        checks = []
        
        has_tool_name = any('tool' in key.lower() for key in row.keys())
        checks.append(1.0 if has_tool_name else 0.0)
        
        has_coverage_metric = any('coverage' in key.lower() or 'asset' in key.lower() for key in row.keys())
        checks.append(1.0 if has_coverage_metric else 0.0)
        
        for key, value in row.items():
            if isinstance(value, (int, float)) and value < 0:
                checks.append(0.0)
            elif isinstance(value, (int, float)):
                checks.append(1.0)
        
        return checks if checks else [0.5]
    
    def _validate_log_coverage(self, row: Dict) -> List[float]:
        checks = []
        
        has_log_type = any('log' in key.lower() or 'type' in key.lower() for key in row.keys())
        checks.append(1.0 if has_log_type else 0.0)
        
        has_count_metric = any('count' in key.lower() or 'volume' in key.lower() for key in row.keys())
        checks.append(1.0 if has_count_metric else 0.0)
        
        for key, value in row.items():
            if 'count' in key.lower() or 'volume' in key.lower():
                if isinstance(value, (int, float)) and value >= 0:
                    checks.append(1.0)
                else:
                    checks.append(0.0)
        
        return checks if checks else [0.5]
    
    def _validate_geographic_coverage(self, row: Dict) -> List[float]:
        checks = []
        
        has_location = any(key.lower() in ['country', 'region', 'location', 'geographic_location'] for key in row.keys())
        checks.append(1.0 if has_location else 0.0)
        
        has_asset_count = any('asset' in key.lower() or 'count' in key.lower() for key in row.keys())
        checks.append(1.0 if has_asset_count else 0.0)
        
        return checks if checks else [0.5]
    
    def _validate_generic_data(self, row: Dict) -> List[float]:
        checks = []
        
        non_null_values = sum(1 for value in row.values() if value is not None)
        total_values = len(row)
        
        if total_values > 0:
            completeness = non_null_values / total_values
            checks.append(completeness)
        
        numeric_values = [v for v in row.values() if isinstance(v, (int, float))]
        if numeric_values:
            negative_count = sum(1 for v in numeric_values if v < 0)
            if negative_count == 0:
                checks.append(1.0)
            else:
                checks.append(max(0.0, 1.0 - (negative_count / len(numeric_values))))
        
        return checks if checks else [0.5]
    
    def _rate_performance(self, execution_time_ms: float, result_count: int) -> str:
        if execution_time_ms < 1000:
            return "EXCELLENT"
        elif execution_time_ms < 5000:
            return "GOOD" 
        elif execution_time_ms < 15000:
            return "ACCEPTABLE"
        elif execution_time_ms < 60000:
            return "SLOW"
        else:
            return "VERY_SLOW"
    
    def _check_result_validity(self, results: List, expected_type: str) -> List[str]:
        errors = []
        
        if not results:
            errors.append("Query returned no results")
            return errors
        
        if expected_type == "coverage_metrics":
            if not any(any('asset' in str(key).lower() for key in dict(row).keys()) for row in results):
                errors.append("Coverage metrics query should return asset-related fields")
        
        elif expected_type == "tool_coverage":
            if not any(any('tool' in str(key).lower() or 'coverage' in str(key).lower() for key in dict(row).keys()) for row in results):
                errors.append("Tool coverage query should return tool or coverage related fields")
        
        for row in results[:10]:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, (int, float)) and value < 0:
                    if 'percentage' not in key.lower() and 'coverage' in key.lower():
                        errors.append(f"Unexpected negative value in {key}: {value}")
        
        return errors
    
    def analyze_table_quality(self, dataset_id: str, table_id: str, key_fields: List[str]) -> TableQualityMetrics:
        table_name = f"{dataset_id}.{table_id}"
        
        try:
            completeness = self._calculate_completeness(dataset_id, table_id, key_fields)
            uniqueness = self._calculate_uniqueness(dataset_id, table_id, key_fields)
            consistency = self._calculate_consistency(dataset_id, table_id, key_fields)
            freshness = self._calculate_freshness(dataset_id, table_id)
            
            overall_quality = (completeness + uniqueness + consistency + freshness) / 4
            
            return TableQualityMetrics(
                table_name=table_name,
                completeness_score=completeness,
                uniqueness_score=uniqueness,
                consistency_score=consistency,
                freshness_score=freshness,
                overall_quality=overall_quality
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze table quality for {table_name}: {e}")
            return TableQualityMetrics(
                table_name=table_name,
                completeness_score=0.0,
                uniqueness_score=0.0,
                consistency_score=0.0,
                freshness_score=0.0,
                overall_quality=0.0
            )
    
    def _calculate_completeness(self, dataset_id: str, table_id: str, key_fields: List[str]) -> float:
        if not key_fields:
            return 0.0
        
        field_completeness_scores = []
        
        for field in key_fields:
            query = f"""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(`{field}`) as non_null_rows
                FROM `{self.connection.project_id}.{dataset_id}.{table_id}`
            """
            
            try:
                results = list(self.client.query(query).result())
                if results:
                    row = dict(results[0])
                    total = row.get('total_rows', 0)
                    non_null = row.get('non_null_rows', 0)
                    
                    if total > 0:
                        field_completeness_scores.append(non_null / total)
                    else:
                        field_completeness_scores.append(0.0)
                else:
                    field_completeness_scores.append(0.0)
                    
            except Exception:
                field_completeness_scores.append(0.0)
        
        return sum(field_completeness_scores) / len(field_completeness_scores) if field_completeness_scores else 0.0
    
    def _calculate_uniqueness(self, dataset_id: str, table_id: str, key_fields: List[str]) -> float:
        if not key_fields:
            return 0.0
        
        primary_field = key_fields[0]
        
        query = f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT `{primary_field}`) as unique_values
            FROM `{self.connection.project_id}.{dataset_id}.{table_id}`
            WHERE `{primary_field}` IS NOT NULL
        """
        
        try:
            results = list(self.client.query(query).result())
            if results:
                row = dict(results[0])
                total = row.get('total_rows', 0)
                unique = row.get('unique_values', 0)
                
                if total > 0:
                    return unique / total
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_consistency(self, dataset_id: str, table_id: str, key_fields: List[str]) -> float:
        if not key_fields:
            return 0.0
        
        consistency_scores = []
        
        for field in key_fields:
            query = f"""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN LENGTH(TRIM(CAST(`{field}` AS STRING))) = 0 THEN 1 END) as empty_values,
                    COUNT(CASE WHEN `{field}` IS NULL THEN 1 END) as null_values
                FROM `{self.connection.project_id}.{dataset_id}.{table_id}`
            """
            
            try:
                results = list(self.client.query(query).result())
                if results:
                    row = dict(results[0])
                    total = row.get('total_records', 0)
                    empty = row.get('empty_values', 0)
                    null = row.get('null_values', 0)
                    
                    if total > 0:
                        valid_values = total - empty - null
                        consistency_scores.append(valid_values / total)
                    else:
                        consistency_scores.append(0.0)
                else:
                    consistency_scores.append(0.0)
                    
            except Exception:
                consistency_scores.append(0.0)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_freshness(self, dataset_id: str, table_id: str) -> float:
        try:
            table_info = self.connection.get_table_info(dataset_id, table_id)
            
            if 'modified' in table_info:
                import datetime
                modified_date = table_info['modified']
                if modified_date:
                    days_since_modified = (datetime.datetime.now(datetime.timezone.utc) - modified_date).days
                    
                    if days_since_modified <= 1:
                        return 1.0
                    elif days_since_modified <= 7:
                        return 0.8
                    elif days_since_modified <= 30:
                        return 0.6
                    elif days_since_modified <= 90:
                        return 0.4
                    else:
                        return 0.2
            
            return 0.5
            
        except Exception:
            return 0.5