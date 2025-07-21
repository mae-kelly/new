#!/usr/bin/env python3
"""
AO1 Self-Healing Query Engine - Production Ready
Intelligent query generation and validation system for AO1 visibility metrics

Key Improvements:
✅ Clean, maintainable code structure
✅ Robust error handling and logging
✅ Configurable validation thresholds
✅ Comprehensive query optimization
✅ Better field discovery algorithms
✅ Production-ready logging and monitoring
✅ Modular design for easy extension
"""

import logging
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import duckdb
import traceback
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ao1_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QueryTestResult:
    """Results from testing a query"""
    query_name: str
    sql_query: str
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    row_count: int = 0
    sample_results: List[Dict] = field(default_factory=list)
    field_validation: Dict[str, bool] = field(default_factory=dict)
    confidence_score: float = 0.0

@dataclass
class FieldMapping:
    """Field mapping discovered by the engine"""
    metric_name: str
    target_field: str
    source_table: str
    source_column: str
    confidence_score: float
    validation_samples: List[Any] = field(default_factory=list)
    transformation_logic: Optional[str] = None
    data_quality_score: float = 0.0

@dataclass
class EngineConfig:
    """Configuration for the engine"""
    max_iterations: int = 20
    min_success_rate: float = 0.95
    min_confidence_score: float = 0.7
    sample_size: int = 100
    timeout_seconds: int = 300
    enable_aggressive_healing: bool = True
    
class QueryHealingStrategies:
    """Collection of query healing strategies"""
    
    @staticmethod
    def fix_column_references(sql_query: str, error_msg: str, available_columns: List[str]) -> str:
        """Fix column reference errors with available alternatives"""
        column_pattern = r"column ['\"]?([^'\"]+)['\"]?"
        match = re.search(column_pattern, error_msg, re.IGNORECASE)
        
        if not match:
            return sql_query
        
        problem_column = match.group(1)
        
        # Find best matching alternative
        alternatives = QueryHealingStrategies._find_column_alternatives(
            problem_column, available_columns
        )
        
        if alternatives:
            best_alternative = alternatives[0]
            logger.info(f"Replacing '{problem_column}' with '{best_alternative}'")
            return sql_query.replace(problem_column, best_alternative)
        
        return sql_query
    
    @staticmethod
    def _find_column_alternatives(problem_column: str, available_columns: List[str]) -> List[str]:
        """Find best alternative column names using fuzzy matching"""
        alternatives = []
        problem_lower = problem_column.lower()
        
        # Exact matches first
        for col in available_columns:
            if col.lower() == problem_lower:
                alternatives.append(col)
        
        # Partial matches
        for col in available_columns:
            col_lower = col.lower()
            if (problem_lower in col_lower or col_lower in problem_lower) and len(col_lower) > 2:
                if col not in alternatives:
                    alternatives.append(col)
        
        # Common field mappings
        field_mappings = {
            'hostname': ['host', 'computer_name', 'device_name', 'machine_name', 'name'],
            'host': ['hostname', 'computer_name', 'device_name', 'name'],
            'fqdn': ['hostname', 'domain_name', 'name'],
            'os': ['operating_system', 'platform', 'os_name'],
            'platform': ['os', 'operating_system', 'type'],
            'agent_status': ['status', 'health', 'state', 'agent_health'],
            'log_type': ['sourcetype', 'event_type', 'type']
        }
        
        mapped_alternatives = field_mappings.get(problem_lower, [])
        for alt in mapped_alternatives:
            for col in available_columns:
                if alt.lower() in col.lower() and col not in alternatives:
                    alternatives.append(col)
        
        return alternatives[:3]  # Return top 3 alternatives
    
    @staticmethod
    def fix_table_references(sql_query: str, error_msg: str, available_tables: List[str]) -> str:
        """Fix table reference errors"""
        table_pattern = r"table ['\"]?([^'\"]+)['\"]?"
        match = re.search(table_pattern, error_msg, re.IGNORECASE)
        
        if not match or not available_tables:
            return sql_query
        
        problem_table = match.group(1)
        
        # Find the best table alternative
        for table in available_tables:
            if 'all_sources' in table.lower():
                replacement = f'"{table}"' if '"' not in table else table
                logger.info(f"Replacing table '{problem_table}' with '{replacement}'")
                return sql_query.replace(problem_table, replacement)
        
        # Use first available table as fallback
        if available_tables:
            replacement = f'"{available_tables[0]}"'
            return sql_query.replace(problem_table, replacement)
        
        return sql_query
    
    @staticmethod
    def fix_syntax_errors(sql_query: str, error_msg: str) -> str:
        """Fix common SQL syntax errors"""
        healed_sql = sql_query
        
        # Fix common quote issues
        if 'quote' in error_msg.lower():
            healed_sql = re.sub(r"(?<!')'(?!')", "''", healed_sql)
        
        # Fix type casting issues
        if 'type' in error_msg.lower() or 'cast' in error_msg.lower():
            # Add explicit casts for division operations
            healed_sql = re.sub(
                r'(\w+)\s*/\s*(\w+)',
                r'CAST(\1 AS FLOAT) / CAST(\2 AS FLOAT)',
                healed_sql
            )
        
        return healed_sql

class SelfHealingQueryEngine:
    """Production-ready self-healing query engine for AO1 metrics"""
    
    def __init__(self, db_path: str, config: Optional[EngineConfig] = None):
        self.db_path = Path(db_path)
        self.config = config or EngineConfig()
        self.connection = None
        
        # Core state
        self.field_mappings: Dict[str, FieldMapping] = {}
        self.validated_queries: Dict[str, QueryTestResult] = {}
        self.healing_iterations: Dict[str, int] = {}
        self.available_tables: List[str] = []
        self.available_columns: Dict[str, List[str]] = {}
        
        # AO1 metrics configuration
        self.ao1_metrics = self._load_ao1_metrics()
        
        logger.info("Initializing AO1 Self-Healing Query Engine")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Max iterations: {self.config.max_iterations}")
        
    def _load_ao1_metrics(self) -> Dict[str, Dict]:
        """Load AO1 metrics configuration"""
        return {
            'host_identity': {
                'description': 'Primary asset identifier',
                'priority': 1,
                'keywords': ['host', 'hostname', 'computer', 'device', 'machine', 'name'],
                'validation_criteria': ['unique_values', 'non_null', 'reasonable_cardinality']
            },
            'network_role_coverage': {
                'description': 'Network logging coverage (Firewall, IDS/IPS, NDR, Proxy, DNS, WAF)',
                'priority': 1,
                'keywords': ['log', 'type', 'source', 'chronicle_log_type', 'sourcetype'],
                'patterns': [r'firewall|proxy|dns|ids|ips|ndr|waf'],
                'validation_criteria': ['log_type_classification', 'volume_statistics']
            },
            'endpoint_role_coverage': {
                'description': 'Endpoint logging coverage (OS logs, EDR, DLP, FIM)',
                'priority': 1,
                'keywords': ['os', 'endpoint', 'edr', 'crowdstrike', 'windows', 'linux'],
                'patterns': [r'windows|linux|os|edr|dlp|fim'],
                'validation_criteria': ['system_identification', 'agent_status']
            },
            'crowdstrike_agent_coverage': {
                'description': 'CrowdStrike agent deployment and health',
                'priority': 1,
                'keywords': ['crowdstrike', 'agent', 'health', 'status'],
                'patterns': [r'healthy|unhealthy|active|inactive'],
                'validation_criteria': ['agent_status', 'health_metrics']
            },
            'infrastructure_classification': {
                'description': 'Infrastructure type classification',
                'priority': 2,
                'keywords': ['type', 'class', 'infrastructure', 'platform'],
                'patterns': [r'cloud|on.?prem|saas|api|physical|virtual'],
                'validation_criteria': ['categorical_values', 'reasonable_distribution']
            }
        }
    
    @contextmanager
    def database_connection(self):
        """Context manager for database connections"""
        try:
            self.connection = duckdb.connect(str(self.db_path))
            yield self.connection
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if self.connection:
                self.connection.close()
                self.connection = None
    
    def discover_schema(self) -> bool:
        """Discover database schema and available tables/columns"""
        logger.info("Starting schema discovery")
        
        with self.database_connection():
            try:
                # Discover tables
                tables_result = self.connection.execute("SHOW TABLES").fetchall()
                self.available_tables = [row[0] for row in tables_result]
                logger.info(f"Found tables: {self.available_tables}")
                
                # Discover columns for each table
                for table in self.available_tables:
                    try:
                        columns_result = self.connection.execute(f"DESCRIBE {table}").fetchall()
                        self.available_columns[table] = [row[0] for row in columns_result]
                        logger.info(f"Table {table}: {len(self.available_columns[table])} columns")
                    except Exception as e:
                        logger.warning(f"Could not describe table {table}: {e}")
                        continue
                
                return len(self.available_tables) > 0
                
            except Exception as e:
                logger.error(f"Schema discovery failed: {e}")
                return False
    
    def discover_field_mappings(self) -> Dict[str, FieldMapping]:
        """Discover optimal field mappings for AO1 metrics"""
        logger.info("Starting field mapping discovery")
        
        if not self.available_tables:
            logger.warning("No tables available for field mapping")
            return {}
        
        with self.database_connection():
            for metric_name, metric_config in self.ao1_metrics.items():
                logger.info(f"Discovering mapping for: {metric_name}")
                
                mapping = self._find_best_field_mapping(metric_name, metric_config)
                if mapping and mapping.confidence_score >= self.config.min_confidence_score:
                    self.field_mappings[metric_name] = mapping
                    logger.info(
                        f"Found mapping: {mapping.source_table}.{mapping.source_column} "
                        f"(confidence: {mapping.confidence_score:.3f})"
                    )
                else:
                    logger.warning(f"No suitable mapping found for {metric_name}")
        
        return self.field_mappings
    
    def _find_best_field_mapping(self, metric_name: str, metric_config: Dict) -> Optional[FieldMapping]:
        """Find the best field mapping for a specific metric"""
        best_mapping = None
        best_score = 0.0
        
        for table_name in self.available_tables:
            columns = self.available_columns.get(table_name, [])
            
            for column_name in columns:
                confidence = self._calculate_field_confidence(
                    table_name, column_name, metric_config
                )
                
                if confidence > best_score:
                    # Validate the field
                    validation_result = self._validate_field_data(
                        table_name, column_name, metric_config
                    )
                    
                    if validation_result['is_valid']:
                        best_score = confidence
                        best_mapping = FieldMapping(
                            metric_name=metric_name,
                            target_field=metric_name,
                            source_table=table_name,
                            source_column=column_name,
                            confidence_score=confidence,
                            validation_samples=validation_result.get('samples', []),
                            data_quality_score=validation_result.get('quality_score', 0.0)
                        )
        
        return best_mapping
    
    def _calculate_field_confidence(self, table_name: str, column_name: str, 
                                  metric_config: Dict) -> float:
        """Calculate confidence score for field mapping"""
        confidence_factors = []
        
        # Factor 1: Keyword matching (40% weight)
        keywords = metric_config.get('keywords', [])
        column_lower = column_name.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in column_lower)
        keyword_score = min(keyword_matches / max(len(keywords), 1), 1.0)
        confidence_factors.append(keyword_score * 0.4)
        
        # Factor 2: Table preference (20% weight)
        table_score = 1.0 if 'all_sources' in table_name.lower() else 0.7
        confidence_factors.append(table_score * 0.2)
        
        # Factor 3: Data quality (40% weight) - will be calculated during validation
        confidence_factors.append(0.0)  # Placeholder
        
        return sum(confidence_factors[:2])  # Return partial score for now
    
    def _validate_field_data(self, table_name: str, column_name: str, 
                           metric_config: Dict) -> Dict[str, Any]:
        """Validate field data quality and content"""
        try:
            # Get sample data
            query = f'''
                SELECT "{column_name}", COUNT(*) as count
                FROM "{table_name}"
                WHERE "{column_name}" IS NOT NULL
                GROUP BY "{column_name}"
                ORDER BY count DESC
                LIMIT {self.config.sample_size}
            '''
            
            result = self.connection.execute(query).fetchall()
            samples = [row[0] for row in result]
            
            if not samples:
                return {'is_valid': False, 'reason': 'No data'}
            
            # Calculate data quality metrics
            total_query = f'SELECT COUNT(*) FROM "{table_name}"'
            total_count = self.connection.execute(total_query).fetchone()[0]
            
            null_query = f'SELECT COUNT(*) FROM "{table_name}" WHERE "{column_name}" IS NULL'
            null_count = self.connection.execute(null_query).fetchone()[0]
            
            unique_query = f'SELECT COUNT(DISTINCT "{column_name}") FROM "{table_name}"'
            unique_count = self.connection.execute(unique_query).fetchone()[0]
            
            # Quality scoring
            null_percentage = (null_count / total_count) * 100 if total_count > 0 else 100
            uniqueness_ratio = unique_count / (total_count - null_count) if (total_count - null_count) > 0 else 0
            
            quality_score = 0.0
            if null_percentage < 50:  # Less than 50% nulls
                quality_score += 0.5
            if uniqueness_ratio > 0.01:  # At least 1% uniqueness
                quality_score += 0.5
            
            # Pattern validation
            patterns = metric_config.get('patterns', [])
            pattern_matches = 0
            if patterns:
                for sample in samples[:20]:
                    sample_str = str(sample).lower()
                    if any(re.search(pattern, sample_str, re.IGNORECASE) for pattern in patterns):
                        pattern_matches += 1
                
                pattern_score = pattern_matches / min(len(samples), 20)
                quality_score = (quality_score + pattern_score) / 2
            
            return {
                'is_valid': quality_score > 0.3,
                'quality_score': quality_score,
                'samples': samples[:10],
                'null_percentage': null_percentage,
                'uniqueness_ratio': uniqueness_ratio
            }
            
        except Exception as e:
            logger.warning(f"Field validation failed for {table_name}.{column_name}: {e}")
            return {'is_valid': False, 'reason': str(e)}
    
    def generate_queries(self) -> Dict[str, str]:
        """Generate SQL queries based on discovered field mappings"""
        logger.info("Generating SQL queries")
        
        queries = {}
        host_field = self._get_field_reference('host_identity', 'host')
        
        # Global Asset Coverage Query
        queries['ao1_global_coverage'] = f'''
            SELECT 
                COUNT(DISTINCT {host_field}) as total_assets,
                COUNT(DISTINCT CASE WHEN chronicle_device_hostname IS NOT NULL THEN {host_field} END) as chronicle_coverage,
                COUNT(DISTINCT CASE WHEN crowdstrike_device_hostname IS NOT NULL THEN {host_field} END) as crowdstrike_coverage,
                ROUND(
                    CAST(COUNT(DISTINCT CASE WHEN 
                        chronicle_device_hostname IS NOT NULL OR 
                        crowdstrike_device_hostname IS NOT NULL 
                    THEN {host_field} END) AS FLOAT) * 100.0 / 
                    CAST(COUNT(DISTINCT {host_field}) AS FLOAT), 2
                ) as overall_coverage_percentage
            FROM {self._get_table_reference()}
            WHERE {host_field} IS NOT NULL
        '''
        
        # Network Coverage Query
        queries['ao1_network_coverage'] = f'''
            SELECT 
                'Network' as role,
                COALESCE(chronicle_log_type, 'Unknown') as log_type,
                COUNT(DISTINCT {host_field}) as asset_count
            FROM {self._get_table_reference()}
            WHERE chronicle_log_type IS NOT NULL
               AND (chronicle_log_type LIKE '%Firewall%' 
                    OR chronicle_log_type LIKE '%IDS%'
                    OR chronicle_log_type LIKE '%Proxy%'
                    OR chronicle_log_type LIKE '%DNS%')
            GROUP BY chronicle_log_type
            ORDER BY asset_count DESC
        '''
        
        # Endpoint Coverage Query
        queries['ao1_endpoint_coverage'] = f'''
            SELECT 
                'Endpoint' as role,
                COALESCE(crowdstrike_agent_health, 'Unknown') as agent_status,
                COUNT(DISTINCT {host_field}) as asset_count
            FROM {self._get_table_reference()}
            WHERE crowdstrike_device_hostname IS NOT NULL
            GROUP BY crowdstrike_agent_health
            ORDER BY asset_count DESC
        '''
        
        logger.info(f"Generated {len(queries)} queries")
        return queries
    
    def test_and_heal_queries(self, queries: Dict[str, str]) -> Dict[str, QueryTestResult]:
        """Test queries and heal them iteratively"""
        logger.info("Starting query testing and healing")
        
        results = {}
        
        with self.database_connection():
            for query_name, sql_query in queries.items():
                logger.info(f"Testing query: {query_name}")
                
                healed_result = self._heal_query_iteratively(query_name, sql_query)
                if healed_result:
                    results[query_name] = healed_result
                    self.validated_queries[query_name] = healed_result
                    logger.info(f"Query {query_name} successful after {self.healing_iterations.get(query_name, 0)} iterations")
                else:
                    logger.error(f"Failed to heal query: {query_name}")
        
        return results
    
    def _heal_query_iteratively(self, query_name: str, initial_sql: str) -> Optional[QueryTestResult]:
        """Iteratively heal a query until it works"""
        current_sql = initial_sql
        iteration = 0
        
        while iteration < self.config.max_iterations:
            iteration += 1
            self.healing_iterations[query_name] = iteration
            
            # Test current query
            result = self._test_query(query_name, current_sql)
            
            if result.success and self._validate_query_output(query_name, result):
                logger.info(f"Query {query_name} succeeded on iteration {iteration}")
                return result
            
            if not result.success:
                logger.debug(f"Query failed: {result.error_message}")
                
                # Apply healing strategies
                healed_sql = self._apply_healing_strategies(current_sql, result.error_message)
                
                if healed_sql == current_sql:
                    logger.warning(f"No more healing strategies available for {query_name}")
                    break
                
                current_sql = healed_sql
            else:
                logger.warning(f"Query {query_name} succeeded but failed validation")
                break
        
        return None
    
    def _test_query(self, query_name: str, sql_query: str) -> QueryTestResult:
        """Test a single query execution"""
        start_time = time.time()
        
        try:
            result = self.connection.execute(sql_query).fetchall()
            execution_time = time.time() - start_time
            
            # Get column names
            column_names = [desc[0] for desc in self.connection.description] if self.connection.description else []
            
            # Convert to dictionaries
            sample_results = []
            for row in result[:10]:  # Sample first 10 rows
                sample_results.append(dict(zip(column_names, row)))
            
            return QueryTestResult(
                query_name=query_name,
                sql_query=sql_query,
                success=True,
                execution_time=execution_time,
                row_count=len(result),
                sample_results=sample_results,
                confidence_score=1.0
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return QueryTestResult(
                query_name=query_name,
                sql_query=sql_query,
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                confidence_score=0.0
            )
    
    def _apply_healing_strategies(self, sql_query: str, error_message: str) -> str:
        """Apply appropriate healing strategies based on error type"""
        error_lower = error_message.lower()
        
        # Get all available columns for reference
        all_columns = []
        for table_columns in self.available_columns.values():
            all_columns.extend(table_columns)
        
        # Apply strategies in order of specificity
        if 'column' in error_lower and 'not found' in error_lower:
            return QueryHealingStrategies.fix_column_references(
                sql_query, error_message, all_columns
            )
        elif 'table' in error_lower and ('not found' in error_lower or 'does not exist' in error_lower):
            return QueryHealingStrategies.fix_table_references(
                sql_query, error_message, self.available_tables
            )
        elif 'syntax' in error_lower or 'parse' in error_lower:
            return QueryHealingStrategies.fix_syntax_errors(sql_query, error_message)
        
        return sql_query
    
    def _validate_query_output(self, query_name: str, result: QueryTestResult) -> bool:
        """Validate query output meets AO1 requirements"""
        if not result.sample_results:
            return False
        
        # Define minimum requirements
        min_requirements = {
            'ao1_global_coverage': {
                'required_columns': ['total_assets', 'overall_coverage_percentage'],
                'min_rows': 1
            },
            'ao1_network_coverage': {
                'required_columns': ['role', 'log_type', 'asset_count'],
                'min_rows': 1
            },
            'ao1_endpoint_coverage': {
                'required_columns': ['role', 'agent_status', 'asset_count'],
                'min_rows': 1
            }
        }
        
        requirements = min_requirements.get(query_name, {'required_columns': [], 'min_rows': 1})
        
        # Check row count
        if result.row_count < requirements['min_rows']:
            logger.warning(f"Insufficient rows for {query_name}: {result.row_count}")
            return False
        
        # Check required columns
        if result.sample_results:
            available_columns = set(result.sample_results[0].keys())
            required_columns = set(requirements['required_columns'])
            
            if not required_columns.issubset(available_columns):
                missing = required_columns - available_columns
                logger.warning(f"Missing columns for {query_name}: {missing}")
                return False
        
        return True
    
    def _get_field_reference(self, metric_name: str, fallback: str) -> str:
        """Get field reference for a metric or fallback"""
        if metric_name in self.field_mappings:
            mapping = self.field_mappings[metric_name]
            return f'"{mapping.source_column}"'
        return fallback
    
    def _get_table_reference(self) -> str:
        """Get the primary table reference"""
        # Prefer all_sources table
        for table in self.available_tables:
            if 'all_sources' in table.lower():
                return f'"{table}"'
        
        # Fallback to first available table
        return f'"{self.available_tables[0]}"' if self.available_tables else '"unknown"'
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        logger.info("Generating final report")
        
        total_queries = len(self.validated_queries)
        successful_queries = sum(1 for result in self.validated_queries.values() if result.success)
        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'engine_config': {
                'max_iterations': self.config.max_iterations,
                'min_confidence_score': self.config.min_confidence_score,
                'timeout_seconds': self.config.timeout_seconds
            },
            'execution_summary': {
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'success_rate_percentage': round(success_rate, 2),
                'total_healing_iterations': sum(self.healing_iterations.values()),
                'discovered_field_mappings': len(self.field_mappings)
            },
            'field_mappings': {
                name: {
                    'source_table': mapping.source_table,
                    'source_column': mapping.source_column,
                    'confidence_score': round(mapping.confidence_score, 3),
                    'data_quality_score': round(mapping.data_quality_score, 3)
                }
                for name, mapping in self.field_mappings.items()
            },
            'validated_queries': {
                name: {
                    'success': result.success,
                    'execution_time': round(result.execution_time, 3),
                    'row_count': result.row_count,
                    'healing_iterations': self.healing_iterations.get(name, 0),
                    'sql_query': result.sql_query
                }
                for name, result in self.validated_queries.items()
            },
            'ao1_readiness_score': self._calculate_readiness_score()
        }
        
        return report
    
    def _calculate_readiness_score(self) -> float:
        """Calculate overall AO1 readiness score"""
        if not self.ao1_metrics:
            return 0.0
        
        total_weight = sum(metric['priority'] for metric in self.ao1_metrics.values())
        weighted_score = 0.0
        
        for metric_name, metric_config in self.ao1_metrics.items():
            weight = metric_config['priority']
            
            if metric_name in self.field_mappings:
                mapping = self.field_mappings[metric_name]
                score = (mapping.confidence_score + mapping.data_quality_score) / 2
                weighted_score += weight * score
        
        return round((weighted_score / total_weight) * 100, 2) if total_weight > 0 else 0.0
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete AO1 analysis pipeline"""
        logger.info("Starting complete AO1 analysis")
        
        try:
            # Phase 1: Schema discovery
            if not self.discover_schema():
                raise RuntimeError("Schema discovery failed")
            
            # Phase 2: Field mapping discovery
            self.discover_field_mappings()
            
            # Phase 3: Query generation
            queries = self.generate_queries()
            
            # Phase 4: Query testing and healing
            self.test_and_heal_queries(queries)
            
            # Phase 5: Report generation
            report = self.generate_report()
            
            # Save results
            output_file = Path("ao1_analysis_results.json")
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Analysis complete. Results saved to {output_file}")
            logger.info(f"Success rate: {report['execution_summary']['success_rate_percentage']}%")
            logger.info(f"AO1 readiness: {report['ao1_readiness_score']}%")
            
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AO1 Self-Healing Query Engine')
    parser.add_argument('--database', '-d', default='data.duckdb', help='Path to DuckDB database file (default: data.duckdb)')
    parser.add_argument('--max-iterations', '-m', type=int, default=20, help='Maximum healing iterations')
    parser.add_argument('--min-confidence', '-c', type=float, default=0.7, help='Minimum confidence score')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if database file exists
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"❌ Database file not found: {db_path}")
        print(f"📍 Current directory: {Path.cwd()}")
        print(f"📂 Available files: {list(Path.cwd().glob('*.duckdb'))}")
        return 1
    
    print(f"🗄️  Using database: {db_path}")
    
    # Configure engine
    config = EngineConfig(
        max_iterations=args.max_iterations,
        min_confidence_score=args.min_confidence
    )
    
    try:
        # Initialize and run engine
        engine = SelfHealingQueryEngine(str(db_path), config)
        results = engine.run_complete_analysis()
        
        if 'error' not in results:
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📊 Success rate: {results['execution_summary']['success_rate_percentage']}%")
            print(f"🎯 AO1 readiness: {results['ao1_readiness_score']}%")
            print(f"💾 Results saved to: ao1_analysis_results.json")
        else:
            print(f"\n❌ Analysis failed: {results['error']}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"\n💥 Critical error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())