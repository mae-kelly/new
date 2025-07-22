#!/usr/bin/env python3

import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from models import FieldIntelligence, QueryResult
from database_connector import DatabaseConnector

logger = logging.getLogger(__name__)

@dataclass
class QueryAttempt:
    sql: str
    field_combination: List[str]
    result_count: int
    execution_success: bool
    validation_score: float
    validation_notes: List[str]
    execution_time_ms: float = 0.0

@dataclass
class MetricResult:
    metric_name: str
    best_query: QueryAttempt
    metric_value: float
    confidence_score: float
    alternative_queries: List[QueryAttempt]

class AdaptiveQueryEngine:
    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        self.field_combinations_tried = set()
        self.successful_patterns = {}
        
    def find_global_visibility_score(self, fields: List[FieldIntelligence]) -> MetricResult:
        """Try different approaches to calculate global visibility until we get sensible numbers"""
        
        attempts = []
        
        # Strategy 1: CMDB assets vs logging assets
        cmdb_attempts = self._try_cmdb_based_visibility(fields)
        attempts.extend(cmdb_attempts)
        
        # Strategy 2: Host-based visibility 
        host_attempts = self._try_host_based_visibility(fields)
        attempts.extend(host_attempts)
        
        # Strategy 3: Event source based visibility
        source_attempts = self._try_source_based_visibility(fields)
        attempts.extend(source_attempts)
        
        # Strategy 4: Time-based active assets
        time_attempts = self._try_time_based_visibility(fields)
        attempts.extend(time_attempts)
        
        # Validate and pick best approach
        best_attempt = self._validate_and_select_best(attempts, 'global_visibility')
        
        return MetricResult(
            metric_name='Global Visibility Score',
            best_query=best_attempt,
            metric_value=self._extract_percentage_from_result(best_attempt),
            confidence_score=best_attempt.validation_score,
            alternative_queries=attempts
        )
        
    def _try_cmdb_based_visibility(self, fields: List[FieldIntelligence]) -> List[QueryAttempt]:
        """Try CMDB-based visibility calculations"""
        attempts = []
        
        # Find CMDB-like fields
        cmdb_fields = [f for f in fields if any(kw in f.name.lower() for kw in 
                      ['cmdb', 'asset', 'inventory', 'ci_name', 'configuration_item'])]
        
        # Find logging/event fields  
        log_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                     ['log', 'event', 'message', 'timestamp', 'source', 'index'])]
        
        if not cmdb_fields or not log_fields:
            return attempts
            
        for cmdb_field in cmdb_fields[:3]:  # Try top 3 CMDB fields
            for log_field in log_fields[:3]:   # Try top 3 log fields
                
                # Try direct join
                sql = self._build_cmdb_visibility_query(cmdb_field, log_field, join_type='DIRECT')
                attempt = self._execute_and_validate_query(sql, [cmdb_field.get_key(), log_field.get_key()])
                if attempt:
                    attempts.append(attempt)
                
                # Try hostname-based join
                if any(kw in cmdb_field.name.lower() for kw in ['host', 'name']):
                    sql = self._build_cmdb_visibility_query(cmdb_field, log_field, join_type='HOSTNAME')
                    attempt = self._execute_and_validate_query(sql, [cmdb_field.get_key(), log_field.get_key()])
                    if attempt:
                        attempts.append(attempt)
                        
        return attempts
        
    def _try_host_based_visibility(self, fields: List[FieldIntelligence]) -> List[QueryAttempt]:
        """Try host/hostname based visibility"""
        attempts = []
        
        # Find hostname fields
        host_fields = [f for f in fields if any(kw in f.name.lower() for kw in 
                      ['host', 'hostname', 'server', 'computer', 'device', 'node'])]
        
        # Find timestamp fields for recency
        time_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                      ['time', 'timestamp', 'date', 'created', '_at'])]
        
        if not host_fields:
            return attempts
            
        for host_field in host_fields[:3]:
            # Try simple host count
            sql = f"""
            WITH all_hosts AS (
                SELECT COUNT(DISTINCT {host_field.name}) as total_hosts
                FROM {host_field.table}
                WHERE {host_field.name} IS NOT NULL
            ),
            active_hosts AS (
                SELECT COUNT(DISTINCT {host_field.name}) as logging_hosts
                FROM {host_field.table}
                WHERE {host_field.name} IS NOT NULL
                  AND {self._get_time_filter(host_field, time_fields)}
            )
            SELECT 
                ah.total_hosts,
                ac.logging_hosts,
                ROUND(100.0 * ac.logging_hosts / ah.total_hosts, 2) as visibility_percentage
            FROM all_hosts ah, active_hosts ac
            """
            
            attempt = self._execute_and_validate_query(sql, [host_field.get_key()])
            if attempt:
                attempts.append(attempt)
                
        return attempts
        
    def _try_source_based_visibility(self, fields: List[FieldIntelligence]) -> List[QueryAttempt]:
        """Try source/platform based visibility"""
        attempts = []
        
        # Find source fields
        source_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                        ['source', 'sourcetype', 'platform', 'system', 'tool'])]
        
        # Find asset identifier fields
        asset_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                       ['asset', 'host', 'hostname', 'device', 'endpoint'])]
        
        if not source_fields or not asset_fields:
            return attempts
            
        for source_field in source_fields[:2]:
            for asset_field in asset_fields[:2]:
                if source_field.table == asset_field.table:
                    sql = f"""
                    WITH platform_assets AS (
                        SELECT 
                            COUNT(DISTINCT {asset_field.name}) as total_from_platform,
                            {source_field.name} as platform
                        FROM {source_field.table}
                        WHERE {asset_field.name} IS NOT NULL 
                          AND {source_field.name} IS NOT NULL
                        GROUP BY {source_field.name}
                    ),
                    overall_totals AS (
                        SELECT 
                            COUNT(DISTINCT {asset_field.name}) as grand_total
                        FROM {asset_field.table}
                        WHERE {asset_field.name} IS NOT NULL
                    )
                    SELECT 
                        SUM(pa.total_from_platform) as assets_with_sources,
                        ot.grand_total as total_assets,
                        ROUND(100.0 * SUM(pa.total_from_platform) / ot.grand_total, 2) as visibility_percentage
                    FROM platform_assets pa, overall_totals ot
                    """
                    
                    attempt = self._execute_and_validate_query(sql, [source_field.get_key(), asset_field.get_key()])
                    if attempt:
                        attempts.append(attempt)
                        
        return attempts
        
    def _try_time_based_visibility(self, fields: List[FieldIntelligence]) -> List[QueryAttempt]:
        """Try time-based active asset visibility"""
        attempts = []
        
        # Find timestamp fields
        time_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                      ['timestamp', '_time', 'created', 'occurred', 'date'])]
        
        # Find asset fields
        asset_fields = [f for f in fields if any(kw in f.name.lower() for kw in
                       ['host', 'hostname', 'asset', 'device', 'source'])]
        
        if not time_fields or not asset_fields:
            return attempts
            
        for time_field in time_fields[:2]:
            for asset_field in asset_fields[:3]:
                if time_field.table == asset_field.table:
                    # Try different time windows
                    for days in [1, 7, 30]:
                        sql = f"""
                        WITH recent_assets AS (
                            SELECT COUNT(DISTINCT {asset_field.name}) as active_assets
                            FROM {time_field.table}
                            WHERE {asset_field.name} IS NOT NULL
                              AND {self._build_time_filter(time_field.name, days)}
                        ),
                        all_time_assets AS (
                            SELECT COUNT(DISTINCT {asset_field.name}) as total_assets  
                            FROM {asset_field.table}
                            WHERE {asset_field.name} IS NOT NULL
                        )
                        SELECT 
                            ra.active_assets,
                            ata.total_assets,
                            ROUND(100.0 * ra.active_assets / ata.total_assets, 2) as visibility_percentage,
                            {days} as time_window_days
                        FROM recent_assets ra, all_time_assets ata
                        """
                        
                        attempt = self._execute_and_validate_query(sql, [time_field.get_key(), asset_field.get_key()])
                        if attempt:
                            attempts.append(attempt)
                            
        return attempts
        
    def _execute_and_validate_query(self, sql: str, field_combination: List[str]) -> Optional[QueryAttempt]:
        """Execute query and validate results make sense"""
        
        combo_key = "|".join(sorted(field_combination))
        if combo_key in self.field_combinations_tried:
            return None
            
        self.field_combinations_tried.add(combo_key)
        
        try:
            import time
            start_time = time.time()
            
            result = self.db_connector.execute_query(sql)
            
            execution_time = (time.time() - start_time) * 1000
            
            if not result:
                return QueryAttempt(
                    sql=sql,
                    field_combination=field_combination,
                    result_count=0,
                    execution_success=False,
                    validation_score=0.0,
                    validation_notes=["No results returned"],
                    execution_time_ms=execution_time
                )
                
            # Validate the results
            validation_score, validation_notes = self._validate_visibility_results(result)
            
            return QueryAttempt(
                sql=sql,
                field_combination=field_combination,
                result_count=len(result),
                execution_success=True,
                validation_score=validation_score,
                validation_notes=validation_notes,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.debug(f"Query execution failed: {e}")
            return QueryAttempt(
                sql=sql,
                field_combination=field_combination,
                result_count=0,
                execution_success=False,
                validation_score=0.0,
                validation_notes=[f"Execution error: {str(e)}"],
                execution_time_ms=0.0
            )
            
    def _validate_visibility_results(self, result: List[Tuple]) -> Tuple[float, List[str]]:
        """Validate if visibility results make business sense"""
        
        if not result or len(result) == 0:
            return 0.0, ["No data returned"]
            
        validation_notes = []
        score_components = []
        
        # Look for percentage values
        for row in result:
            for value in row:
                if isinstance(value, (int, float)):
                    # Check if it looks like a percentage
                    if 0 <= value <= 100:
                        # Good percentage range
                        if 10 <= value <= 95:  # Reasonable visibility range
                            score_components.append(0.9)
                            validation_notes.append(f"Reasonable visibility percentage: {value}%")
                        elif value < 10:
                            score_components.append(0.6) 
                            validation_notes.append(f"Low visibility detected: {value}%")
                        else:  # > 95%
                            score_components.append(0.8)
                            validation_notes.append(f"Very high visibility: {value}%")
                    elif value > 100:
                        score_components.append(0.3)
                        validation_notes.append(f"Invalid percentage > 100%: {value}")
                    elif 100 < value < 10000:  # Could be raw counts
                        score_components.append(0.7)
                        validation_notes.append(f"Appears to be asset count: {value}")
                    else:
                        score_components.append(0.4)
                        validation_notes.append(f"Unusual numeric value: {value}")
                        
        # Check for expected columns/structure
        if len(result[0]) >= 2:  # Should have at least 2 columns for comparison
            score_components.append(0.8)
            validation_notes.append("Query returned expected multi-column structure")
        else:
            score_components.append(0.5)
            validation_notes.append("Query returned single column - may be incomplete")
            
        # Check result count reasonableness
        if 1 <= len(result) <= 10:  # Reasonable for summary metrics
            score_components.append(0.8)
            validation_notes.append("Appropriate result count for visibility metric")
        elif len(result) > 1000:
            score_components.append(0.4)
            validation_notes.append("Very large result set - may need aggregation")
        else:
            score_components.append(0.6)
            
        final_score = sum(score_components) / len(score_components) if score_components else 0.0
        return min(1.0, final_score), validation_notes
        
    def _validate_and_select_best(self, attempts: List[QueryAttempt], metric_type: str) -> QueryAttempt:
        """Select the best query attempt based on validation scores"""
        
        if not attempts:
            return QueryAttempt(
                sql="-- No valid queries found",
                field_combination=[],
                result_count=0,
                execution_success=False,
                validation_score=0.0,
                validation_notes=["No query attempts generated"]
            )
            
        # Filter successful attempts
        successful_attempts = [a for a in attempts if a.execution_success and a.validation_score > 0.5]
        
        if not successful_attempts:
            # Return best failed attempt for debugging
            return max(attempts, key=lambda a: a.validation_score)
            
        # Sort by validation score, then by execution time
        successful_attempts.sort(key=lambda a: (a.validation_score, -a.execution_time_ms), reverse=True)
        
        return successful_attempts[0]
        
    def _extract_percentage_from_result(self, attempt: QueryAttempt) -> float:
        """Extract the main percentage metric from query results"""
        if not attempt.execution_success:
            return 0.0
            
        try:
            # Re-execute to get actual results (in real implementation, would cache)
            result = self.db_connector.execute_query(attempt.sql)
            if result and len(result) > 0:
                # Look for percentage-like values
                for row in result:
                    for value in row:
                        if isinstance(value, (int, float)) and 0 <= value <= 100:
                            return float(value)
        except:
            pass
            
        return 0.0
        
    def _build_cmdb_visibility_query(self, cmdb_field: FieldIntelligence, 
                                   log_field: FieldIntelligence, join_type: str) -> str:
        """Build CMDB-based visibility query"""
        
        if join_type == 'DIRECT':
            return f"""
            WITH cmdb_assets AS (
                SELECT COUNT(DISTINCT {cmdb_field.name}) as total_cmdb_assets
                FROM {cmdb_field.table}
                WHERE {cmdb_field.name} IS NOT NULL
            ),
            logging_assets AS (
                SELECT COUNT(DISTINCT {log_field.name}) as logging_assets  
                FROM {log_field.table}
                WHERE {log_field.name} IS NOT NULL
            )
            SELECT 
                ca.total_cmdb_assets,
                la.logging_assets,
                ROUND(100.0 * la.logging_assets / ca.total_cmdb_assets, 2) as visibility_percentage
            FROM cmdb_assets ca, logging_assets la
            """
        else:  # HOSTNAME join
            return f"""
            WITH cmdb_hosts AS (
                SELECT DISTINCT {cmdb_field.name} as hostname
                FROM {cmdb_field.table} 
                WHERE {cmdb_field.name} IS NOT NULL
            ),
            logging_hosts AS (
                SELECT DISTINCT {log_field.name} as hostname
                FROM {log_field.table}
                WHERE {log_field.name} IS NOT NULL
            )
            SELECT 
                COUNT(ch.hostname) as total_assets,
                COUNT(lh.hostname) as visible_assets,
                ROUND(100.0 * COUNT(lh.hostname) / COUNT(ch.hostname), 2) as visibility_percentage
            FROM cmdb_hosts ch
            LEFT JOIN logging_hosts lh ON ch.hostname = lh.hostname
            """
            
    def _get_time_filter(self, field: FieldIntelligence, time_fields: List[FieldIntelligence]) -> str:
        """Get appropriate time filter for recency"""
        matching_time_fields = [tf for tf in time_fields if tf.table == field.table]
        
        if matching_time_fields:
            time_field = matching_time_fields[0] 
            return self._build_time_filter(time_field.name, 7)
        else:
            return "1=1"
            
    def _build_time_filter(self, time_field_name: str, days: int) -> str:
        """Build time filter SQL"""
        return f"DATE({time_field_name}) >= DATE('now', '-{days} days')"