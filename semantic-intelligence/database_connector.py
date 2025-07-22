#!/usr/bin/env python3

import sqlite3
import logging
from typing import Dict, List, Tuple, Any, Optional

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class DatabaseConnector:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.connection = None
        self.is_duckdb = False
        
    def connect(self) -> bool:
        try:
            if self.database_path.endswith('.duckdb') and DUCKDB_AVAILABLE:
                self.connection = duckdb.connect(self.database_path)
                self.is_duckdb = True
                logger.info("Connected to DuckDB database")
            else:
                self.connection = sqlite3.connect(self.database_path)
                self.is_duckdb = False
                logger.info("Connected to SQLite database")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
            
    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            
    def discover_schema(self) -> Dict[str, List[Tuple[str, str]]]:
        if not self.connection:
            return {}
            
        schema = {}
        try:
            if self.is_duckdb:
                schema = self._discover_duckdb_schema()
            else:
                schema = self._discover_sqlite_schema()
                
            logger.info(f"Discovered {len(schema)} tables with {sum(len(cols) for cols in schema.values())} columns")
            return schema
            
        except Exception as e:
            logger.error(f"Schema discovery failed: {e}")
            return {}
            
    def _discover_duckdb_schema(self) -> Dict[str, List[Tuple[str, str]]]:
        schema = {}
        
        # Get table names
        tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        tables = [row[0] for row in self.connection.execute(tables_query).fetchall()]
        
        # Get columns for each table
        for table in tables:
            columns_query = f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table}' AND table_schema = 'main'
            ORDER BY ordinal_position
            """
            columns = self.connection.execute(columns_query).fetchall()
            schema[table] = [(col[0], col[1]) for col in columns]
            
        return schema
        
    def _discover_sqlite_schema(self) -> Dict[str, List[Tuple[str, str]]]:
        schema = {}
        
        # Get table names
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables = [row[0] for row in self.connection.execute(tables_query).fetchall()]
        
        # Get columns for each table
        for table in tables:
            columns = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
            schema[table] = [(col[1], col[2]) for col in columns]  # (name, type)
            
        return schema
        
    def sample_field_data(self, table: str, column: str, sample_size: int = 1000) -> List[Any]:
        if not self.connection:
            return []
            
        samples = []
        try:
            # Use different sampling strategies based on database type
            if self.is_duckdb:
                query = f"""
                SELECT DISTINCT {column} 
                FROM {table} 
                WHERE {column} IS NOT NULL 
                USING SAMPLE {min(sample_size, 10000)}
                LIMIT {sample_size}
                """
            else:
                query = f"""
                SELECT DISTINCT {column} 
                FROM {table} 
                WHERE {column} IS NOT NULL 
                ORDER BY RANDOM() 
                LIMIT {sample_size}
                """
                
            result = self.connection.execute(query).fetchall()
            samples = [row[0] for row in result]
            
        except Exception as e:
            logger.debug(f"Sampling failed for {table}.{column}: {e}")
            # Fallback to simple query
            try:
                fallback_query = f"""
                SELECT DISTINCT {column} 
                FROM {table} 
                WHERE {column} IS NOT NULL 
                LIMIT {sample_size}
                """
                result = self.connection.execute(fallback_query).fetchall()
                samples = [row[0] for row in result]
            except Exception as e2:
                logger.debug(f"Fallback sampling also failed for {table}.{column}: {e2}")
                
        return samples
        
    def get_table_stats(self, table: str) -> Dict[str, Any]:
        if not self.connection:
            return {}
            
        stats = {}
        try:
            # Get row count
            count_query = f"SELECT COUNT(*) FROM {table}"
            row_count = self.connection.execute(count_query).fetchone()[0]
            stats['row_count'] = row_count
            
            # Get table size (approximate)
            if self.is_duckdb:
                # DuckDB doesn't have easy table size queries
                stats['size_bytes'] = None
            else:
                # SQLite specific
                size_query = f"SELECT SUM(pgsize) FROM dbstat WHERE name = '{table}'"
                try:
                    size_result = self.connection.execute(size_query).fetchone()
                    stats['size_bytes'] = size_result[0] if size_result and size_result[0] else None
                except:
                    stats['size_bytes'] = None
                    
        except Exception as e:
            logger.debug(f"Failed to get stats for table {table}: {e}")
            stats['row_count'] = 0
            stats['size_bytes'] = None
            
        return stats
        
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Tuple]:
        if not self.connection:
            return []
            
        try:
            if params:
                result = self.connection.execute(query, params).fetchall()
            else:
                result = self.connection.execute(query).fetchall()
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
            
    def validate_query(self, query: str) -> bool:
        if not self.connection:
            return False
            
        try:
            # Try to prepare the query with EXPLAIN
            explain_query = f"EXPLAIN {query}"
            self.connection.execute(explain_query)
            return True
        except Exception as e:
            logger.debug(f"Query validation failed: {e}")
            return False
            
    def get_column_stats(self, table: str, column: str) -> Dict[str, Any]:
        if not self.connection:
            return {}
            
        stats = {}
        try:
            # Basic column statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as total_count,
                COUNT({column}) as non_null_count,
                COUNT(DISTINCT {column}) as unique_count
            FROM {table}
            """
            
            result = self.connection.execute(stats_query).fetchone()
            if result:
                stats['total_count'] = result[0]
                stats['non_null_count'] = result[1]
                stats['unique_count'] = result[2]
                stats['null_count'] = result[0] - result[1]
                stats['completeness'] = result[1] / result[0] if result[0] > 0 else 0.0
                stats['uniqueness'] = result[2] / result[1] if result[1] > 0 else 0.0
                
        except Exception as e:
            logger.debug(f"Failed to get column stats for {table}.{column}: {e}")
            
        return stats