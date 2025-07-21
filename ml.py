#!/usr/bin/env python3
import logging
import json
import time
import re
import sqlite3
import sys
import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MinimalFieldIntelligence:
    name: str
    table: str
    data_type: str
    sample_values: List[Any] = field(default_factory=list)
    semantic_type: str = "unknown"
    confidence: float = 0.0
    ao1_relevance: float = 0.0
    business_context: str = ""
    security_relevance: float = 0.0
    quality_score: float = 0.0
    intelligence_score: float = 0.0
    pattern_strength: float = 0.0
    
@dataclass
class MinimalIntelligentQuery:
    name: str
    description: str
    sql: str
    ao1_requirement: str
    priority: str
    perfection_score: float = 0.0
    validation_status: str = "untested"

class MinimalSemanticEngine:
    def __init__(self):
        self.patterns = {
            'hostname': {
                'regex': [r'.*\.(com|net|org|edu|gov|local)$', r'^(web|db|mail|server|host)', r'\b(srv|web|db)\d*\b'],
                'keywords': ['server', 'computer', 'machine', 'device', 'host'],
                'weight': 0.95
            },
            'ip_address': {
                'regex': [r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$', r'^10\.|^172\.|^192\.168\.'],
                'keywords': ['address', 'ip', 'network'],
                'weight': 0.95
            },
            'security_event': {
                'regex': [r'\b(alert|critical|warning|error|breach|attack|threat)\b', r'\b(block|deny|drop|quarantine)\b'],
                'keywords': ['security', 'threat', 'incident', 'alert'],
                'weight': 0.98
            },
            'cloud_resource': {
                'regex': [r'\b(aws|azure|gcp|google|amazon)\b', r'\b(ec2|s3|rds|lambda|vpc)\b'],
                'keywords': ['cloud', 'virtual', 'container', 'service'],
                'weight': 0.88
            },
            'network_device': {
                'regex': [r'\b(firewall|router|switch|proxy|gateway)\b', r'\b(cisco|juniper|fortinet)\b'],
                'keywords': ['network', 'device', 'equipment'],
                'weight': 0.85
            },
            'endpoint': {
                'regex': [r'\b(windows|linux|macos|ubuntu)\b', r'\b(workstation|laptop|desktop|endpoint)\b'],
                'keywords': ['computer', 'workstation', 'device', 'system'],
                'weight': 0.85
            },
            'application': {
                'regex': [r'\b(web|http|https|api|service|application)\b', r'\b(apache|nginx|database|sql)\b'],
                'keywords': ['application', 'software', 'service'],
                'weight': 0.75
            },
            'identity': {
                'regex': [r'\b(user|username|userid|account|identity)\b', r'\b(authentication|authorization)\b'],
                'keywords': ['user', 'identity', 'account', 'person'],
                'weight': 0.90
            },
            'log_type': {
                'regex': [r'\b(syslog|eventlog|audit|access|error)\b', r'\b(info|warn|error|fatal)\b'],
                'keywords': ['log', 'record', 'event', 'message'],
                'weight': 0.80
            },
            'geographic': {
                'regex': [r'\b(country|region|city|state|datacenter|site|location)\b'],
                'keywords': ['location', 'place', 'region', 'area'],
                'weight': 0.75
            },
            'asset_identifier': {
                'regex': [r'\b(asset.?id|device.?id|serial|uuid|guid)\b'],
                'keywords': ['identifier', 'id', 'tag', 'number'],
                'weight': 0.95
            },
            'security_control': {
                'regex': [r'\b(edr|antivirus|dlp|siem|crowdstrike|tanium|splunk)\b'],
                'keywords': ['security', 'protection', 'defense', 'control'],
                'weight': 0.98
            },
            'business_unit': {
                'regex': [r'\b(department|division|unit|organization)\b', r'\b(finance|hr|it|security|operations)\b'],
                'keywords': ['organization', 'department', 'unit'],
                'weight': 0.65
            },
            'compliance': {
                'regex': [r'\b(compliance|audit|regulation|standard)\b', r'\b(sox|pci|hipaa|gdpr|iso|nist)\b'],
                'keywords': ['compliance', 'regulation', 'standard'],
                'weight': 0.85
            },
            'performance': {
                'regex': [r'\b(cpu|memory|disk|network|performance|metric)\b'],
                'keywords': ['performance', 'metric', 'measurement'],
                'weight': 0.55
            },
            'time_field': {
                'regex': [r'\b(timestamp|datetime|date|time|created|modified)\b', r'\d{4}-\d{2}-\d{2}|\d{10}|\d{13}'],
                'keywords': ['time', 'date', 'timestamp'],
                'weight': 0.75
            }
        }
        
    def analyze_field(self, field_name: str, sample_values: List[Any]) -> Dict[str, Any]:
        field_text = field_name.lower()
        sample_text = ' '.join([str(val) for val in sample_values if val is not None])[:1000].lower()
        
        semantic_scores = {}
        
        for semantic_type, type_data in self.patterns.items():
            score = 0.0
            
            for pattern in type_data['regex']:
                field_matches = len(re.findall(pattern, field_text, re.IGNORECASE))
                sample_matches = len(re.findall(pattern, sample_text, re.IGNORECASE))
                score += (field_matches * 0.4 + sample_matches * 0.6) / max(len(sample_values), 1)
                
            for keyword in type_data['keywords']:
                if keyword in field_text or keyword in sample_text:
                    score += 0.2
                    
            semantic_scores[semantic_type] = min(score, 1.0)
            
        return {
            'semantic_scores': semantic_scores,
            'pattern_strength': self.calculate_pattern_strength(sample_values),
            'quality_metrics': self.calculate_quality_metrics(sample_values)
        }
        
    def calculate_pattern_strength(self, values: List[Any]) -> float:
        if not values:
            return 0.0
            
        pattern_counts = Counter()
        for value in values[:100]:
            if value is not None:
                pattern = re.sub(r'\d', 'N', re.sub(r'[a-zA-Z]', 'A', str(value)))
                pattern_counts[pattern] += 1
                
        if not pattern_counts:
            return 0.0
            
        most_common_count = pattern_counts.most_common(1)[0][1]
        return most_common_count / min(len(values), 100)
        
    def calculate_quality_metrics(self, values: List[Any]) -> Dict:
        if not values:
            return {'completeness': 0, 'uniqueness': 0}
            
        non_null_count = len([v for v in values if v is not None])
        unique_count = len(set(values))
        
        return {
            'completeness': non_null_count / len(values),
            'uniqueness': unique_count / len(values)
        }

class MinimalIntelligentAO1Engine:
    def __init__(self, database_path: str, perfection_threshold: float = 0.90, max_iterations: int = 10000):
        self.database_path = database_path
        self.perfection_threshold = perfection_threshold
        self.max_iterations = max_iterations
        self.field_intelligence: Dict[str, MinimalFieldIntelligence] = {}
        self.queries: List[MinimalIntelligentQuery] = []
        self.semantic_engine = MinimalSemanticEngine()
        self.iteration_count = 0
        self.perfection_score = 0.0
        self.connection = None
        
        self.ao1_requirements = {
            'global_view': {'description': 'Global Asset Coverage', 'priority': 'Critical'},
            'infrastructure_type': {'description': 'Infrastructure Type breakdown', 'priority': 'Critical'},
            'security_control_coverage': {'description': 'Security Control Coverage', 'priority': 'Critical'},
            'network_role_coverage': {'description': 'Network Role Coverage', 'priority': 'High'},
            'endpoint_role_coverage': {'description': 'Endpoint Role Coverage', 'priority': 'High'},
            'identity_authentication': {'description': 'Identity & Authentication', 'priority': 'High'},
            'logging_compliance': {'description': 'Logging Compliance', 'priority': 'Critical'},
            'visibility_factors': {'description': 'Visibility Factors', 'priority': 'Critical'}
        }
        
    def connect_database(self):
        try:
            if self.database_path.endswith('.duckdb') and DUCKDB_AVAILABLE:
                self.connection = duckdb.connect(self.database_path)
                logger.info("Connected to DuckDB database")
            else:
                self.connection = sqlite3.connect(self.database_path)
                logger.info("Connected to SQLite database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
            
    def discover_schema(self) -> Dict[str, List[str]]:
        schema = {}
        try:
            if DUCKDB_AVAILABLE and isinstance(self.connection, duckdb.DuckDBPyConnection):
                tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                tables = [row[0] for row in self.connection.execute(tables_query).fetchall()]
                
                for table in tables:
                    columns_query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}'"
                    columns = self.connection.execute(columns_query).fetchall()
                    schema[table] = [(col[0], col[1]) for col in columns]
            else:
                tables = [row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                for table in tables:
                    columns = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
                    schema[table] = [(col[1], col[2]) for col in columns]
                    
            logger.info(f"Schema discovery: {len(schema)} tables, {sum(len(cols) for cols in schema.values())} columns")
            return schema
        except Exception as e:
            logger.error(f"Schema discovery failed: {e}")
            return {}
            
    def sample_field_data(self, table: str, column: str, sample_size: int = 1000) -> List[Any]:
        samples = []
        try:
            query = f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT {sample_size}"
            samples = [row[0] for row in self.connection.execute(query).fetchall()]
        except Exception as e:
            logger.debug(f"Sampling failed for {table}.{column}: {e}")
            
        return samples
        
    def analyze_field_intelligent(self, table: str, column: str, data_type: str) -> MinimalFieldIntelligence:
        try:
            sample_values = self.sample_field_data(table, column)
            analysis = self.semantic_engine.analyze_field(column, sample_values)
            
            semantic_scores = analysis['semantic_scores']
            semantic_type = max(semantic_scores.keys(), key=lambda k: semantic_scores[k]) if semantic_scores else "unknown"
            confidence = max(semantic_scores.values()) if semantic_scores else 0.0
            
            ao1_relevance = self.calculate_ao1_relevance(semantic_type, semantic_scores)
            business_context = self.infer_business_context(semantic_type)
            security_relevance = self.calculate_security_relevance(semantic_type, semantic_scores)
            quality_score = self.calculate_quality_score(analysis['quality_metrics'])
            intelligence_score = self.calculate_intelligence_score(confidence, quality_score, ao1_relevance)
            
            field_intel = MinimalFieldIntelligence(
                name=column,
                table=table,
                data_type=data_type,
                sample_values=sample_values[:20],
                semantic_type=semantic_type,
                confidence=confidence,
                ao1_relevance=ao1_relevance,
                business_context=business_context,
                security_relevance=security_relevance,
                quality_score=quality_score,
                intelligence_score=intelligence_score,
                pattern_strength=analysis['pattern_strength']
            )
            
            logger.info(f"Analyzed {table}.{column}: {semantic_type} (confidence: {confidence:.3f})")
            return field_intel
            
        except Exception as e:
            logger.error(f"Field analysis failed for {table}.{column}: {e}")
            return MinimalFieldIntelligence(name=column, table=table, data_type=data_type)
            
    def calculate_ao1_relevance(self, semantic_type: str, semantic_scores: Dict[str, float]) -> float:
        pattern_data = self.semantic_engine.patterns.get(semantic_type, {})
        weight = pattern_data.get('weight', 0.5)
        base_score = semantic_scores.get(semantic_type, 0.0) * weight
        
        high_value_types = ['hostname', 'ip_address', 'security_event', 'security_control', 'asset_identifier']
        bonus = sum(semantic_scores.get(hvt, 0.0) * 0.1 for hvt in high_value_types if hvt != semantic_type)
        
        return min(base_score + bonus, 1.0)
        
    def infer_business_context(self, semantic_type: str) -> str:
        contexts = {
            'hostname': 'IT Infrastructure - Server and endpoint identification',
            'ip_address': 'Network Infrastructure - IP address management',
            'security_event': 'Security Operations - Threat detection and response',
            'cloud_resource': 'Cloud Infrastructure - Cloud service management',
            'network_device': 'Network Operations - Network device management',
            'endpoint': 'Endpoint Management - Endpoint security and monitoring',
            'application': 'Application Operations - Application monitoring',
            'identity': 'Identity Management - Identity and access control',
            'log_type': 'Log Management - Log analytics and monitoring',
            'geographic': 'Geographic Intelligence - Location-based analytics',
            'asset_identifier': 'Asset Management - Asset tracking and lifecycle',
            'security_control': 'Security Controls - Security orchestration',
            'business_unit': 'Business Operations - Organizational analytics',
            'compliance': 'Compliance Management - Compliance monitoring',
            'performance': 'Performance Intelligence - Performance analytics',
            'time_field': 'Temporal Analytics - Time-series analysis'
        }
        return contexts.get(semantic_type, 'Data Analytics')
        
    def calculate_security_relevance(self, semantic_type: str, semantic_scores: Dict[str, float]) -> float:
        security_weights = {
            'security_event': 1.0, 'security_control': 1.0, 'identity': 0.9,
            'hostname': 0.8, 'ip_address': 0.8, 'network_device': 0.8,
            'endpoint': 0.8, 'compliance': 0.9
        }
        return semantic_scores.get(semantic_type, 0.0) * security_weights.get(semantic_type, 0.3)
        
    def calculate_quality_score(self, quality_metrics: Dict) -> float:
        completeness = quality_metrics.get('completeness', 0.0)
        uniqueness = quality_metrics.get('uniqueness', 0.0)
        return (completeness * 0.6 + uniqueness * 0.4)
        
    def calculate_intelligence_score(self, confidence: float, quality_score: float, ao1_relevance: float) -> float:
        return (confidence * 0.4 + quality_score * 0.3 + ao1_relevance * 0.3)
        
    def generate_intelligent_query(self, requirement: str, requirement_data: Dict) -> MinimalIntelligentQuery:
        relevant_fields = self.find_relevant_fields(requirement)
        
        if not relevant_fields:
            logger.warning(f"No relevant fields found for requirement: {requirement}")
            return None
            
        sql = self.generate_ao1_query(requirement, relevant_fields)
        
        query = MinimalIntelligentQuery(
            name=f"AO1_{requirement.upper()}",
            description=requirement_data['description'],
            sql=sql,
            ao1_requirement=requirement,
            priority=requirement_data['priority']
        )
        
        query.perfection_score = self.calculate_perfection_score(query, relevant_fields)
        query.validation_status = self.validate_query(query.sql)
        
        logger.info(f"Generated query {requirement}: perfection={query.perfection_score:.3f}")
        return query
        
    def find_relevant_fields(self, requirement: str) -> List[MinimalFieldIntelligence]:
        requirement_mappings = {
            'global_view': ['asset_identifier', 'hostname', 'ip_address'],
            'infrastructure_type': ['cloud_resource', 'network_device', 'endpoint', 'application'],
            'security_control_coverage': ['security_control', 'endpoint', 'security_event'],
            'network_role_coverage': ['network_device', 'security_event', 'ip_address'],
            'endpoint_role_coverage': ['endpoint', 'security_control', 'log_type'],
            'identity_authentication': ['identity', 'security_event', 'endpoint'],
            'logging_compliance': ['log_type', 'security_event', 'compliance'],
            'visibility_factors': ['hostname', 'ip_address', 'asset_identifier']
        }
        
        target_types = requirement_mappings.get(requirement, [])
        
        candidates = []
        for field in self.field_intelligence.values():
            if field.semantic_type in target_types:
                score = field.ao1_relevance * 0.4 + field.confidence * 0.3 + field.quality_score * 0.3
                candidates.append((field, score))
                
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [field for field, score in candidates[:8] if score > 0.5]
        
    def generate_ao1_query(self, requirement: str, relevant_fields: List[MinimalFieldIntelligence]) -> str:
        if not relevant_fields:
            return f"-- No relevant fields found for {requirement}"
            
        primary_field = relevant_fields[0]
        
        sql = f"""
        -- AO1 {requirement.replace('_', ' ').title()} Analysis
        WITH analysis AS (
            SELECT 
                {primary_field.name} as dimension,
                COUNT(*) as total_count,
                COUNT(DISTINCT {primary_field.name}) as unique_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as percentage
            FROM {primary_field.table}
            WHERE {primary_field.name} IS NOT NULL
            GROUP BY {primary_field.name}
        )
        SELECT 
            dimension,
            total_count,
            unique_count,
            percentage,
            CASE 
                WHEN total_count > (SELECT AVG(total_count) FROM analysis) THEN 'HIGH'
                WHEN total_count < (SELECT AVG(total_count) / 2 FROM analysis) THEN 'LOW'
                ELSE 'MEDIUM'
            END as volume_classification
        FROM analysis
        ORDER BY total_count DESC
        LIMIT 20
        """
        
        return sql
        
    def calculate_perfection_score(self, query: MinimalIntelligentQuery, fields: List[MinimalFieldIntelligence]) -> float:
        if not fields:
            return 0.0
            
        avg_intelligence = sum(f.intelligence_score for f in fields) / len(fields)
        coverage = len(fields) / 8.0  # Normalize to expected field count
        priority_weight = 1.0 if query.priority == 'Critical' else 0.8 if query.priority == 'High' else 0.6
        
        return min((avg_intelligence * 0.5 + coverage * 0.3 + priority_weight * 0.2), 1.0)
        
    def validate_query(self, sql: str) -> str:
        try:
            test_sql = f"SELECT COUNT(*) FROM ({sql}) LIMIT 1"
            self.connection.execute(test_sql)
            return "valid"
        except:
            return "invalid"
            
    def improve_iteration(self) -> bool:
        improved = False
        
        # Evolve low-performing fields
        low_performers = [f for f in self.field_intelligence.values() if f.intelligence_score < 0.7]
        for field in low_performers[:3]:  # Limit to avoid performance issues
            enhanced_samples = self.sample_field_data(field.table, field.name, 2000)
            if len(enhanced_samples) > len(field.sample_values):
                field.sample_values = enhanced_samples[:50]
                reanalysis = self.semantic_engine.analyze_field(field.name, field.sample_values)
                new_intelligence = self.calculate_intelligence_score(
                    max(reanalysis['semantic_scores'].values()) if reanalysis['semantic_scores'] else 0.0,
                    self.calculate_quality_score(reanalysis['quality_metrics']),
                    field.ao1_relevance
                )
                if new_intelligence > field.intelligence_score:
                    field.intelligence_score = new_intelligence
                    improved = True
                    
        return improved
        
    def calculate_perfection_score_overall(self) -> float:
        if not self.queries:
            return 0.0
            
        field_score = sum(f.intelligence_score for f in self.field_intelligence.values()) / len(self.field_intelligence) if self.field_intelligence else 0.0
        query_score = sum(q.perfection_score for q in self.queries) / len(self.queries)
        coverage_score = len(self.queries) / len(self.ao1_requirements)
        
        return (field_score * 0.4 + query_score * 0.4 + coverage_score * 0.2)
        
    def pursue_perfection(self):
        logger.info(f"Pursuing perfection (threshold: {self.perfection_threshold})")
        
        start_time = time.time()
        breakthrough_moments = []
        
        while self.iteration_count < self.max_iterations and self.perfection_score < self.perfection_threshold:
            improved = self.improve_iteration()
            new_perfection = self.calculate_perfection_score_overall()
            
            if new_perfection > self.perfection_score + 0.01:
                breakthrough = {
                    'iteration': self.iteration_count,
                    'old_score': self.perfection_score,
                    'new_score': new_perfection,
                    'timestamp': datetime.now().isoformat()
                }
                breakthrough_moments.append(breakthrough)
                logger.info(f"Breakthrough! Iteration {self.iteration_count}: {self.perfection_score:.4f} → {new_perfection:.4f}")
                
            self.perfection_score = new_perfection
            self.iteration_count += 1
            
            if self.iteration_count % 1000 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Progress: {self.iteration_count}/{self.max_iterations} | Perfection: {self.perfection_score:.4f}")
                
        total_time = time.time() - start_time
        
        if self.perfection_score >= self.perfection_threshold:
            logger.info(f"PERFECTION ACHIEVED! Score: {self.perfection_score:.4f} in {self.iteration_count} iterations ({total_time:.1f}s)")
        else:
            logger.info(f"Maximum iterations reached. Final score: {self.perfection_score:.4f} ({total_time:.1f}s)")
            
        return breakthrough_moments
        
    def run_analysis(self, save_results: bool = True) -> Dict:
        logger.info("Starting Minimal Intelligent AO1 Engine Analysis...")
        start_time = time.time()
        
        try:
            logger.info("Phase 1: Schema Discovery")
            self.connect_database()
            schema = self.discover_schema()
            
            if not schema:
                raise Exception("No schema discovered")
                
            logger.info("Phase 2: Field Analysis")
            field_count = 0
            for table, columns in schema.items():
                for column_name, data_type in columns:
                    field_intel = self.analyze_field_intelligent(table, column_name, data_type)
                    self.field_intelligence[f"{table}.{column_name}"] = field_intel
                    field_count += 1
                    
                    if field_count % 10 == 0:
                        logger.info(f"Analyzed {field_count} fields...")
                        
            logger.info("Phase 3: Query Generation")
            for requirement, req_data in self.ao1_requirements.items():
                query = self.generate_intelligent_query(requirement, req_data)
                if query:
                    self.queries.append(query)
                    
            logger.info("Phase 4: Perfection Pursuit")
            breakthrough_moments = self.pursue_perfection()
            
            results = {'analysis_completed': True}
            
            if save_results:
                logger.info("Saving results...")
                results['output_files'] = self.save_results(breakthrough_moments)
                
            total_time = time.time() - start_time
            
            logger.info(f"""
MINIMAL INTELLIGENT AO1 ANALYSIS COMPLETE!
┌─────────────────────────────────────────┐
│ Perfection Score: {self.perfection_score:.4f}                  │
│ Fields Analyzed: {len(self.field_intelligence)}                     │
│ Queries Generated: {len(self.queries)}                   │
│ Iterations: {self.iteration_count}                        │
│ Analysis Time: {total_time:.1f} seconds          │
│ Breakthroughs: {len(breakthrough_moments)}                    │
└─────────────────────────────────────────┘
            """)
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {'analysis_completed': False, 'error': str(e)}
            
    def save_results(self, breakthrough_moments: List[Dict]) -> List[str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'database_path': self.database_path,
                'perfection_score': self.perfection_score,
                'iterations': self.iteration_count
            },
            'field_intelligence': {
                key: {
                    'name': field.name, 'table': field.table, 'semantic_type': field.semantic_type,
                    'confidence': field.confidence, 'ao1_relevance': field.ao1_relevance,
                    'intelligence_score': field.intelligence_score, 'quality_score': field.quality_score,
                    'business_context': field.business_context
                }
                for key, field in self.field_intelligence.items()
            },
            'queries': [
                {
                    'name': query.name, 'description': query.description,
                    'sql': query.sql, 'priority': query.priority,
                    'perfection_score': query.perfection_score,
                    'validation_status': query.validation_status
                }
                for query in self.queries
            ],
            'breakthrough_moments': breakthrough_moments
        }
        
        results_file = f"minimal_ao1_analysis_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Save SQL queries
        sql_file = f"minimal_ao1_queries_{timestamp}.sql"
        with open(sql_file, 'w') as f:
            f.write(f"-- Minimal Intelligent AO1 Queries\n")
            f.write(f"-- Generated: {datetime.now().isoformat()}\n")
            f.write(f"-- Perfection Score: {self.perfection_score:.4f}\n\n")
            
            for query in self.queries:
                f.write(f"-- {query.name}: {query.description}\n")
                f.write(f"-- Priority: {query.priority} | Perfection: {query.perfection_score:.3f}\n")
                f.write(query.sql)
                f.write("\n\n" + "="*60 + "\n\n")
                
        logger.info(f"Saved results: {results_file}, {sql_file}")
        return [results_file, sql_file]

def main():
    parser = argparse.ArgumentParser(description="Minimal Intelligent AO1 Engine - Zero segfault, maximum intelligence")
    
    parser.add_argument('-d', '--database', required=True, help='Path to database file')
    parser.add_argument('-p', '--perfection-threshold', type=float, default=0.90, help='Perfection threshold (default: 0.90)')
    parser.add_argument('-m', '--max-iterations', type=int, default=10000, help='Maximum iterations (default: 10000)')
    parser.add_argument('-s', '--save-results', action='store_true', help='Save results to files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.database):
        print(f"Database file not found: {args.database}")
        sys.exit(1)
        
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        print("🧠 Initializing Minimal Intelligent AO1 Engine...")
        print("📦 Zero problematic dependencies - Maximum stability")
        
        engine = MinimalIntelligentAO1Engine(
            database_path=args.database,
            perfection_threshold=args.perfection_threshold,
            max_iterations=args.max_iterations
        )
        
        print(f"🎯 Target Perfection: {args.perfection_threshold}")
        print("🚀 Beginning analysis...\n")
        
        results = engine.run_analysis(save_results=args.save_results)
        
        if results.get('analysis_completed'):
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📊 Perfection Score: {engine.perfection_score:.4f}")
            print(f"🔍 Fields Analyzed: {len(engine.field_intelligence)}")
            print(f"✨ Queries Generated: {len(engine.queries)}")
            
            high_intelligence = len([f for f in engine.field_intelligence.values() if f.intelligence_score > 0.8])
            print(f"🧠 High Intelligence Fields: {high_intelligence}")
            
            if args.save_results and 'output_files' in results:
                print(f"\n📁 Output Files:")
                for file_path in results['output_files']:
                    print(f"   📄 {file_path}")
                    
        else:
            print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Analysis interrupted")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()