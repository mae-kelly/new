#!/usr/bin/env python3

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

from models import FieldIntelligence, QueryResult, AnalysisResults
from database_connector import DatabaseConnector
from semantic_analyzer import SemanticAnalyzer
from relationship_engine import RelationshipEngine
from query_generator import QueryGenerator

logger = logging.getLogger(__name__)

class IntelligentSemanticEngine:
    def __init__(self, database_path: str, intelligence_threshold: float = 0.8):
        self.database_path = database_path
        self.intelligence_threshold = intelligence_threshold
        
        self.database_connector = DatabaseConnector(database_path)
        self.semantic_analyzer = SemanticAnalyzer()
        self.relationship_engine = RelationshipEngine()
        self.query_generator = QueryGenerator()
        
        self.field_intelligence: Dict[str, FieldIntelligence] = {}
        self.relationships: Dict[str, Dict[str, float]] = {}
        self.clusters: Dict[str, List[str]] = {}
        self.generated_queries: List[QueryResult] = []
        
    def run_analysis(self) -> AnalysisResults:
        start_time = time.time()
        results = AnalysisResults()
        
        try:
            logger.info("Starting intelligent semantic analysis")
            
            # Phase 1: Database Connection
            if not self.database_connector.connect():
                results.error_message = "Failed to connect to database"
                return results
                
            # Phase 2: Schema Discovery
            logger.info("Discovering database schema")
            schema = self.database_connector.discover_schema()
            if not schema:
                results.error_message = "No schema discovered"
                return results
                
            results.total_fields = sum(len(columns) for columns in schema.values())
            
            # Phase 3: Field Analysis
            logger.info(f"Analyzing {results.total_fields} fields")
            self._analyze_fields(schema, results)
            
            # Phase 4: Relationship Analysis
            logger.info("Analyzing field relationships")
            self._analyze_relationships(results)
            
            # Phase 5: Clustering
            logger.info("Creating semantic clusters")
            self._create_clusters(results)
            
            # Phase 6: Query Generation
            logger.info("Generating intelligent queries")
            self._generate_queries(results)
            
            results.processing_time_seconds = time.time() - start_time
            results.success = True
            
            logger.info(f"Analysis completed successfully in {results.processing_time_seconds:.2f} seconds")
            
        except Exception as e:
            results.error_message = str(e)
            results.processing_time_seconds = time.time() - start_time
            logger.error(f"Analysis failed: {e}")
            
        finally:
            self.database_connector.disconnect()
            
        return results
        
    def _analyze_fields(self, schema: Dict[str, List], results: AnalysisResults):
        analyzed_count = 0
        high_intelligence_count = 0
        
        for table_name, columns in schema.items():
            for column_name, data_type in columns:
                try:
                    # Sample field data
                    sample_values = self.database_connector.sample_field_data(
                        table_name, column_name, sample_size=2000
                    )
                    
                    # Analyze field
                    field_intelligence = self.semantic_analyzer.analyze_field(
                        column_name, table_name, data_type, sample_values
                    )
                    
                    # Store results
                    field_key = field_intelligence.get_key()
                    self.field_intelligence[field_key] = field_intelligence
                    
                    analyzed_count += 1
                    
                    if field_intelligence.intelligence_score >= self.intelligence_threshold:
                        high_intelligence_count += 1
                        
                    if analyzed_count % 10 == 0:
                        logger.info(f"Analyzed {analyzed_count}/{results.total_fields} fields")
                        
                except Exception as e:
                    logger.debug(f"Failed to analyze {table_name}.{column_name}: {e}")
                    
        results.analyzed_fields = analyzed_count
        results.high_intelligence_fields = high_intelligence_count
        
    def _analyze_relationships(self, results: AnalysisResults):
        high_intelligence_fields = [
            field for field in self.field_intelligence.values()
            if field.intelligence_score >= self.intelligence_threshold * 0.8
        ]
        
        if len(high_intelligence_fields) < 2:
            logger.warning("Not enough high-intelligence fields for relationship analysis")
            return
            
        self.relationships = self.relationship_engine.analyze_relationships(high_intelligence_fields)
        
        # Count meaningful relationships
        relationship_count = 0
        for field_relationships in self.relationships.values():
            relationship_count += len([r for r in field_relationships.values() if r > 0.5])
            
        results.relationships_found = relationship_count // 2  # Each relationship counted twice
        
    def _create_clusters(self, results: AnalysisResults):
        high_intelligence_fields = [
            field for field in self.field_intelligence.values()
            if field.intelligence_score >= self.intelligence_threshold * 0.8
        ]
        
        if len(high_intelligence_fields) < 3:
            logger.warning("Not enough fields for clustering")
            return
            
        self.clusters = self.relationship_engine.create_clusters(
            high_intelligence_fields, threshold=0.7
        )
        
        results.clusters_created = len(self.clusters)
        
    def _generate_queries(self, results: AnalysisResults):
        high_intelligence_fields = [
            field for field in self.field_intelligence.values()
            if field.intelligence_score >= self.intelligence_threshold * 0.9
        ]
        
        if not high_intelligence_fields:
            logger.warning("No high-intelligence fields available for query generation")
            return
            
        self.generated_queries = self.query_generator.generate_all_queries(
            high_intelligence_fields, self.relationships
        )
        
        results.queries_generated = len(self.generated_queries)
        
    def save_results(self, output_dir: str = ".") -> List[str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = []
        
        # Save field intelligence
        field_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'database_path': self.database_path,
                'total_fields': len(self.field_intelligence),
                'high_intelligence_fields': len([f for f in self.field_intelligence.values() 
                                               if f.intelligence_score >= self.intelligence_threshold])
            },
            'fields': {}
        }
        
        for key, field in self.field_intelligence.items():
            field_data['fields'][key] = {
                'name': field.name,
                'table': field.table,
                'data_type': field.data_type,
                'intelligence_score': field.intelligence_score,
                'confidence_level': field.confidence_level,
                'business_context': field.business_context,
                'semantic_density': field.semantic_profile.semantic_density if field.semantic_profile else 0.0,
                'complexity_score': field.semantic_profile.complexity_score if field.semantic_profile else 0.0
            }
            
        field_file = f"{output_dir}/field_intelligence_{timestamp}.json"
        with open(field_file, 'w') as f:
            json.dump(field_data, f, indent=2, default=str)
        output_files.append(field_file)
        
        # Save relationships
        if self.relationships:
            relationship_file = f"{output_dir}/relationships_{timestamp}.json"
            with open(relationship_file, 'w') as f:
                json.dump(self.relationships, f, indent=2)
            output_files.append(relationship_file)
            
        # Save clusters
        if self.clusters:
            cluster_file = f"{output_dir}/clusters_{timestamp}.json"
            with open(cluster_file, 'w') as f:
                json.dump(self.clusters, f, indent=2)
            output_files.append(cluster_file)
            
        # Save queries
        if self.generated_queries:
            query_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_queries': len(self.generated_queries)
                },
                'queries': []
            }
            
            for query in self.generated_queries:
                query_data['queries'].append({
                    'name': query.name,
                    'description': query.description,
                    'intelligence_score': query.intelligence_score,
                    'complexity_rating': query.complexity_rating,
                    'estimated_performance': query.estimated_performance,
                    'field_count': query.field_count,
                    'tables_used': query.tables_used,
                    'sql': query.sql
                })
                
            query_file = f"{output_dir}/queries_{timestamp}.json"
            with open(query_file, 'w') as f:
                json.dump(query_data, f, indent=2)
            output_files.append(query_file)
            
            # Save SQL files
            sql_file = f"{output_dir}/queries_{timestamp}.sql"
            with open(sql_file, 'w') as f:
                f.write(f"-- Intelligent Semantic Queries\n")
                f.write(f"-- Generated: {datetime.now().isoformat()}\n")
                f.write(f"-- Database: {self.database_path}\n\n")
                
                for query in self.generated_queries:
                    f.write(f"-- {query.name}: {query.description}\n")
                    f.write(f"-- Intelligence Score: {query.intelligence_score:.3f}\n")
                    f.write(f"-- Complexity: {query.complexity_rating}/10\n")
                    f.write(f"-- Performance: {query.estimated_performance}\n")
                    f.write(query.sql)
                    f.write("\n\n" + "="*80 + "\n\n")
                    
            output_files.append(sql_file)
            
        return output_files
        
    def get_top_fields(self, limit: int = 20) -> List[FieldIntelligence]:
        sorted_fields = sorted(
            self.field_intelligence.values(),
            key=lambda f: f.intelligence_score,
            reverse=True
        )
        return sorted_fields[:limit]
        
    def get_field_by_domain(self, domain: str) -> List[FieldIntelligence]:
        domain_fields = []
        for field in self.field_intelligence.values():
            if (field.business_context and 
                field.business_context.get('domain_classification') == domain):
                domain_fields.append(field)
                
        return sorted(domain_fields, key=lambda f: f.intelligence_score, reverse=True)
        
    def get_analysis_summary(self) -> Dict[str, Any]:
        total_fields = len(self.field_intelligence)
        if total_fields == 0:
            return {'error': 'No fields analyzed'}
            
        intelligence_scores = [f.intelligence_score for f in self.field_intelligence.values()]
        
        return {
            'total_fields': total_fields,
            'high_intelligence_fields': len([f for f in self.field_intelligence.values() 
                                           if f.intelligence_score >= self.intelligence_threshold]),
            'avg_intelligence_score': sum(intelligence_scores) / len(intelligence_scores),
            'max_intelligence_score': max(intelligence_scores),
            'relationships_found': len(self.relationships),
            'clusters_created': len(self.clusters),
            'queries_generated': len(self.generated_queries),
            'top_domains': self._get_top_domains(),
            'table_coverage': len(set(f.table for f in self.field_intelligence.values()))
        }
        
    def _get_top_domains(self) -> Dict[str, int]:
        domain_counts = defaultdict(int)
        for field in self.field_intelligence.values():
            if field.business_context:
                domain = field.business_context.get('domain_classification', 'unknown')
                domain_counts[domain] += 1
        return dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10])