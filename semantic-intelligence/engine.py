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
            logger.info("Starting intelligent semantic analysis for AO1 visibility metrics")
            
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
            
            # Phase 3: Field Analysis (Enhanced for AO1)
            logger.info(f"Analyzing {results.total_fields} fields with AO1 focus")
            self._analyze_fields_for_ao1(schema, results)
            
            # Phase 4: Relationship Analysis
            logger.info("Analyzing field relationships")
            self._analyze_relationships(results)
            
            # Phase 5: Clustering
            logger.info("Creating semantic clusters")
            self._create_clusters(results)
            
            # Phase 6: Query Generation (Optional - AO1 handles this now)
            if results.high_intelligence_fields > 0:
                logger.info("Generating traditional queries")
                self._generate_queries(results)
            
            results.processing_time_seconds = time.time() - start_time
            results.success = True
            
            logger.info(f"Analysis completed successfully in {results.processing_time_seconds:.2f} seconds")
            logger.info(f"Ready for AO1 dashboard generation with {results.high_intelligence_fields} intelligent fields")
            
        except Exception as e:
            results.error_message = str(e)
            results.processing_time_seconds = time.time() - start_time
            logger.error(f"Analysis failed: {e}")
            
        finally:
            # Keep connection open for AO1 engine
            pass  # Don't disconnect here - AO1 engine needs it
            
        return results
        
    def _analyze_fields_for_ao1(self, schema: Dict[str, List], results: AnalysisResults):
        """Enhanced field analysis with AO1 visibility focus"""
        analyzed_count = 0
        high_intelligence_count = 0
        ao1_relevant_count = 0
        
        for table_name, columns in schema.items():
            for column_name, data_type in columns:
                try:
                    # Sample field data with larger sample for AO1
                    sample_values = self.database_connector.sample_field_data(
                        table_name, column_name, sample_size=3000  # Larger sample for AO1
                    )
                    
                    # Analyze field with AO1 enhancement
                    field_intelligence = self.semantic_analyzer.analyze_field(
                        column_name, table_name, data_type, sample_values
                    )
                    
                    # AO1-specific enhancement
                    self._enhance_field_for_ao1(field_intelligence)
                    
                    # Store results
                    field_key = field_intelligence.get_key()
                    self.field_intelligence[field_key] = field_intelligence
                    
                    analyzed_count += 1
                    
                    if field_intelligence.intelligence_score >= self.intelligence_threshold:
                        high_intelligence_count += 1
                        
                    # Track AO1-relevant fields
                    if self._is_ao1_relevant(field_intelligence):
                        ao1_relevant_count += 1
                        
                    if analyzed_count % 10 == 0:
                        logger.debug(f"Analyzed {analyzed_count}/{results.total_fields} fields")
                        
                except Exception as e:
                    logger.debug(f"Failed to analyze {table_name}.{column_name}: {e}")
                    
        results.analyzed_fields = analyzed_count
        results.high_intelligence_fields = high_intelligence_count
        
        logger.info(f"Found {ao1_relevant_count} AO1-relevant fields out of {high_intelligence_count} high-intelligence fields")
        
    def _enhance_field_for_ao1(self, field: FieldIntelligence):
        """Enhance field intelligence specifically for AO1 visibility metrics"""
        
        # Boost intelligence score for AO1-critical fields
        ao1_boost = 0.0
        field_name_lower = field.name.lower()
        
        # Asset identifier fields (critical for AO1)
        if any(keyword in field_name_lower for keyword in ['asset', 'host', 'hostname', 'device', 'ci_name', 'server']):
            ao1_boost += 0.15
            if not field.business_context:
                field.business_context = {}
            field.business_context['ao1_category'] = 'asset_identifier'
            
        # Logging activity fields (critical for AO1)
        elif any(keyword in field_name_lower for keyword in ['log', 'event', 'message', 'activity', 'count']):
            ao1_boost += 0.12
            if not field.business_context:
                field.business_context = {}
            field.business_context['ao1_category'] = 'logging_activity'
            
        # Platform/source fields (important for AO1)
        elif any(keyword in field_name_lower for keyword in ['source', 'platform', 'tool', 'index', 'sourcetype']):
            ao1_boost += 0.10
            if not field.business_context:
                field.business_context = {}
            field.business_context['ao1_category'] = 'platform_source'
            
        # Time fields (important for AO1)
        elif any(keyword in field_name_lower for keyword in ['time', 'date', 'timestamp', 'created', 'occurred']):
            ao1_boost += 0.08
            if not field.business_context:
                field.business_context = {}
            field.business_context['ao1_category'] = 'temporal'
            
        # Infrastructure fields (valuable for AO1)
        elif any(keyword in field_name_lower for keyword in ['environment', 'env', 'type', 'category', 'region']):
            ao1_boost += 0.06
            if not field.business_context:
                field.business_context = {}
            field.business_context['ao1_category'] = 'infrastructure'
        
        # Apply AO1 boost
        field.intelligence_score = min(1.0, field.intelligence_score + ao1_boost)
        
        # Check sample values for AO1 relevance
        if field.sample_values:
            self._analyze_sample_values_for_ao1(field)
            
    def _analyze_sample_values_for_ao1(self, field: FieldIntelligence):
        """Analyze sample values for AO1-specific patterns"""
        
        if not field.sample_values:
            return
            
        sample_text = ' '.join(str(v).lower() for v in field.sample_values[:20])
        
        # Platform detection
        platform_indicators = ['splunk', 'chronicle', 'crowdstrike', 'bigquery', 'theom', 'wiz', 'falcon']
        if any(platform in sample_text for platform in platform_indicators):
            if not field.business_context:
                field.business_context = {}
            field.business_context['contains_platform_names'] = True
            field.intelligence_score = min(1.0, field.intelligence_score + 0.05)
            
        # Infrastructure detection
        infra_indicators = ['cloud', 'aws', 'azure', 'gcp', 'onprem', 'datacenter', 'saas']
        if any(infra in sample_text for infra in infra_indicators):
            if not field.business_context:
                field.business_context = {}
            field.business_context['contains_infrastructure_terms'] = True
            field.intelligence_score = min(1.0, field.intelligence_score + 0.03)
            
    def _is_ao1_relevant(self, field: FieldIntelligence) -> bool:
        """Check if field is relevant for AO1 visibility metrics"""
        
        if not field.business_context:
            return False
            
        ao1_categories = ['asset_identifier', 'logging_activity', 'platform_source', 'temporal', 'infrastructure']
        return field.business_context.get('ao1_category') in ao1_categories
        
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
        
    def get_top_fields(self, limit: int = 20) -> List[FieldIntelligence]:
        """Get top intelligent fields, prioritizing AO1-relevant ones"""
        sorted_fields = sorted(
            self.field_intelligence.values(),
            key=lambda f: (
                1.0 if self._is_ao1_relevant(f) else 0.8,  # AO1 relevance boost
                f.intelligence_score
            ),
            reverse=True
        )
        return sorted_fields[:limit]
        
    def get_ao1_fields(self) -> Dict[str, List[FieldIntelligence]]:
        """Get fields categorized by AO1 relevance"""
        ao1_fields = {
            'asset_identifier': [],
            'logging_activity': [],
            'platform_source': [],
            'temporal': [],
            'infrastructure': []
        }
        
        for field in self.field_intelligence.values():
            if field.business_context and 'ao1_category' in field.business_context:
                category = field.business_context['ao1_category']
                if category in ao1_fields:
                    ao1_fields[category].append(field)
                    
        # Sort each category by intelligence score
        for category in ao1_fields:
            ao1_fields[category].sort(key=lambda f: f.intelligence_score, reverse=True)
            
        return ao1_fields
        
    def save_results(self, output_dir: str = ".") -> List[str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = []
        
        # Save field intelligence with AO1 enhancement
        field_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'database_path': self.database_path,
                'total_fields': len(self.field_intelligence),
                'high_intelligence_fields': len([f for f in self.field_intelligence.values() 
                                               if f.intelligence_score >= self.intelligence_threshold]),
                'ao1_relevant_fields': len([f for f in self.field_intelligence.values() 
                                          if self._is_ao1_relevant(f)])
            },
            'fields': {},
            'ao1_categories': {}
        }
        
        # Get AO1 categorized fields
        ao1_fields = self.get_ao1_fields()
        for category, fields in ao1_fields.items():
            field_data['ao1_categories'][category] = [f.get_key() for f in fields[:10]]  # Top 10 per category
        
        for key, field in self.field_intelligence.items():
            field_data['fields'][key] = {
                'name': field.name,
                'table': field.table,
                'data_type': field.data_type,
                'intelligence_score': field.intelligence_score,
                'confidence_level': field.confidence_level,
                'business_context': field.business_context,
                'semantic_density': field.semantic_profile.semantic_density if field.semantic_profile else 0.0,
                'complexity_score': field.semantic_profile.complexity_score if field.semantic_profile else 0.0,
                'ao1_relevant': self._is_ao1_relevant(field)
            }
            
        field_file = f"{output_dir}/field_intelligence_ao1_{timestamp}.json"
        with open(field_file, 'w') as f:
            json.dump(field_data, f, indent=2, default=str)
        output_files.append(field_file)
        
        # Save other results (relationships, clusters, queries) - existing code
        if self.relationships:
            relationship_file = f"{output_dir}/relationships_{timestamp}.json"
            with open(relationship_file, 'w') as f:
                json.dump(self.relationships, f, indent=2)
            output_files.append(relationship_file)
            
        if self.clusters:
            cluster_file = f"{output_dir}/clusters_{timestamp}.json"
            with open(cluster_file, 'w') as f:
                json.dump(self.clusters, f, indent=2)
            output_files.append(cluster_file)
            
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
        ao1_relevant = len([f for f in self.field_intelligence.values() if self._is_ao1_relevant(f)])
        
        return {
            'total_fields': total_fields,
            'high_intelligence_fields': len([f for f in self.field_intelligence.values() 
                                           if f.intelligence_score >= self.intelligence_threshold]),
            'ao1_relevant_fields': ao1_relevant,
            'avg_intelligence_score': sum(intelligence_scores) / len(intelligence_scores),
            'max_intelligence_score': max(intelligence_scores),
            'relationships_found': len(self.relationships),
            'clusters_created': len(self.clusters),
            'queries_generated': len(self.generated_queries),
            'top_domains': self._get_top_domains(),
            'table_coverage': len(set(f.table for f in self.field_intelligence.values())),
            'ao1_readiness': 'READY' if ao1_relevant >= 4 else 'LIMITED'
        }
        
    def _get_top_domains(self) -> Dict[str, int]:
        domain_counts = defaultdict(int)
        for field in self.field_intelligence.values():
            if field.business_context:
                domain = field.business_context.get('domain_classification', 'unknown')
                domain_counts[domain] += 1
        return dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10])