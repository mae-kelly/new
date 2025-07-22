#!/usr/bin/env python3

import statistics
from typing import Dict, List, Any
from models import FieldIntelligence, QueryResult
from query_templates import QueryTemplates

class QueryGenerator:
    def __init__(self):
        self.templates = {
            'asset_discovery': {
                'description': 'Comprehensive asset discovery and classification',
                'min_intelligence': 0.7,
                'relevant_domains': ['identity', 'network', 'system'],
                'generator': QueryTemplates.asset_discovery
            },
            'security_analysis': {
                'description': 'Security event correlation and threat detection',
                'min_intelligence': 0.8,
                'relevant_domains': ['security', 'temporal', 'network'],
                'generator': QueryTemplates.security_analysis
            },
            'user_behavior': {
                'description': 'User activity patterns and access analysis',
                'min_intelligence': 0.75,
                'relevant_domains': ['identity', 'temporal', 'business'],
                'generator': QueryTemplates.user_behavior
            },
            'network_topology': {
                'description': 'Network infrastructure mapping and analysis',
                'min_intelligence': 0.7,
                'relevant_domains': ['network', 'system', 'location'],
                'generator': QueryTemplates.network_topology
            },
            'compliance_monitoring': {
                'description': 'Compliance status tracking and audit trail',
                'min_intelligence': 0.65,
                'relevant_domains': ['business', 'temporal', 'status'],
                'generator': QueryTemplates.compliance_monitoring
            },
            'performance_analytics': {
                'description': 'System performance metrics and optimization',
                'min_intelligence': 0.7,
                'relevant_domains': ['system', 'temporal', 'status'],
                'generator': QueryTemplates.performance_analytics
            }
        }
        
    def generate_query(self, template_name: str, fields: List[FieldIntelligence], 
                      relationships: Dict = None) -> QueryResult:
        template = self.templates.get(template_name)
        if not template:
            return None
            
        eligible_fields = self._select_fields(fields, template)
        if not eligible_fields:
            return None
            
        sql = template['generator'](eligible_fields, relationships or {})
        
        query = QueryResult(
            name=f"INTELLIGENT_{template_name.upper()}",
            description=template['description'],
            sql=sql,
            field_count=len(eligible_fields),
            tables_used=list(set(f.table for f in eligible_fields))
        )
        
        query.intelligence_score = self._calculate_query_intelligence(query, eligible_fields)
        query.complexity_rating = self._rate_complexity(sql)
        query.estimated_performance = self._estimate_performance(query)
        
        return query
        
    def _select_fields(self, fields: List[FieldIntelligence], template: Dict) -> List[FieldIntelligence]:
        min_intelligence = template['min_intelligence']
        relevant_domains = template['relevant_domains']
        
        candidates = []
        
        for field in fields:
            if field.intelligence_score >= min_intelligence * 0.9:
                relevance = self._calculate_domain_relevance(field, relevant_domains)
                total_score = (field.intelligence_score * 0.6 + relevance * 0.4)
                
                if total_score >= min_intelligence:
                    candidates.append((field, total_score))
                    
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [field for field, score in candidates[:10]]
        
    def _calculate_domain_relevance(self, field: FieldIntelligence, relevant_domains: List[str]) -> float:
        if not field.semantic_profile or not field.semantic_profile.pattern_features:
            return 0.0
            
        semantic_patterns = field.semantic_profile.pattern_features.get('semantic_patterns', [])
        if not semantic_patterns:
            return 0.0
            
        # Map domain names to semantic pattern indices
        domain_mapping = {
            'identity': 0, 'network': 1, 'security': 2, 'temporal': 3,
            'business': 4, 'system': 5, 'location': 6, 'status': 7
        }
        
        relevance_scores = []
        for domain in relevant_domains:
            if domain in domain_mapping:
                idx = domain_mapping[domain]
                if idx < len(semantic_patterns):
                    relevance_scores.append(semantic_patterns[idx])
                    
        return statistics.mean(relevance_scores) if relevance_scores else 0.0
        
    def _calculate_query_intelligence(self, query: QueryResult, fields: List[FieldIntelligence]) -> float:
        if not fields:
            return 0.0
            
        # Average field intelligence
        avg_field_intelligence = statistics.mean([f.intelligence_score for f in fields])
        
        # Table diversity factor
        table_diversity = len(query.tables_used) / max(len(fields), 1)
        
        # Complexity factor
        complexity_factor = min(1.0, query.complexity_rating / 10.0)
        
        return (avg_field_intelligence * 0.5 + table_diversity * 0.3 + complexity_factor * 0.2)
        
    def _rate_complexity(self, sql: str) -> int:
        complexity_indicators = [
            'WITH', 'CASE', 'WHEN', 'OVER', 'PARTITION', 'WINDOW',
            'PERCENTILE_CONT', 'ROW_NUMBER', 'RANK', 'NTILE',
            'GROUP BY', 'ORDER BY', 'HAVING', 'UNION', 'JOIN'
        ]
        
        sql_upper = sql.upper()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in sql_upper)
        
        # Count subqueries and CTEs
        subquery_count = sql_upper.count('SELECT') - 1
        cte_count = sql_upper.count('WITH')
        
        total_complexity = complexity_score + subquery_count + (cte_count * 2)
        return min(10, max(1, total_complexity))
        
    def _estimate_performance(self, query: QueryResult) -> str:
        # Simple heuristics for performance estimation
        if query.complexity_rating <= 3 and query.field_count <= 2:
            return "FAST"
        elif query.complexity_rating <= 6 and query.field_count <= 5:
            return "MODERATE"
        elif query.complexity_rating <= 8:
            return "SLOW"
        else:
            return "VERY_SLOW"
            
    def generate_all_queries(self, fields: List[FieldIntelligence], 
                           relationships: Dict = None) -> List[QueryResult]:
        queries = []
        
        for template_name in self.templates.keys():
            query = self.generate_query(template_name, fields, relationships)
            if query and query.intelligence_score > 0.5:
                queries.append(query)
                
        queries.sort(key=lambda q: q.intelligence_score, reverse=True)
        return queries