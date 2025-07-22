import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .config import QUERY_TEMPLATES, PROJECT_ID
from .semantic_analyzer import FieldAnalysis

logger = logging.getLogger(__name__)

@dataclass
class GeneratedQuery:
    purpose: str
    query: str
    source_tables: List[str]
    key_fields: List[str]
    expected_result_type: str
    confidence_score: float

class QueryGenerator:
    def __init__(self):
        self.project_id = PROJECT_ID
        self.templates = QUERY_TEMPLATES
        
    def generate_ao1_queries(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> List[GeneratedQuery]:
        queries = []
        
        asset_queries = self._generate_asset_coverage_queries(semantic_results)
        queries.extend(asset_queries)
        
        tool_queries = self._generate_tool_coverage_queries(semantic_results)
        queries.extend(tool_queries)
        
        log_queries = self._generate_log_source_queries(semantic_results)
        queries.extend(log_queries)
        
        geographic_queries = self._generate_geographic_queries(semantic_results)
        queries.extend(geographic_queries)
        
        relationship_queries = self._generate_relationship_queries(semantic_results)
        queries.extend(relationship_queries)
        
        infrastructure_queries = self._generate_infrastructure_type_queries(semantic_results)
        queries.extend(infrastructure_queries)
        
        system_classification_queries = self._generate_system_classification_queries(semantic_results)
        queries.extend(system_classification_queries)
        
        url_fqdn_queries = self._generate_url_fqdn_coverage_queries(semantic_results)
        queries.extend(url_fqdn_queries)
        
        ipam_vpc_queries = self._generate_ipam_vpc_queries(semantic_results)
        queries.extend(ipam_vpc_queries)
        
        network_zone_queries = self._generate_network_zone_queries(semantic_results)
        queries.extend(network_zone_queries)
        
        log_volume_queries = self._generate_log_volume_queries(semantic_results)
        queries.extend(log_volume_queries)
        
        return queries
    
    def _generate_infrastructure_type_queries(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> List[GeneratedQuery]:
        queries = []
        infra_tables = self._find_tables_with_category(semantic_results, 'infrastructure_type')
        asset_tables = self._find_tables_with_category(semantic_results, 'asset_identity')
        
        for table_name, infra_fields in infra_tables.items():
            infra_field = self._find_best_field(infra_fields, 'infrastructure_type')
            asset_field = self._find_best_field(infra_fields, ['asset_identity', 'network_identity'])
            
            if not infra_field:
                continue
                
            if asset_field:
                query = f"""
                    WITH infrastructure_coverage AS (
                        SELECT 
                            `{infra_field.field_name}` as infrastructure_type,
                            COUNT(DISTINCT COALESCE(LOWER(TRIM(`{asset_field.field_name}`)), '')) as unique_assets,
                            COUNT(*) as total_records
                        FROM `{self.project_id}.{table_name}`
                        WHERE `{infra_field.field_name}` IS NOT NULL
                        AND `{asset_field.field_name}` IS NOT NULL
                        GROUP BY `{infra_field.field_name}`
                    )
                    SELECT 
                        infrastructure_type,
                        unique_assets,
                        total_records,
                        ROUND(unique_assets * 100.0 / SUM(unique_assets) OVER(), 2) as percentage_of_assets
                    FROM infrastructure_coverage
                    ORDER BY unique_assets DESC
                """
                key_fields = [infra_field.field_name, asset_field.field_name]
                confidence = (infra_field.confidence_score + asset_field.confidence_score) / 2
            else:
                query = f"""
                    SELECT 
                        `{infra_field.field_name}` as infrastructure_type,
                        COUNT(*) as record_count,
                        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                    FROM `{self.project_id}.{table_name}`
                    WHERE `{infra_field.field_name}` IS NOT NULL
                    GROUP BY `{infra_field.field_name}`
                    ORDER BY record_count DESC
                """
                key_fields = [infra_field.field_name]
                confidence = infra_field.confidence_score
            
            queries.append(GeneratedQuery(
                purpose=f"Infrastructure Type Distribution - {table_name}",
                query=query.strip(),
                source_tables=[table_name],
                key_fields=key_fields,
                expected_result_type="infrastructure_coverage",
                confidence_score=confidence
            ))
        
        return queries
    
    def _generate_system_classification_queries(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> List[GeneratedQuery]:
        queries = []
        system_tables = self._find_tables_with_category(semantic_results, 'system_classification')
        
        for table_name, system_fields in system_tables.items():
            system_field = self._find_best_field(system_fields, 'system_classification')
            asset_field = self._find_best_field(system_fields, ['asset_identity', 'network_identity'])
            
            if not system_field:
                continue
            
            if asset_field:
                query = f"""
                    WITH system_classification_coverage AS (
                        SELECT 
                            `{system_field.field_name}` as system_type,
                            COUNT(DISTINCT COALESCE(LOWER(TRIM(`{asset_field.field_name}`)), '')) as classified_assets,
                            COUNT(*) as total_entries
                        FROM `{self.project_id}.{table_name}`
                        WHERE `{system_field.field_name}` IS NOT NULL
                        AND `{asset_field.field_name}` IS NOT NULL
                        GROUP BY `{system_field.field_name}`
                    )
                    SELECT 
                        system_type,
                        classified_assets,
                        total_entries,
                        ROUND(classified_assets * 100.0 / SUM(classified_assets) OVER(), 2) as classification_percentage
                    FROM system_classification_coverage
                    ORDER BY classified_assets DESC
                """
                key_fields = [system_field.field_name, asset_field.field_name]
                confidence = (system_field.confidence_score + asset_field.confidence_score) / 2
            else:
                query = f"""
                    SELECT 
                        `{system_field.field_name}` as system_type,
                        COUNT(*) as classification_count
                    FROM `{self.project_id}.{table_name}`
                    WHERE `{system_field.field_name}` IS NOT NULL
                    GROUP BY `{system_field.field_name}`
                    ORDER BY classification_count DESC
                """
                key_fields = [system_field.field_name]
                confidence = system_field.confidence_score
            
            queries.append(GeneratedQuery(
                purpose=f"System Classification Analysis - {table_name}",
                query=query.strip(),
                source_tables=[table_name],
                key_fields=key_fields,
                expected_result_type="system_classification",
                confidence_score=confidence
            ))
        
        return queries
    
    def _generate_url_fqdn_coverage_queries(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> List[GeneratedQuery]:
        queries = []
        
        for table_name, fields in semantic_results.items():
            url_fields = [f for f in fields if 'url' in f.field_name.lower() or 'fqdn' in f.field_name.lower()]
            asset_field = self._find_best_field(fields, ['asset_identity', 'network_identity'])
            
            for url_field in url_fields:
                if url_field.confidence_score < 0.6:
                    continue
                
                if asset_field:
                    query = f"""
                        WITH url_fqdn_coverage AS (
                            SELECT 
                                CASE 
                                    WHEN `{url_field.field_name}` IS NOT NULL AND TRIM(`{url_field.field_name}`) != '' THEN 'HAS_URL_FQDN'
                                    ELSE 'NO_URL_FQDN'
                                END as url_fqdn_status,
                                COUNT(DISTINCT COALESCE(LOWER(TRIM(`{asset_field.field_name}`)), '')) as asset_count
                            FROM `{self.project_id}.{table_name}`
                            WHERE `{asset_field.field_name}` IS NOT NULL
                            GROUP BY url_fqdn_status
                        )
                        SELECT 
                            url_fqdn_status,
                            asset_count,
                            ROUND(asset_count * 100.0 / SUM(asset_count) OVER(), 2) as coverage_percentage
                        FROM url_fqdn_coverage
                        ORDER BY asset_count DESC
                    """
                    key_fields = [url_field.field_name, asset_field.field_name]
                    confidence = (url_field.confidence_score + asset_field.confidence_score) / 2
                else:
                    query = f"""
                        SELECT 
                            CASE 
                                WHEN `{url_field.field_name}` IS NOT NULL AND TRIM(`{url_field.field_name}`) != '' THEN 'HAS_URL_FQDN'
                                ELSE 'NO_URL_FQDN'
                            END as url_fqdn_status,
                            COUNT(*) as record_count
                        FROM `{self.project_id}.{table_name}`
                        GROUP BY url_fqdn_status
                        ORDER BY record_count DESC
                    """
                    key_fields = [url_field.field_name]
                    confidence = url_field.confidence_score
                
                queries.append(GeneratedQuery(
                    purpose=f"URL/FQDN Coverage Analysis - {table_name}.{url_field.field_name}",
                    query=query.strip(),
                    source_tables=[table_name],
                    key_fields=key_fields,
                    expected_result_type="url_fqdn_coverage",
                    confidence_score=confidence
                ))
        
        return queries
    
    def _generate_ipam_vpc_queries(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> List[GeneratedQuery]:
        queries = []
        
        for table_name, fields in semantic_results.items():
            ipam_fields = [f for f in fields if 'ipam' in f.field_name.lower() or 'vpc' in f.field_name.lower()]
            ip_fields = [f for f in fields if f.ao1_category == 'network_identity' and 'ip' in f.field_name.lower()]
            
            for ipam_field in ipam_fields:
                for ip_field in ip_fields:
                    if ipam_field.confidence_score < 0.6 or ip_field.confidence_score < 0.6:
                        continue
                    
                    query = f"""
                        WITH ipam_vpc_coverage AS (
                            SELECT 
                                `{ipam_field.field_name}` as ipam_vpc_identifier,
                                COUNT(DISTINCT `{ip_field.field_name}`) as unique_ips,
                                COUNT(*) as total_records
                            FROM `{self.project_id}.{table_name}`
                            WHERE `{ipam_field.field_name}` IS NOT NULL
                            AND `{ip_field.field_name}` IS NOT NULL
                            GROUP BY `{ipam_field.field_name}`
                        )
                        SELECT 
                            ipam_vpc_identifier,
                            unique_ips,
                            total_records,
                            ROUND(unique_ips * 100.0 / SUM(unique_ips) OVER(), 2) as ip_distribution_percentage
                        FROM ipam_vpc_coverage
                        ORDER BY unique_ips DESC
                    """
                    
                    queries.append(GeneratedQuery(
                        purpose=f"iPAM/VPC Coverage Analysis - {table_name}",
                        query=query.strip(),
                        source_tables=[table_name],
                        key_fields=[ipam_field.field_name, ip_field.field_name],
                        expected_result_type="ipam_vpc_coverage",
                        confidence_score=(ipam_field.confidence_score + ip_field.confidence_score) / 2
                    ))
        
        return queries
    
    def _generate_network_zone_queries(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> List[GeneratedQuery]:
        queries = []
        
        for table_name, fields in semantic_results.items():
            zone_fields = [f for f in fields if 'zone' in f.field_name.lower() or 'network' in f.field_name.lower()]
            asset_field = self._find_best_field(fields, ['asset_identity', 'network_identity'])
            
            for zone_field in zone_fields:
                if zone_field.confidence_score < 0.6:
                    continue
                
                if asset_field:
                    query = f"""
                        WITH network_zone_coverage AS (
                            SELECT 
                                `{zone_field.field_name}` as network_zone,
                                COUNT(DISTINCT COALESCE(LOWER(TRIM(`{asset_field.field_name}`)), '')) as assets_in_zone,
                                COUNT(*) as total_zone_records
                            FROM `{self.project_id}.{table_name}`
                            WHERE `{zone_field.field_name}` IS NOT NULL
                            AND `{asset_field.field_name}` IS NOT NULL
                            GROUP BY `{zone_field.field_name}`
                        )
                        SELECT 
                            network_zone,
                            assets_in_zone,
                            total_zone_records,
                            ROUND(assets_in_zone * 100.0 / SUM(assets_in_zone) OVER(), 2) as zone_coverage_percentage
                        FROM network_zone_coverage
                        ORDER BY assets_in_zone DESC
                    """
                    key_fields = [zone_field.field_name, asset_field.field_name]
                    confidence = (zone_field.confidence_score + asset_field.confidence_score) / 2
                else:
                    query = f"""
                        SELECT 
                            `{zone_field.field_name}` as network_zone,
                            COUNT(*) as zone_record_count
                        FROM `{self.project_id}.{table_name}`
                        WHERE `{zone_field.field_name}` IS NOT NULL
                        GROUP BY `{zone_field.field_name}`
                        ORDER BY zone_record_count DESC
                    """
                    key_fields = [zone_field.field_name]
                    confidence = zone_field.confidence_score
                
                queries.append(GeneratedQuery(
                    purpose=f"Network Zone Coverage - {table_name}.{zone_field.field_name}",
                    query=query.strip(),
                    source_tables=[table_name],
                    key_fields=key_fields,
                    expected_result_type="network_zone_coverage",
                    confidence_score=confidence
                ))
        
        return queries
    
    def _generate_log_volume_queries(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> List[GeneratedQuery]:
        queries = []
        
        for table_name, fields in semantic_results.items():
            volume_fields = [f for f in fields if 'volume' in f.field_name.lower() or 'ingest' in f.field_name.lower() or 'count' in f.field_name.lower()]
            log_field = self._find_best_field(fields, 'log_sources')
            asset_field = self._find_best_field(fields, ['asset_identity', 'network_identity'])
            
            for volume_field in volume_fields:
                if volume_field.confidence_score < 0.6:
                    continue
                
                if log_field and asset_field:
                    query = f"""
                        WITH log_volume_analysis AS (
                            SELECT 
                                `{log_field.field_name}` as log_type,
                                COALESCE(LOWER(TRIM(`{asset_field.field_name}`)), 'unknown') as asset_identifier,
                                SUM(CAST(`{volume_field.field_name}` AS NUMERIC)) as total_log_volume,
                                COUNT(*) as log_entries
                            FROM `{self.project_id}.{table_name}`
                            WHERE `{volume_field.field_name}` IS NOT NULL
                            AND `{log_field.field_name}` IS NOT NULL
                            GROUP BY `{log_field.field_name}`, asset_identifier
                        )
                        SELECT 
                            log_type,
                            COUNT(DISTINCT asset_identifier) as assets_with_logs,
                            SUM(total_log_volume) as total_volume,
                            ROUND(AVG(total_log_volume), 2) as avg_volume_per_asset,
                            SUM(log_entries) as total_entries
                        FROM log_volume_analysis
                        WHERE asset_identifier != 'unknown'
                        GROUP BY log_type
                        ORDER BY total_volume DESC
                    """
                    key_fields = [volume_field.field_name, log_field.field_name, asset_field.field_name]
                    confidence = (volume_field.confidence_score + log_field.confidence_score + asset_field.confidence_score) / 3
                else:
                    query = f"""
                        SELECT 
                            SUM(CAST(`{volume_field.field_name}` AS NUMERIC)) as total_volume,
                            COUNT(*) as total_records,
                            ROUND(AVG(CAST(`{volume_field.field_name}` AS NUMERIC)), 2) as average_volume
                        FROM `{self.project_id}.{table_name}`
                        WHERE `{volume_field.field_name}` IS NOT NULL
                    """
                    key_fields = [volume_field.field_name]
                    confidence = volume_field.confidence_score
                
                queries.append(GeneratedQuery(
                    purpose=f"Log Volume Analysis - {table_name}.{volume_field.field_name}",
                    query=query.strip(),
                    source_tables=[table_name],
                    key_fields=key_fields,
                    expected_result_type="log_volume_analysis",
                    confidence_score=confidence
                ))
        
        return queries
    
    def _generate_asset_coverage_queries(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> List[GeneratedQuery]:
        queries = []
        asset_tables = self._find_tables_with_category(semantic_results, 'asset_identity')
        
        for table_name, asset_fields in asset_tables.items():
            primary_field = self._find_best_field(asset_fields, 'asset_identity')
            if not primary_field:
                continue
            
            fallback_fields = [f for f in asset_fields if f.ao1_category in ['network_identity', 'asset_identity'] and f != primary_field]
            fallback_field = fallback_fields[0] if fallback_fields else primary_field
            
            query = f"""
                WITH asset_baseline AS (
                    SELECT DISTINCT 
                        COALESCE(
                            NULLIF(LOWER(TRIM(`{primary_field.field_name}`)), ''),
                            NULLIF(LOWER(TRIM(`{fallback_field.field_name}`)), '')
                        ) as asset_identifier,
                        `{primary_field.field_name}` as primary_asset_id,
                        `{fallback_field.field_name}` as fallback_asset_id
                    FROM `{self.project_id}.{table_name}`
                    WHERE COALESCE(`{primary_field.field_name}`, `{fallback_field.field_name}`) IS NOT NULL
                    AND COALESCE(`{primary_field.field_name}`, `{fallback_field.field_name}`) != ''
                )
                SELECT 
                    COUNT(DISTINCT asset_identifier) as total_unique_assets,
                    COUNT(DISTINCT primary_asset_id) as assets_with_primary_id,
                    COUNT(DISTINCT fallback_asset_id) as assets_with_fallback_id,
                    ROUND(COUNT(DISTINCT primary_asset_id) * 100.0 / COUNT(DISTINCT asset_identifier), 2) as primary_id_coverage_pct
                FROM asset_baseline
                WHERE asset_identifier IS NOT NULL
            """
            
            queries.append(GeneratedQuery(
                purpose=f"Asset Coverage Baseline - {table_name}",
                query=query.strip(),
                source_tables=[table_name],
                key_fields=[primary_field.field_name, fallback_field.field_name],
                expected_result_type="coverage_metrics",
                confidence_score=primary_field.confidence_score
            ))
        
        return queries
    
    def _generate_tool_coverage_queries(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> List[GeneratedQuery]:
        queries = []
        
        tool_tables = self._find_tables_with_category(semantic_results, 'security_tools')
        asset_tables = self._find_tables_with_category(semantic_results, 'asset_identity')
        
        if not asset_tables:
            return queries
        
        baseline_table = max(asset_tables.keys(), key=lambda t: len(asset_tables[t]))
        baseline_fields = asset_tables[baseline_table]
        baseline_asset_field = self._find_best_field(baseline_fields, 'asset_identity')
        
        if not baseline_asset_field:
            return queries
        
        for tool_table, tool_fields in tool_tables.items():
            tool_asset_field = self._find_best_field(tool_fields, ['asset_identity', 'network_identity'])
            if not tool_asset_field:
                continue
            
            tool_name = self._extract_tool_name(tool_table, tool_fields)
            
            query = f"""
                WITH baseline_assets AS (
                    SELECT DISTINCT LOWER(TRIM(`{baseline_asset_field.field_name}`)) as asset_identifier
                    FROM `{self.project_id}.{baseline_table}`
                    WHERE `{baseline_asset_field.field_name}` IS NOT NULL 
                    AND TRIM(`{baseline_asset_field.field_name}`) != ''
                ),
                tool_assets AS (
                    SELECT DISTINCT LOWER(TRIM(`{tool_asset_field.field_name}`)) as asset_identifier
                    FROM `{self.project_id}.{tool_table}`
                    WHERE `{tool_asset_field.field_name}` IS NOT NULL 
                    AND TRIM(`{tool_asset_field.field_name}`) != ''
                )
                SELECT 
                    '{tool_name}' as tool_name,
                    COUNT(DISTINCT ba.asset_identifier) as total_baseline_assets,
                    COUNT(DISTINCT ta.asset_identifier) as tool_covered_assets,
                    ROUND(COUNT(DISTINCT ta.asset_identifier) * 100.0 / COUNT(DISTINCT ba.asset_identifier), 2) as coverage_percentage,
                    COUNT(DISTINCT ba.asset_identifier) - COUNT(DISTINCT ta.asset_identifier) as coverage_gap
                FROM baseline_assets ba
                LEFT JOIN tool_assets ta ON ba.asset_identifier = ta.asset_identifier
            """
            
            queries.append(GeneratedQuery(
                purpose=f"Tool Coverage Analysis - {tool_name}",
                query=query.strip(),
                source_tables=[baseline_table, tool_table],
                key_fields=[baseline_asset_field.field_name, tool_asset_field.field_name],
                expected_result_type="tool_coverage",
                confidence_score=min(baseline_asset_field.confidence_score, tool_asset_field.confidence_score)
            ))
        
        return queries
    
    def _generate_log_source_queries(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> List[GeneratedQuery]:
        queries = []
        log_tables = self._find_tables_with_category(semantic_results, 'log_sources')
        
        for table_name, log_fields in log_tables.items():
            log_type_field = self._find_best_field(log_fields, 'log_sources')
            asset_field = self._find_best_field(log_fields, ['asset_identity', 'network_identity'])
            
            if not log_type_field:
                continue
            
            if asset_field:
                query = f"""
                    WITH log_coverage AS (
                        SELECT 
                            `{log_type_field.field_name}` as log_type,
                            COALESCE(LOWER(TRIM(`{asset_field.field_name}`)), 'unknown') as asset_identifier,
                            COUNT(*) as log_volume
                        FROM `{self.project_id}.{table_name}`
                        WHERE `{log_type_field.field_name}` IS NOT NULL
                        GROUP BY `{log_type_field.field_name}`, asset_identifier
                    )
                    SELECT 
                        log_type,
                        COUNT(DISTINCT asset_identifier) as assets_generating_logs,
                        SUM(log_volume) as total_log_volume,
                        ROUND(AVG(log_volume), 2) as avg_logs_per_asset
                    FROM log_coverage
                    WHERE asset_identifier != 'unknown'
                    GROUP BY log_type
                    ORDER BY assets_generating_logs DESC
                """
                key_fields = [log_type_field.field_name, asset_field.field_name]
                confidence = (log_type_field.confidence_score + asset_field.confidence_score) / 2
            else:
                query = f"""
                    SELECT 
                        `{log_type_field.field_name}` as log_type,
                        COUNT(*) as total_events,
                        COUNT(DISTINCT DATE(`{log_type_field.field_name}`)) as days_with_data
                    FROM `{self.project_id}.{table_name}`
                    WHERE `{log_type_field.field_name}` IS NOT NULL
                    GROUP BY `{log_type_field.field_name}`
                    ORDER BY total_events DESC
                """
                key_fields = [log_type_field.field_name]
                confidence = log_type_field.confidence_score
            
            queries.append(GeneratedQuery(
                purpose=f"Log Source Analysis - {table_name}",
                query=query.strip(),
                source_tables=[table_name],
                key_fields=key_fields,
                expected_result_type="log_coverage",
                confidence_score=confidence
            ))
        
        return queries
    
    def _generate_geographic_queries(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> List[GeneratedQuery]:
        queries = []
        geo_tables = self._find_tables_with_category(semantic_results, 'geographic_data')
        
        for table_name, geo_fields in geo_tables.items():
            geo_field = self._find_best_field(geo_fields, 'geographic_data')
            asset_field = self._find_best_field(geo_fields, ['asset_identity', 'network_identity'])
            
            if not geo_field:
                continue
            
            if asset_field:
                query = f"""
                    WITH geographic_distribution AS (
                        SELECT 
                            `{geo_field.field_name}` as geographic_location,
                            COUNT(DISTINCT COALESCE(LOWER(TRIM(`{asset_field.field_name}`)), '')) as unique_assets,
                            COUNT(*) as total_records
                        FROM `{self.project_id}.{table_name}`
                        WHERE `{geo_field.field_name}` IS NOT NULL
                        AND `{asset_field.field_name}` IS NOT NULL
                        AND TRIM(`{asset_field.field_name}`) != ''
                        GROUP BY `{geo_field.field_name}`
                    )
                    SELECT 
                        geographic_location,
                        unique_assets,
                        total_records,
                        ROUND(unique_assets * 100.0 / SUM(unique_assets) OVER(), 2) as asset_percentage
                    FROM geographic_distribution
                    ORDER BY unique_assets DESC
                """
                key_fields = [geo_field.field_name, asset_field.field_name]
                confidence = (geo_field.confidence_score + asset_field.confidence_score) / 2
            else:
                query = f"""
                    SELECT 
                        `{geo_field.field_name}` as geographic_location,
                        COUNT(*) as record_count,
                        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                    FROM `{self.project_id}.{table_name}`
                    WHERE `{geo_field.field_name}` IS NOT NULL
                    GROUP BY `{geo_field.field_name}`
                    ORDER BY record_count DESC
                """
                key_fields = [geo_field.field_name]
                confidence = geo_field.confidence_score
            
            queries.append(GeneratedQuery(
                purpose=f"Geographic Distribution - {table_name}",
                query=query.strip(),
                source_tables=[table_name],
                key_fields=key_fields,
                expected_result_type="geographic_coverage",
                confidence_score=confidence
            ))
        
        return queries
    
    def _generate_relationship_queries(self, semantic_results: Dict[str, List[FieldAnalysis]]) -> List[GeneratedQuery]:
        queries = []
        
        asset_tables = self._find_tables_with_category(semantic_results, 'asset_identity')
        tool_tables = self._find_tables_with_category(semantic_results, 'security_tools')
        
        for asset_table, asset_fields in asset_tables.items():
            for tool_table, tool_fields in tool_tables.items():
                if asset_table == tool_table:
                    continue
                
                asset_key = self._find_best_field(asset_fields, 'asset_identity')
                tool_key = self._find_best_field(tool_fields, ['asset_identity', 'network_identity'])
                
                if not asset_key or not tool_key:
                    continue
                
                tool_name = self._extract_tool_name(tool_table, tool_fields)
                
                query = f"""
                    WITH asset_tool_mapping AS (
                        SELECT 
                            a.`{asset_key.field_name}` as asset_id,
                            t.`{tool_key.field_name}` as tool_asset_id,
                            CASE 
                                WHEN LOWER(TRIM(a.`{asset_key.field_name}`)) = LOWER(TRIM(t.`{tool_key.field_name}`)) THEN 'EXACT_MATCH'
                                WHEN LOWER(TRIM(a.`{asset_key.field_name}`)) LIKE CONCAT('%', LOWER(TRIM(t.`{tool_key.field_name}`)), '%') THEN 'PARTIAL_MATCH'
                                WHEN LOWER(TRIM(t.`{tool_key.field_name}`)) LIKE CONCAT('%', LOWER(TRIM(a.`{asset_key.field_name}`)), '%') THEN 'REVERSE_MATCH'
                                ELSE 'NO_MATCH'
                            END as match_type
                        FROM `{self.project_id}.{asset_table}` a
                        FULL OUTER JOIN `{self.project_id}.{tool_table}` t
                        ON LOWER(TRIM(a.`{asset_key.field_name}`)) = LOWER(TRIM(t.`{tool_key.field_name}`))
                        WHERE a.`{asset_key.field_name}` IS NOT NULL OR t.`{tool_key.field_name}` IS NOT NULL
                    )
                    SELECT 
                        match_type,
                        COUNT(*) as relationship_count,
                        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                    FROM asset_tool_mapping
                    GROUP BY match_type
                    ORDER BY relationship_count DESC
                """
                
                queries.append(GeneratedQuery(
                    purpose=f"Asset-Tool Relationship - {asset_table} to {tool_name}",
                    query=query.strip(),
                    source_tables=[asset_table, tool_table],
                    key_fields=[asset_key.field_name, tool_key.field_name],
                    expected_result_type="relationship_analysis",
                    confidence_score=min(asset_key.confidence_score, tool_key.confidence_score)
                ))
        
        return queries
    
    def _find_tables_with_category(self, semantic_results: Dict[str, List[FieldAnalysis]], category: str) -> Dict[str, List[FieldAnalysis]]:
        matching_tables = {}
        for table_name, fields in semantic_results.items():
            category_fields = [f for f in fields if f.ao1_category == category and f.confidence_score > 0.6]
            if category_fields:
                matching_tables[table_name] = category_fields
        return matching_tables
    
    def _find_best_field(self, fields: List[FieldAnalysis], categories: str | List[str]) -> Optional[FieldAnalysis]:
        if isinstance(categories, str):
            categories = [categories]
        
        candidates = [f for f in fields if f.ao1_category in categories]
        if not candidates:
            return None
        
        return max(candidates, key=lambda f: f.confidence_score)
    
    def _extract_tool_name(self, table_name: str, fields: List[FieldAnalysis]) -> str:
        table_lower = table_name.lower()
        
        tool_indicators = {
            'crowdstrike': 'CrowdStrike',
            'chronicle': 'Chronicle', 
            'splunk': 'Splunk',
            'edr': 'EDR',
            'endpoint': 'Endpoint Security',
            'agent': 'Security Agent',
            'security': 'Security Tool'
        }
        
        for indicator, name in tool_indicators.items():
            if indicator in table_lower:
                return name
        
        for field in fields:
            field_lower = field.field_name.lower()
            for indicator, name in tool_indicators.items():
                if indicator in field_lower:
                    return name
        
        return table_name.split('.')[-1].replace('_', ' ').title()