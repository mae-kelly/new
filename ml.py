#!/usr/bin/env python3

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FieldMapping:
    table: str
    column: str
    field_type: str  # 'hostname', 'ip', 'log_type', 'region', 'status', etc.
    confidence: float
    sample_values: List[str]
    unique_count: int
    total_count: int
    patterns: List[str]

@dataclass
class SchemaIntelligence:
    tables: List[str]
    field_mappings: Dict[str, FieldMapping]
    hostname_fields: List[FieldMapping]
    ip_fields: List[FieldMapping]
    log_type_fields: List[FieldMapping]
    region_fields: List[FieldMapping]
    status_fields: List[FieldMapping]
    timestamp_fields: List[FieldMapping]

class IntelligentAO1Engine:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.connection = None
        self.schema_intel = None
        
    @contextmanager
    def db_connection(self):
        try:
            self.connection = duckdb.connect(str(self.db_path))
            yield self.connection
        finally:
            if self.connection:
                self.connection.close()
                self.connection = None
    
    def discover_actual_schema(self) -> SchemaIntelligence:
        """Discover the actual schema and intelligently map fields to AO1 concepts"""
        logger.info("🔍 DISCOVERING ACTUAL DATABASE SCHEMA")
        
        with self.db_connection():
            # Get all tables
            tables = self._get_tables()
            logger.info(f"📊 Found tables: {tables}")
            
            all_mappings = {}
            hostname_fields = []
            ip_fields = []
            log_type_fields = []
            region_fields = []
            status_fields = []
            timestamp_fields = []
            
            for table in tables:
                columns = self._get_columns(table)
                logger.info(f"🔎 Analyzing {table}: {len(columns)} columns")
                
                for column in columns:
                    mapping = self._analyze_field_intelligence(table, column)
                    if mapping:
                        key = f"{table}.{column}"
                        all_mappings[key] = mapping
                        
                        # Categorize by field type
                        if mapping.field_type == 'hostname':
                            hostname_fields.append(mapping)
                        elif mapping.field_type == 'ip':
                            ip_fields.append(mapping)
                        elif mapping.field_type == 'log_type':
                            log_type_fields.append(mapping)
                        elif mapping.field_type == 'region':
                            region_fields.append(mapping)
                        elif mapping.field_type == 'status':
                            status_fields.append(mapping)
                        elif mapping.field_type == 'timestamp':
                            timestamp_fields.append(mapping)
            
            self.schema_intel = SchemaIntelligence(
                tables=tables,
                field_mappings=all_mappings,
                hostname_fields=sorted(hostname_fields, key=lambda x: x.confidence, reverse=True),
                ip_fields=sorted(ip_fields, key=lambda x: x.confidence, reverse=True),
                log_type_fields=sorted(log_type_fields, key=lambda x: x.confidence, reverse=True),
                region_fields=sorted(region_fields, key=lambda x: x.confidence, reverse=True),
                status_fields=sorted(status_fields, key=lambda x: x.confidence, reverse=True),
                timestamp_fields=sorted(timestamp_fields, key=lambda x: x.confidence, reverse=True)
            )
            
            logger.info(f"🧠 SCHEMA INTELLIGENCE SUMMARY:")
            logger.info(f"   📡 Hostname fields: {len(hostname_fields)}")
            logger.info(f"   🌐 IP fields: {len(ip_fields)}")
            logger.info(f"   📋 Log type fields: {len(log_type_fields)}")
            logger.info(f"   🌍 Region fields: {len(region_fields)}")
            logger.info(f"   ⚡ Status fields: {len(status_fields)}")
            logger.info(f"   ⏰ Timestamp fields: {len(timestamp_fields)}")
            
            return self.schema_intel
    
    def _get_tables(self) -> List[str]:
        """Get all table names"""
        try:
            result = self.connection.execute("SHOW TABLES").fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.warning(f"Could not get tables: {e}")
            # Try alternative methods
            possible_tables = ['combined', 'all_sources', 'main', 'data']
            existing = []
            for table in possible_tables:
                try:
                    self.connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
                    existing.append(table)
                except:
                    continue
            return existing
    
    def _get_columns(self, table: str) -> List[str]:
        """Get all column names for a table"""
        try:
            result = self.connection.execute(f"DESCRIBE {table}").fetchall()
            return [row[0] for row in result]
        except Exception as e:
            try:
                # Alternative method
                result = self.connection.execute(f"SELECT * FROM {table} LIMIT 0").description
                return [col[0] for col in result]
            except:
                logger.warning(f"Could not get columns for {table}: {e}")
                return []
    
    def _analyze_field_intelligence(self, table: str, column: str) -> Optional[FieldMapping]:
        """Analyze a field to determine its semantic meaning for AO1"""
        
        # Get sample data
        samples = self._get_sample_data(table, column, limit=200)
        if not samples:
            return None
        
        # Clean samples
        clean_samples = [str(s).strip() for s in samples if s is not None and str(s).strip()]
        if not clean_samples:
            return None
        
        unique_count = len(set(clean_samples))
        total_count = len(clean_samples)
        
        # Analyze patterns and determine field type
        field_type, confidence, patterns = self._classify_field_type(column, clean_samples)
        
        if confidence < 0.3:
            return None
        
        return FieldMapping(
            table=table,
            column=column,
            field_type=field_type,
            confidence=confidence,
            sample_values=clean_samples[:5],
            unique_count=unique_count,
            total_count=total_count,
            patterns=patterns
        )
    
    def _get_sample_data(self, table: str, column: str, limit: int = 200) -> List[Any]:
        """Get sample data from a column"""
        queries = [
            f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {limit}',
            f'SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT {limit}',
            f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {limit}',
            f'SELECT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT {limit}'
        ]
        
        for query in queries:
            try:
                result = self.connection.execute(query).fetchall()
                return [row[0] for row in result if row[0] is not None]
            except Exception as e:
                continue
        
        logger.debug(f"Could not get samples for {table}.{column}")
        return []
    
    def _classify_field_type(self, column_name: str, samples: List[str]) -> Tuple[str, float, List[str]]:
        """Classify what type of field this is for AO1 purposes"""
        
        column_lower = column_name.lower()
        patterns = []
        
        # Hostname detection
        if self._is_hostname_field(column_lower, samples):
            confidence = self._calculate_hostname_confidence(samples)
            patterns = self._extract_hostname_patterns(samples)
            return 'hostname', confidence, patterns
        
        # IP address detection
        if self._is_ip_field(column_lower, samples):
            confidence = self._calculate_ip_confidence(samples)
            patterns = self._extract_ip_patterns(samples)
            return 'ip', confidence, patterns
        
        # Log type detection
        if self._is_log_type_field(column_lower, samples):
            confidence = self._calculate_log_type_confidence(samples)
            patterns = self._extract_log_type_patterns(samples)
            return 'log_type', confidence, patterns
        
        # Region/geographic detection
        if self._is_region_field(column_lower, samples):
            confidence = self._calculate_region_confidence(samples)
            patterns = self._extract_region_patterns(samples)
            return 'region', confidence, patterns
        
        # Status/health detection
        if self._is_status_field(column_lower, samples):
            confidence = self._calculate_status_confidence(samples)
            patterns = self._extract_status_patterns(samples)
            return 'status', confidence, patterns
        
        # Timestamp detection
        if self._is_timestamp_field(column_lower, samples):
            confidence = self._calculate_timestamp_confidence(samples)
            patterns = self._extract_timestamp_patterns(samples)
            return 'timestamp', confidence, patterns
        
        return 'unknown', 0.0, []
    
    def _is_hostname_field(self, column_name: str, samples: List[str]) -> bool:
        """Check if this looks like a hostname field"""
        hostname_indicators = [
            'host', 'hostname', 'device', 'server', 'machine', 
            'computer', 'node', 'endpoint', 'asset', 'system'
        ]
        
        # Check column name
        if any(indicator in column_name for indicator in hostname_indicators):
            return True
        
        # Check sample patterns
        hostname_like = 0
        for sample in samples[:20]:
            if self._looks_like_hostname(sample):
                hostname_like += 1
        
        return hostname_like > len(samples[:20]) * 0.3
    
    def _looks_like_hostname(self, value: str) -> bool:
        """Check if a value looks like a hostname"""
        if not isinstance(value, str) or len(value) < 3:
            return False
        
        # Basic hostname patterns
        if re.match(r'^[a-zA-Z0-9\-\.]+$', value):
            if '.' in value and not value.replace('.', '').isdigit():
                return True
            if '-' in value and any(c.isalpha() for c in value):
                return True
            if any(c.isalpha() for c in value) and len(value) > 3:
                return True
        
        return False
    
    def _calculate_hostname_confidence(self, samples: List[str]) -> float:
        """Calculate confidence that these are hostnames"""
        hostname_count = sum(1 for s in samples if self._looks_like_hostname(s))
        base_confidence = hostname_count / len(samples)
        
        # Boost confidence based on patterns
        has_domains = any('.' in s and not s.replace('.', '').isdigit() for s in samples)
        has_dashes = any('-' in s for s in samples)
        reasonable_length = np.mean([len(s) for s in samples]) > 5
        
        if has_domains:
            base_confidence += 0.2
        if has_dashes:
            base_confidence += 0.1
        if reasonable_length:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _extract_hostname_patterns(self, samples: List[str]) -> List[str]:
        """Extract hostname patterns for analysis"""
        patterns = []
        
        # Domain patterns
        domains = set()
        for sample in samples:
            if '.' in sample and not sample.replace('.', '').isdigit():
                parts = sample.split('.')
                if len(parts) > 1:
                    domains.add('.'.join(parts[-2:]))  # Last two parts
        
        if domains:
            patterns.append(f"Domains: {', '.join(list(domains)[:5])}")
        
        # Naming conventions
        if any('-' in s for s in samples):
            patterns.append("Uses hyphen naming convention")
        
        # Length analysis
        avg_length = np.mean([len(s) for s in samples])
        patterns.append(f"Average length: {avg_length:.1f} characters")
        
        return patterns
    
    def _is_ip_field(self, column_name: str, samples: List[str]) -> bool:
        """Check if this looks like an IP field"""
        ip_indicators = ['ip', 'addr', 'address', 'src', 'dst', 'source', 'dest']
        
        if any(indicator in column_name for indicator in ip_indicators):
            return True
        
        ip_count = sum(1 for s in samples if self._looks_like_ip(s))
        return ip_count > len(samples) * 0.5
    
    def _looks_like_ip(self, value: str) -> bool:
        """Check if a value looks like an IP address"""
        if not isinstance(value, str):
            return False
        
        # IPv4 pattern
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ipv4_pattern, value):
            parts = value.split('.')
            return all(0 <= int(part) <= 255 for part in parts)
        
        # IPv6 pattern (basic)
        if ':' in value and len(value) > 10:
            return True
        
        return False
    
    def _calculate_ip_confidence(self, samples: List[str]) -> float:
        """Calculate confidence that these are IP addresses"""
        ip_count = sum(1 for s in samples if self._looks_like_ip(s))
        return ip_count / len(samples)
    
    def _extract_ip_patterns(self, samples: List[str]) -> List[str]:
        """Extract IP patterns"""
        patterns = []
        
        # Network ranges
        private_count = sum(1 for s in samples if self._is_private_ip(s))
        if private_count > 0:
            patterns.append(f"Private IPs: {private_count}/{len(samples)}")
        
        # IPv6 count
        ipv6_count = sum(1 for s in samples if ':' in s)
        if ipv6_count > 0:
            patterns.append(f"IPv6 addresses: {ipv6_count}")
        
        return patterns
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is in private range"""
        if not self._looks_like_ip(ip):
            return False
        
        private_ranges = ['10.', '192.168.', '172.']
        return any(ip.startswith(range_start) for range_start in private_ranges)
    
    def _is_log_type_field(self, column_name: str, samples: List[str]) -> bool:
        """Check if this looks like a log type field"""
        log_indicators = ['type', 'source', 'category', 'kind', 'event', 'log']
        
        if any(indicator in column_name for indicator in log_indicators):
            return True
        
        # Check for log-like values
        log_terms = [
            'firewall', 'dns', 'proxy', 'web', 'auth', 'syslog', 'windows',
            'linux', 'network', 'security', 'audit', 'access', 'error'
        ]
        
        log_like_count = 0
        for sample in samples:
            if any(term in str(sample).lower() for term in log_terms):
                log_like_count += 1
        
        return log_like_count > len(samples) * 0.3
    
    def _calculate_log_type_confidence(self, samples: List[str]) -> float:
        """Calculate confidence for log type field"""
        log_terms = [
            'firewall', 'dns', 'proxy', 'web', 'auth', 'syslog', 'windows',
            'linux', 'network', 'security', 'audit', 'access', 'error',
            'edr', 'endpoint', 'dlp', 'ids', 'ips', 'waf'
        ]
        
        matches = sum(1 for s in samples if any(term in str(s).lower() for term in log_terms))
        base_confidence = matches / len(samples)
        
        # Boost if we have good variety but not too much
        unique_ratio = len(set(samples)) / len(samples)
        if 0.1 < unique_ratio < 0.8:  # Good variety but not random
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _extract_log_type_patterns(self, samples: List[str]) -> List[str]:
        """Extract log type patterns"""
        patterns = []
        
        # Count by category
        categories = defaultdict(int)
        for sample in samples:
            sample_lower = str(sample).lower()
            if 'firewall' in sample_lower:
                categories['firewall'] += 1
            elif 'dns' in sample_lower:
                categories['dns'] += 1
            elif 'web' in sample_lower or 'http' in sample_lower:
                categories['web'] += 1
            elif 'auth' in sample_lower:
                categories['auth'] += 1
            elif 'windows' in sample_lower or 'win' in sample_lower:
                categories['windows'] += 1
            elif 'linux' in sample_lower:
                categories['linux'] += 1
        
        if categories:
            top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            patterns.append(f"Top types: {', '.join([f'{k}({v})' for k, v in top_categories])}")
        
        return patterns
    
    def _is_region_field(self, column_name: str, samples: List[str]) -> bool:
        """Check if this looks like a region field"""
        region_indicators = ['region', 'country', 'location', 'geo', 'zone', 'area']
        
        if any(indicator in column_name for indicator in region_indicators):
            return True
        
        # Check for geographic values
        geo_terms = [
            'us', 'usa', 'uk', 'ca', 'de', 'fr', 'jp', 'au', 'north', 'south',
            'east', 'west', 'america', 'europe', 'asia', 'pacific'
        ]
        
        geo_count = sum(1 for s in samples if any(term in str(s).lower() for term in geo_terms))
        return geo_count > len(samples) * 0.3
    
    def _calculate_region_confidence(self, samples: List[str]) -> float:
        """Calculate confidence for region field"""
        geo_terms = [
            'us', 'usa', 'uk', 'ca', 'de', 'fr', 'jp', 'au', 'north', 'south',
            'east', 'west', 'america', 'europe', 'asia', 'pacific', 'emea', 'apac'
        ]
        
        matches = sum(1 for s in samples if any(term in str(s).lower() for term in geo_terms))
        return matches / len(samples)
    
    def _extract_region_patterns(self, samples: List[str]) -> List[str]:
        """Extract region patterns"""
        patterns = []
        
        # Count regions
        regions = Counter(str(s).lower() for s in samples)
        top_regions = regions.most_common(3)
        
        if top_regions:
            patterns.append(f"Top regions: {', '.join([f'{r}({c})' for r, c in top_regions])}")
        
        return patterns
    
    def _is_status_field(self, column_name: str, samples: List[str]) -> bool:
        """Check if this looks like a status field"""
        status_indicators = ['status', 'state', 'health', 'condition', 'active', 'enabled']
        
        if any(indicator in column_name for indicator in status_indicators):
            return True
        
        # Check for status-like values
        status_terms = [
            'active', 'inactive', 'enabled', 'disabled', 'up', 'down',
            'online', 'offline', 'healthy', 'unhealthy', 'ok', 'error',
            'good', 'bad', 'running', 'stopped', 'true', 'false'
        ]
        
        status_count = sum(1 for s in samples if str(s).lower() in status_terms)
        return status_count > len(samples) * 0.5
    
    def _calculate_status_confidence(self, samples: List[str]) -> float:
        """Calculate confidence for status field"""
        status_terms = [
            'active', 'inactive', 'enabled', 'disabled', 'up', 'down',
            'online', 'offline', 'healthy', 'unhealthy', 'ok', 'error',
            'good', 'bad', 'running', 'stopped', 'true', 'false', '1', '0'
        ]
        
        matches = sum(1 for s in samples if str(s).lower() in status_terms)
        base_confidence = matches / len(samples)
        
        # Boost if low cardinality (typical for status fields)
        unique_count = len(set(str(s).lower() for s in samples))
        if unique_count <= 10:
            base_confidence += 0.3
        
        return min(base_confidence, 1.0)
    
    def _extract_status_patterns(self, samples: List[str]) -> List[str]:
        """Extract status patterns"""
        patterns = []
        
        status_counts = Counter(str(s).lower() for s in samples)
        patterns.append(f"Status values: {dict(status_counts.most_common(5))}")
        
        return patterns
    
    def _is_timestamp_field(self, column_name: str, samples: List[str]) -> bool:
        """Check if this looks like a timestamp field"""
        time_indicators = ['time', 'date', 'timestamp', 'created', 'updated', 'last']
        
        if any(indicator in column_name for indicator in time_indicators):
            return True
        
        # Check for timestamp-like values
        timestamp_count = sum(1 for s in samples if self._looks_like_timestamp(s))
        return timestamp_count > len(samples) * 0.5
    
    def _looks_like_timestamp(self, value: str) -> bool:
        """Check if value looks like a timestamp"""
        if not isinstance(value, str):
            return False
        
        # Common timestamp patterns
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{10,13}',          # Unix timestamp
        ]
        
        return any(re.search(pattern, value) for pattern in timestamp_patterns)
    
    def _calculate_timestamp_confidence(self, samples: List[str]) -> float:
        """Calculate confidence for timestamp field"""
        timestamp_count = sum(1 for s in samples if self._looks_like_timestamp(s))
        return timestamp_count / len(samples)
    
    def _extract_timestamp_patterns(self, samples: List[str]) -> List[str]:
        """Extract timestamp patterns"""
        patterns = []
        
        # Identify format
        has_iso = any('-' in str(s) and len(str(s)) > 8 for s in samples)
        has_slash = any('/' in str(s) for s in samples)
        has_unix = any(str(s).isdigit() and len(str(s)) >= 10 for s in samples)
        
        formats = []
        if has_iso:
            formats.append("ISO format")
        if has_slash:
            formats.append("slash format") 
        if has_unix:
            formats.append("Unix timestamp")
        
        if formats:
            patterns.append(f"Formats: {', '.join(formats)}")
        
        return patterns
    
    def generate_intelligent_ao1_queries(self) -> Dict[str, str]:
        """Generate AO1 queries based on actual discovered schema"""
        
        if not self.schema_intel:
            raise ValueError("Must run discover_actual_schema() first")
        
        logger.info("⚡ GENERATING INTELLIGENT AO1 QUERIES")
        
        queries = {}
        
        # Use the best fields we found for each query type
        primary_hostname = self.schema_intel.hostname_fields[0] if self.schema_intel.hostname_fields else None
        primary_log_type = self.schema_intel.log_type_fields[0] if self.schema_intel.log_type_fields else None
        primary_region = self.schema_intel.region_fields[0] if self.schema_intel.region_fields else None
        primary_status = self.schema_intel.status_fields[0] if self.schema_intel.status_fields else None
        primary_ip = self.schema_intel.ip_fields[0] if self.schema_intel.ip_fields else None
        
        # 1. Global Asset Coverage
        if primary_hostname:
            queries['global_asset_coverage'] = f"""
-- Global Asset Coverage (Based on discovered hostname field: {primary_hostname.table}.{primary_hostname.column})
SELECT 
    'Global Asset Coverage' as metric_name,
    COUNT(DISTINCT {primary_hostname.column}) as total_unique_assets,
    COUNT(*) as total_records,
    ROUND(COUNT(DISTINCT {primary_hostname.column}) * 100.0 / COUNT(*), 2) as asset_to_record_ratio
FROM {primary_hostname.table}
WHERE {primary_hostname.column} IS NOT NULL;
"""
        
        # 2. Infrastructure Type Coverage
        if primary_hostname:
            queries['infrastructure_type_coverage'] = f"""
-- Infrastructure Type Coverage (Based on hostname patterns in {primary_hostname.table}.{primary_hostname.column})
SELECT 
    CASE 
        WHEN LOWER({primary_hostname.column}) LIKE '%cloud%' OR LOWER({primary_hostname.column}) LIKE '%aws%' 
             OR LOWER({primary_hostname.column}) LIKE '%azure%' OR LOWER({primary_hostname.column}) LIKE '%gcp%' THEN 'Cloud'
        WHEN LOWER({primary_hostname.column}) LIKE '%vm%' OR LOWER({primary_hostname.column}) LIKE '%virtual%' THEN 'Virtual'
        WHEN LOWER({primary_hostname.column}) LIKE '%server%' THEN 'Server'
        WHEN LOWER({primary_hostname.column}) LIKE '%desktop%' OR LOWER({primary_hostname.column}) LIKE '%laptop%' THEN 'Endpoint'
        ELSE 'Other'
    END as infrastructure_type,
    COUNT(DISTINCT {primary_hostname.column}) as asset_count,
    ROUND(COUNT(DISTINCT {primary_hostname.column}) * 100.0 / 
          (SELECT COUNT(DISTINCT {primary_hostname.column}) FROM {primary_hostname.table} WHERE {primary_hostname.column} IS NOT NULL), 2) as percentage
FROM {primary_hostname.table}
WHERE {primary_hostname.column} IS NOT NULL
GROUP BY infrastructure_type
ORDER BY asset_count DESC;
"""
        
        # 3. Regional Coverage (if we found region data)
        if primary_region:
            queries['regional_coverage'] = f"""
-- Regional Coverage Analysis (Based on {primary_region.table}.{primary_region.column})
SELECT 
    {primary_region.column} as region,
    COUNT(DISTINCT {primary_hostname.column if primary_hostname else '*'}) as assets_in_region,
    COUNT(*) as total_logs,
    ROUND(COUNT(DISTINCT {primary_hostname.column if primary_hostname else '*'}) * 100.0 / 
          (SELECT COUNT(DISTINCT {primary_hostname.column if primary_hostname else '*'}) FROM {primary_region.table}), 2) as region_percentage
FROM {primary_region.table}
WHERE {primary_region.column} IS NOT NULL
GROUP BY {primary_region.column}
ORDER BY assets_in_region DESC;
"""
        
        # 4. Log Type Coverage
        if primary_log_type:
            queries['log_type_coverage'] = f"""
-- Log Type Coverage (Based on {primary_log_type.table}.{primary_log_type.column})
SELECT 
    {primary_log_type.column} as log_type,
    COUNT(DISTINCT {primary_hostname.column if primary_hostname else '*'}) as assets_with_this_log_type,
    COUNT(*) as total_logs_of_type,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {primary_log_type.table}), 2) as log_volume_percentage
FROM {primary_log_type.table}
WHERE {primary_log_type.column} IS NOT NULL
GROUP BY {primary_log_type.column}
ORDER BY total_logs_of_type DESC;
"""
        
        # 5. Network vs Endpoint Coverage (based on log types)
        if primary_log_type and primary_hostname:
            queries['network_vs_endpoint_coverage'] = f"""
-- Network vs Endpoint Coverage (Derived from log types in {primary_log_type.table}.{primary_log_type.column})
SELECT 
    CASE 
        WHEN LOWER({primary_log_type.column}) LIKE '%firewall%' OR LOWER({primary_log_type.column}) LIKE '%network%' 
             OR LOWER({primary_log_type.column}) LIKE '%dns%' OR LOWER({primary_log_type.column}) LIKE '%proxy%' THEN 'Network Logs'
        WHEN LOWER({primary_log_type.column}) LIKE '%endpoint%' OR LOWER({primary_log_type.column}) LIKE '%windows%' 
             OR LOWER({primary_log_type.column}) LIKE '%linux%' OR LOWER({primary_log_type.column}) LIKE '%edr%' THEN 'Endpoint Logs'
        ELSE 'Other Logs'
    END as coverage_category,
    COUNT(DISTINCT {primary_hostname.column}) as unique_assets,
    COUNT(*) as total_logs,
    ROUND(COUNT(DISTINCT {primary_hostname.column}) * 100.0 / 
          (SELECT COUNT(DISTINCT {primary_hostname.column}) FROM {primary_log_type.table} WHERE {primary_hostname.column} IS NOT NULL), 2) as asset_coverage_percentage
FROM {primary_log_type.table}
WHERE {primary_log_type.column} IS NOT NULL AND {primary_hostname.column} IS NOT NULL
GROUP BY coverage_category
ORDER BY unique_assets DESC;
"""
        
        # 6. Agent Status Coverage (if we found status data)
        if primary_status and primary_hostname:
            queries['agent_status_coverage'] = f"""
-- Agent Status Coverage (Based on {primary_status.table}.{primary_status.column})
SELECT 
    {primary_status.column} as agent_status,
    COUNT(DISTINCT {primary_hostname.column}) as assets_in_status,
    ROUND(COUNT(DISTINCT {primary_hostname.column}) * 100.0 / 
          (SELECT COUNT(DISTINCT {primary_hostname.column}) FROM {primary_status.table} WHERE {primary_hostname.column} IS NOT NULL), 2) as status_percentage
FROM {primary_status.table}
WHERE {primary_status.column} IS NOT NULL AND {primary_hostname.column} IS NOT NULL
GROUP BY {primary_status.column}
ORDER BY assets_in_status DESC;
"""
        
        # 7. IP Address Coverage (if we found IP data)
        if primary_ip and primary_hostname:
            queries['ip_coverage_analysis'] = f"""
-- IP Address Coverage (Based on {primary_ip.table}.{primary_ip.column})
SELECT 
    CASE 
        WHEN {primary_ip.column} LIKE '10.%' OR {primary_ip.column} LIKE '192.168.%' 
             OR {primary_ip.column} LIKE '172.%' THEN 'Private IP'
        WHEN {primary_ip.column} NOT LIKE '10.%' AND {primary_ip.column} NOT LIKE '192.168.%' 
             AND {primary_ip.column} NOT LIKE '172.%' AND {primary_ip.column} LIKE '%.%.%.%' THEN 'Public IP'
        ELSE 'Other/IPv6'
    END as ip_type,
    COUNT(DISTINCT {primary_ip.column}) as unique_ips,
    COUNT(DISTINCT {primary_hostname.column}) as assets_with_ips,
    ROUND(COUNT(DISTINCT {primary_hostname.column}) * 100.0 / 
          (SELECT COUNT(DISTINCT {primary_hostname.column}) FROM {primary_ip.table} WHERE {primary_hostname.column} IS NOT NULL), 2) as coverage_percentage
FROM {primary_ip.table}
WHERE {primary_ip.column} IS NOT NULL AND {primary_hostname.column} IS NOT NULL
GROUP BY ip_type
ORDER BY unique_ips DESC;
"""
        
        # 8. Data Quality Assessment
        if primary_hostname:
            queries['data_quality_assessment'] = f"""
-- Data Quality Assessment (Based on primary fields discovered)
SELECT 
    'Data Quality Metrics' as assessment_type,
    COUNT(*) as total_records,
    COUNT(DISTINCT {primary_hostname.column}) as unique_assets,
    ROUND(COUNT(DISTINCT {primary_hostname.column}) * 100.0 / COUNT(*), 2) as asset_uniqueness_ratio,
    COUNT(CASE WHEN {primary_hostname.column} IS NULL THEN 1 END) as null_hostname_count,
    ROUND(COUNT(CASE WHEN {primary_hostname.column} IS NULL THEN 1 END) * 100.0 / COUNT(*), 2) as null_percentage
    {f', COUNT(CASE WHEN {primary_log_type.column} IS NULL THEN 1 END) as null_log_type_count' if primary_log_type else ''}
    {f', COUNT(CASE WHEN {primary_region.column} IS NULL THEN 1 END) as null_region_count' if primary_region else ''}
FROM {primary_hostname.table};
"""
        
        # 9. Coverage Summary Across All Fields
        all_tables = list(set(mapping.table for mapping in self.schema_intel.field_mappings.values()))
        if all_tables:
            main_table = all_tables[0]  # Use the first table as primary
            queries['ao1_coverage_summary'] = f"""
-- AO1 Coverage Summary (Comprehensive view from {main_table})
SELECT 
    'AO1 Coverage Summary' as report_type,
    COUNT(*) as total_records_analyzed,
    {f"COUNT(DISTINCT {primary_hostname.column}) as unique_assets_identified," if primary_hostname else "NULL as unique_assets_identified,"}
    {f"COUNT(DISTINCT {primary_log_type.column}) as unique_log_types," if primary_log_type else "NULL as unique_log_types,"}
    {f"COUNT(DISTINCT {primary_region.column}) as unique_regions," if primary_region else "NULL as unique_regions,"}
    {f"COUNT(DISTINCT {primary_ip.column}) as unique_ip_addresses," if primary_ip else "NULL as unique_ip_addresses,"}
    {f"COUNT(DISTINCT {primary_status.column}) as unique_status_values," if primary_status else "NULL as unique_status_values,"}
    CURRENT_TIMESTAMP as analysis_timestamp
FROM {main_table};
"""
        
        logger.info(f"✅ Generated {len(queries)} intelligent AO1 queries based on actual schema")
        
        return queries
    
    def validate_and_execute_queries(self, queries: Dict[str, str]) -> Dict[str, Any]:
        """Validate and execute the generated queries"""
        logger.info("🔍 VALIDATING AND EXECUTING QUERIES")
        
        results = {}
        
        with self.db_connection():
            for query_name, sql in queries.items():
                logger.info(f"   🔄 Testing: {query_name}")
                
                try:
                    result = self.connection.execute(sql).fetchall()
                    column_names = [desc[0] for desc in self.connection.description]
                    
                    results[query_name] = {
                        'status': 'SUCCESS',
                        'row_count': len(result),
                        'columns': column_names,
                        'data': result[:10],  # First 10 rows
                        'sql': sql
                    }
                    
                    logger.info(f"   ✅ {query_name}: {len(result)} rows")
                    
                except Exception as e:
                    results[query_name] = {
                        'status': 'FAILED',
                        'error': str(e),
                        'sql': sql
                    }
                    
                    logger.warning(f"   ❌ {query_name}: {str(e)}")
        
        success_count = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
        logger.info(f"🎯 VALIDATION COMPLETE: {success_count}/{len(queries)} queries successful")
        
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive AO1 readiness report"""
        
        if not self.schema_intel:
            raise ValueError("Must run schema discovery first")
        
        # Generate queries based on discovered schema
        queries = self.generate_intelligent_ao1_queries()
        
        # Validate and execute queries
        query_results = self.validate_and_execute_queries(queries)
        
        # Create comprehensive report
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'database_path': str(self.db_path),
            'schema_intelligence': {
                'tables_discovered': self.schema_intel.tables,
                'total_fields_analyzed': len(self.schema_intel.field_mappings),
                'hostname_fields_found': len(self.schema_intel.hostname_fields),
                'ip_fields_found': len(self.schema_intel.ip_fields),
                'log_type_fields_found': len(self.schema_intel.log_type_fields),
                'region_fields_found': len(self.schema_intel.region_fields),
                'status_fields_found': len(self.schema_intel.status_fields),
                'timestamp_fields_found': len(self.schema_intel.timestamp_fields)
            },
            'field_mappings': {
                f"{mapping.table}.{mapping.column}": {
                    'field_type': mapping.field_type,
                    'confidence': round(mapping.confidence, 3),
                    'unique_values': mapping.unique_count,
                    'total_records': mapping.total_count,
                    'sample_values': mapping.sample_values,
                    'patterns': mapping.patterns
                }
                for mapping in sorted(
                    self.schema_intel.field_mappings.values(), 
                    key=lambda x: x.confidence, 
                    reverse=True
                )[:20]  # Top 20 by confidence
            },
            'ao1_query_results': query_results,
            'ao1_readiness_assessment': {
                'queries_generated': len(queries),
                'queries_successful': sum(1 for r in query_results.values() if r['status'] == 'SUCCESS'),
                'success_rate_percentage': round(
                    sum(1 for r in query_results.values() if r['status'] == 'SUCCESS') / len(queries) * 100, 1
                ) if queries else 0,
                'critical_capabilities': {
                    'asset_identification': len(self.schema_intel.hostname_fields) > 0,
                    'log_type_classification': len(self.schema_intel.log_type_fields) > 0,
                    'geographic_coverage': len(self.schema_intel.region_fields) > 0,
                    'ip_address_tracking': len(self.schema_intel.ip_fields) > 0,
                    'status_monitoring': len(self.schema_intel.status_fields) > 0
                },
                'data_quality_indicators': {
                    'avg_field_confidence': round(
                        np.mean([m.confidence for m in self.schema_intel.field_mappings.values()]), 3
                    ) if self.schema_intel.field_mappings else 0,
                    'high_confidence_fields': sum(
                        1 for m in self.schema_intel.field_mappings.values() if m.confidence > 0.7
                    )
                }
            }
        }
        
        return report
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete intelligent AO1 analysis"""
        logger.info("🚀 STARTING INTELLIGENT AO1 ANALYSIS")
        
        try:
            # Step 1: Discover actual schema
            self.discover_actual_schema()
            
            # Step 2: Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            # Step 3: Save report
            output_file = Path("intelligent_ao1_analysis.json")
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info("🎉 INTELLIGENT ANALYSIS COMPLETE!")
            logger.info(f"📊 Success Rate: {report['ao1_readiness_assessment']['success_rate_percentage']}%")
            logger.info(f"🧠 Fields Analyzed: {report['schema_intelligence']['total_fields_analyzed']}")
            logger.info(f"⚡ Queries Generated: {report['ao1_readiness_assessment']['queries_generated']}")
            logger.info(f"💾 Report saved: {output_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e), 'traceback': traceback.format_exc()}

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Intelligent AO1 Query Generator')
    parser.add_argument('--database', '-d', required=True, help='Path to DuckDB database')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1
    
    print(f"🧠 INTELLIGENT AO1 ANALYSIS ENGINE")
    print(f"🗄️  Database: {db_path}")
    
    try:
        engine = IntelligentAO1Engine(str(db_path))
        results = engine.run_complete_analysis()
        
        if 'error' not in results:
            print(f"\n🎉 ANALYSIS SUCCESSFUL!")
            assessment = results['ao1_readiness_assessment']
            print(f"📊 {assessment['success_rate_percentage']}% query success rate")
            print(f"🔍 {assessment['queries_generated']} AO1 queries generated")
            print(f"✅ {assessment['queries_successful']} queries executed successfully")
            
            capabilities = assessment['critical_capabilities']
            print(f"\n🎯 CRITICAL CAPABILITIES:")
            for capability, available in capabilities.items():
                status = "✅" if available else "❌"
                print(f"   {status} {capability.replace('_', ' ').title()}")
            
            return 0
        else:
            print(f"\n❌ {results['error']}")
            return 1
            
    except Exception as e:
        print(f"\n💥 {e}")
        return 1

if __name__ == "__main__":
    exit(main())