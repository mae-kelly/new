import os
from pathlib import Path

PROJECT_ID = "prj-fisv-p-gcss-sas-dl9dd0f1df"
SERVICE_ACCOUNT_FILE = "gcp_prod_key.json"

SEMANTIC_MODELS_CACHE = "./ao1_models"
RESULTS_OUTPUT_DIR = "./ao1_results"

AO1_CONCEPTS = {
    'asset_identity': {
        'patterns': [
            r'.*host.*', r'.*device.*', r'.*machine.*', r'.*computer.*', r'.*server.*',
            r'.*asset.*', r'.*endpoint.*', r'.*node.*', r'.*system.*', r'.*workstation.*',
            r'.*cmdb.*', r'.*inventory.*', r'.*baseline.*'
        ],
        'content_indicators': [
            'hostname', 'fqdn', 'domain name', 'computer name', 'server name',
            'asset tag', 'device id', 'machine id', 'endpoint name', 'system name',
            'cmdb id', 'asset identifier', 'baseline asset'
        ],
        'value_patterns': [
            r'^[a-zA-Z0-9\-\.]+\.(com|local|corp|internal|net|org)
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*',
            r'.*url.*', r'.*fqdn.*', r'.*domain.*', r'.*dns.*', r'.*vpc.*',
            r'.*zone.*', r'.*subnet.*', r'.*vlan.*', r'.*ipam.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6',
            'url', 'fqdn', 'domain', 'dns name', 'vpc', 'network zone',
            'subnet', 'vlan', 'ipam', 'public ip', 'private ip'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^https?://.*',
            r'^[a-zA-Z0-9\-\.]+\.(com|local|corp|internal|net|org)
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*',
            r'.*tanium.*', r'.*dlp.*', r'.*axonius.*', r'.*falcon.*',
            r'.*health.*', r'.*status.*', r'.*coverage.*', r'.*deployment.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool',
            'tanium', 'dlp', 'axonius', 'agent health', 'reporting status',
            'tool coverage', 'deployment status', 'security control'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*', r'.*tanium.*', r'.*dlp.*',
            r'.*healthy.*', r'.*online.*', r'.*active.*', r'.*deployed.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*',
            r'.*volume.*', r'.*ingest.*', r'.*collection.*', r'.*siem.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event',
            'log volume', 'ingest volume', 'log collection', 'siem data',
            'firewall log', 'dns log', 'authentication log', 'access log'
        ],
        'value_patterns': [
            r'.*_LOG
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}, r'.*_EVENT
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}, r'.*_AUDIT
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}, r'FIREWALL.*', r'DNS.*',
            r'AUTH.*', r'ACCESS.*', r'SECURITY.*', r'NETWORK.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*',
            r'.*datacenter.*', r'.*office.*', r'.*facility.*', r'.*geo.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone',
            'office', 'facility', 'geographic', 'geolocation'
        ],
        'value_patterns': [
            r'^[A-Z]{2}
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}, r'^[A-Z]{3}
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}, r'.*US.*', r'.*EAST.*', r'.*WEST.*',
            r'.*NORTH.*', r'.*SOUTH.*', r'.*DC.*', r'.*SITE.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*',
            r'.*bu.*', r'.*dept.*', r'.*division.*', r'.*team.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department',
            'bu', 'division', 'team', 'application owner', 'service owner'
        ],
        'value_patterns': [
            r'.*_APP
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}, r'.*_SERVICE
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}, r'.*_BU
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}, r'.*_DEPT
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 0.5
    },
    'infrastructure_type': {
        'patterns': [
            r'.*on.*prem.*', r'.*cloud.*', r'.*saas.*', r'.*api.*', r'.*infrastructure.*',
            r'.*platform.*', r'.*environment.*', r'.*tier.*', r'.*category.*'
        ],
        'content_indicators': [
            'on-prem', 'on premise', 'cloud', 'saas', 'api', 'infrastructure type',
            'platform type', 'environment', 'tier', 'category', 'deployment type'
        ],
        'value_patterns': [
            r'.*ON.*PREM.*', r'.*CLOUD.*', r'.*SAAS.*', r'.*API.*',
            r'.*HYBRID.*', r'.*AZURE.*', r'.*AWS.*', r'.*GCP.*'
        ],
        'weight': 0.7
    },
    'system_classification': {
        'patterns': [
            r'.*system.*', r'.*server.*', r'.*workstation.*', r'.*mobile.*', r'.*iot.*',
            r'.*classification.*', r'.*category.*', r'.*type.*', r'.*class.*'
        ],
        'content_indicators': [
            'system type', 'server type', 'workstation', 'mobile device', 'iot device',
            'system classification', 'device category', 'asset class', 'endpoint type'
        ],
        'value_patterns': [
            r'.*SERVER.*', r'.*WORKSTATION.*', r'.*MOBILE.*', r'.*IOT.*',
            r'.*LAPTOP.*', r'.*DESKTOP.*', r'.*TABLET.*'
        ],
        'weight': 0.6
    },
    'coverage_metrics': {
        'patterns': [
            r'.*coverage.*', r'.*visibility.*', r'.*percentage.*', r'.*percent.*', r'.*pct.*',
            r'.*count.*', r'.*total.*', r'.*baseline.*', r'.*gap.*'
        ],
        'content_indicators': [
            'coverage', 'visibility', 'percentage', 'percent', 'coverage percentage',
            'asset count', 'total assets', 'baseline', 'coverage gap', 'visibility gap'
        ],
        'value_patterns': [
            r'^\d{1,3}\.?\d*%?
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}, r'^\d+/\d+
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}, r'^\d+\s*of\s*\d+
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 0.8
    }
}
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z0-9\-]+\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
},
            r'^\w+\-\w+\-\d+
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$'
        ],
        'weight': 0.5
    }
}

FUZZY_MATCH_THRESHOLD = 0.8
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
CONTENT_ANALYSIS_SAMPLE_SIZE = 100
MIN_CONFIDENCE_SCORE = 0.6

QUERY_TEMPLATES = {
    'asset_coverage': """
        WITH baseline AS (
            SELECT DISTINCT COALESCE(LOWER(TRIM({primary_key})), LOWER(TRIM({fallback_key}))) as asset_id
            FROM `{table}` 
            WHERE COALESCE({primary_key}, {fallback_key}) IS NOT NULL
        )
        SELECT COUNT(DISTINCT asset_id) as total_assets FROM baseline
    """,
    'tool_coverage': """
        SELECT 
            '{tool_name}' as tool,
            COUNT(DISTINCT COALESCE(LOWER(TRIM({key_field})), '')) as covered_assets
        FROM `{table}`
        WHERE {key_field} IS NOT NULL AND TRIM({key_field}) != ''
    """,
    'data_quality': """
        SELECT 
            COUNT(*) as total_rows,
            COUNT({key_field}) as non_null_values,
            COUNT(DISTINCT {key_field}) as unique_values,
            ROUND(COUNT({key_field}) * 100.0 / COUNT(*), 2) as completeness_pct
        FROM `{table}`
    """
}