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
            r'.*cmdb.*', r'.*inventory.*', r'.*baseline.*', r'.*hostname.*', r'.*servername.*'
        ],
        'content_indicators': [
            'hostname', 'fqdn', 'domain name', 'computer name', 'server name',
            'asset tag', 'device id', 'machine id', 'endpoint name', 'system name',
            'cmdb id', 'asset identifier', 'baseline asset', 'host identifier',
            'device identifier', 'machine identifier', 'system identifier'
        ],
        'value_patterns': [
            r'^[a-zA-Z0-9\-\.]+\.(com|local|corp|internal|net|org)$',
            r'^[a-zA-Z0-9\-]+\d+$',
            r'^[a-zA-Z]+\-[a-zA-Z0-9\-]+$',
            r'^\w+\-\w+\-\d+$',
            r'^[a-zA-Z0-9]{8,}$'
        ],
        'weight': 1.0
    },
    'network_identity': {
        'patterns': [
            r'.*ip.*', r'.*addr.*', r'.*network.*', r'.*inet.*', r'.*cidr.*',
            r'.*url.*', r'.*fqdn.*', r'.*domain.*', r'.*dns.*', r'.*vpc.*',
            r'.*zone.*', r'.*subnet.*', r'.*vlan.*', r'.*ipam.*', r'.*gateway.*'
        ],
        'content_indicators': [
            'ip address', 'network address', 'inet address', 'ipv4', 'ipv6',
            'url', 'fqdn', 'domain', 'dns name', 'vpc', 'network zone',
            'subnet', 'vlan', 'ipam', 'public ip', 'private ip', 'gateway',
            'router', 'switch', 'network device'
        ],
        'value_patterns': [
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$',
            r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$',
            r'^https?://.*',
            r'^[a-zA-Z0-9\-\.]+\.(com|local|corp|internal|net|org)$'
        ],
        'weight': 1.0
    },
    'security_tools': {
        'patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*edr.*',
            r'.*agent.*', r'.*sensor.*', r'.*endpoint.*', r'.*security.*',
            r'.*tanium.*', r'.*dlp.*', r'.*axonius.*', r'.*falcon.*',
            r'.*health.*', r'.*status.*', r'.*coverage.*', r'.*deployment.*',
            r'.*antivirus.*', r'.*firewall.*', r'.*ids.*', r'.*ips.*'
        ],
        'content_indicators': [
            'crowdstrike', 'chronicle', 'splunk', 'falcon', 'edr', 'antivirus',
            'agent', 'sensor', 'endpoint protection', 'security tool',
            'tanium', 'dlp', 'axonius', 'agent health', 'reporting status',
            'tool coverage', 'deployment status', 'security control',
            'firewall', 'intrusion detection', 'malware', 'threat detection'
        ],
        'value_patterns': [
            r'.*crowdstrike.*', r'.*chronicle.*', r'.*splunk.*', r'.*falcon.*',
            r'.*agent.*', r'.*sensor.*', r'.*edr.*', r'.*tanium.*', r'.*dlp.*',
            r'.*healthy.*', r'.*online.*', r'.*active.*', r'.*deployed.*',
            r'.*installed.*', r'.*running.*', r'.*enabled.*'
        ],
        'weight': 0.9
    },
    'log_sources': {
        'patterns': [
            r'.*log.*', r'.*event.*', r'.*audit.*', r'.*source.*', r'.*type.*',
            r'.*volume.*', r'.*ingest.*', r'.*collection.*', r'.*siem.*',
            r'.*activity.*', r'.*trace.*', r'.*record.*'
        ],
        'content_indicators': [
            'log type', 'event type', 'source type', 'audit log', 'security event',
            'log volume', 'ingest volume', 'log collection', 'siem data',
            'firewall log', 'dns log', 'authentication log', 'access log',
            'system log', 'application log', 'security log', 'network log'
        ],
        'value_patterns': [
            r'.*_LOG$', r'.*_EVENT$', r'.*_AUDIT$', r'FIREWALL.*', r'DNS.*',
            r'AUTH.*', r'ACCESS.*', r'SECURITY.*', r'NETWORK.*', r'SYSTEM.*',
            r'APPLICATION.*', r'WINDOWS.*', r'LINUX.*'
        ],
        'weight': 0.8
    },
    'geographic_data': {
        'patterns': [
            r'.*country.*', r'.*region.*', r'.*location.*', r'.*site.*', r'.*zone.*',
            r'.*datacenter.*', r'.*office.*', r'.*facility.*', r'.*geo.*',
            r'.*city.*', r'.*state.*', r'.*continent.*'
        ],
        'content_indicators': [
            'country', 'region', 'location', 'site', 'datacenter', 'zone',
            'office', 'facility', 'geographic', 'geolocation', 'city',
            'state', 'province', 'continent', 'timezone'
        ],
        'value_patterns': [
            r'^[A-Z]{2}$', r'^[A-Z]{3}$', r'.*US.*', r'.*EAST.*', r'.*WEST.*',
            r'.*NORTH.*', r'.*SOUTH.*', r'.*DC.*', r'.*SITE.*', r'.*REGION.*'
        ],
        'weight': 0.6
    },
    'business_context': {
        'patterns': [
            r'.*business.*', r'.*unit.*', r'.*application.*', r'.*service.*', r'.*owner.*',
            r'.*bu.*', r'.*dept.*', r'.*division.*', r'.*team.*', r'.*org.*'
        ],
        'content_indicators': [
            'business unit', 'application', 'service', 'owner', 'department',
            'bu', 'division', 'team', 'application owner', 'service owner',
            'organization', 'cost center', 'project', 'portfolio'
        ],
        'value_patterns': [
            r'.*_APP$', r'.*_SERVICE$', r'.*_BU$', r'.*_DEPT$', r'.*_ORG$'
        ],
        'weight': 0.5
    },
    'infrastructure_type': {
        'patterns': [
            r'.*on.*prem.*', r'.*cloud.*', r'.*saas.*', r'.*api.*', r'.*infrastructure.*',
            r'.*platform.*', r'.*environment.*', r'.*tier.*', r'.*category.*',
            r'.*virtual.*', r'.*physical.*', r'.*container.*'
        ],
        'content_indicators': [
            'on-prem', 'on premise', 'cloud', 'saas', 'api', 'infrastructure type',
            'platform type', 'environment', 'tier', 'category', 'deployment type',
            'virtual machine', 'container', 'kubernetes', 'docker', 'serverless'
        ],
        'value_patterns': [
            r'.*ON.*PREM.*', r'.*CLOUD.*', r'.*SAAS.*', r'.*API.*',
            r'.*HYBRID.*', r'.*AZURE.*', r'.*AWS.*', r'.*GCP.*', r'.*VIRTUAL.*',
            r'.*PHYSICAL.*', r'.*CONTAINER.*'
        ],
        'weight': 0.7
    },
    'system_classification': {
        'patterns': [
            r'.*system.*', r'.*server.*', r'.*workstation.*', r'.*mobile.*', r'.*iot.*',
            r'.*classification.*', r'.*category.*', r'.*type.*', r'.*class.*',
            r'.*critical.*', r'.*production.*', r'.*development.*'
        ],
        'content_indicators': [
            'system type', 'server type', 'workstation', 'mobile device', 'iot device',
            'system classification', 'device category', 'asset class', 'endpoint type',
            'critical system', 'production system', 'development system'
        ],
        'value_patterns': [
            r'.*SERVER.*', r'.*WORKSTATION.*', r'.*MOBILE.*', r'.*IOT.*',
            r'.*LAPTOP.*', r'.*DESKTOP.*', r'.*TABLET.*', r'.*CRITICAL.*',
            r'.*PRODUCTION.*', r'.*DEV.*', r'.*TEST.*'
        ],
        'weight': 0.6
    },
    'coverage_metrics': {
        'patterns': [
            r'.*coverage.*', r'.*visibility.*', r'.*percentage.*', r'.*percent.*', r'.*pct.*',
            r'.*count.*', r'.*total.*', r'.*baseline.*', r'.*gap.*', r'.*score.*'
        ],
        'content_indicators': [
            'coverage', 'visibility', 'percentage', 'percent', 'coverage percentage',
            'asset count', 'total assets', 'baseline', 'coverage gap', 'visibility gap',
            'compliance score', 'security score', 'risk score'
        ],
        'value_patterns': [
            r'^\d{1,3}\.?\d*%?$', r'^\d+/\d+$', r'^\d+\s*of\s*\d+$',
            r'^\d+\.\d+$', r'^\d+$'
        ],
        'weight': 0.8
    }
}

FUZZY_MATCH_THRESHOLD = 0.75
SEMANTIC_SIMILARITY_THRESHOLD = 0.6
MIN_CONFIDENCE_SCORE = 0.4

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