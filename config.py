import os
from dotenv import load_dotenv

load_dotenv()

class QuantumConfig:
    ENV = "test" if os.path.exists("test_mode.flag") else "prod"
    
    ADVANCED_PATTERN_MATRICES = {
        'host_identifiers': {
            'metamorphic_patterns': [
                r'(?i)([a-z]{2,4}[\-_]?)?(srv|server|host|node|machine|system|device|endpoint|workstation|computer|pc|ws|wks|laptop|desktop|vm|instance)[\-_]?([a-z]{0,8}[\-_]?)?(\d{1,6})?(\.[a-z]{2,8})*',
                r'(?i)[a-z]{1,63}[\-_\.]{1,3}[a-z]{1,63}[\-_\.]{0,3}([a-z]{0,63})?(\.(local|corp|internal|company))?',
                r'(?i)[0-9a-f]{8}[\-]?[0-9a-f]{4}[\-]?[0-9a-f]{4}[\-]?[0-9a-f]{4}[\-]?[0-9a-f]{12}',
                r'(?i)(mailsvr|build-agent|analytics-node|exec-laptop|orchestrator|perimeter-fw)',
                r'(?i)[A-Z]{2,4}-[A-Z]{2,10}-\d{2,6}'
            ],
            'obfuscated_indicators': [
                'asset_reference', 'primary_fqdn', 'machine_identity', 'device_uuid', 'endpoint_guid',
                'system_identifier', 'hardware_fingerprint', 'unique_id', 'principal_name',
                'subject_identifier', 'resource_name', 'target_system', 'source_host', 'sensor_id'
            ],
            'contextual_embeddings': [
                'unique system designation in enterprise infrastructure',
                'computing device with network presence and identity',
                'managed endpoint with organizational assignment'
            ]
        },
        'network_entities': {
            'metamorphic_patterns': [
                r'(?i)\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(/[0-9]{1,2})?\b',
                r'(?i)\b([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
                r'(?i)(([a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?\.)*[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?)',
                r'(?i)(eth0|ens3|bond0|mgmt|wifi|vpn|cni|flannel):[0-9\.:]+',
                r'(?i)[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}:[0-9]{1,5}'
            ],
            'obfuscated_indicators': [
                'origin_addr', 'destination_addr', 'client_source', 'interface_config', 'network_config',
                'connection_point', 'network_node', 'routing_destination', 'packet_source'
            ],
            'contextual_embeddings': [
                'network layer addressing for packet routing',
                'internet protocol identifier for communication'
            ]
        },
        'platform_systems': {
            'metamorphic_patterns': [
                r'(?i)(microsoft\s+)?(windows\s*)?(server\s*)?(20(08|12|16|19|22)|10|11|xp|vista|7|8)',
                r'(?i)(red\s*hat\s*)?(enterprise\s*)?(linux|rhel)(\s*[0-9](\.[0-9])*)?',
                r'(?i)(ubuntu|debian|centos|fedora|suse)(\s*[0-9]{1,2}(\.[0-9]{1,2})*)?',
                r'(?i)(unix|aix|solaris|hp\-ux|freebsd)(\s*[0-9](\.[0-9])*)?',
                r'(?i)(mac\s*os|macos|osx|darwin)',
                r'(?i)(vmware|esxi|hyper\-v|citrix|xen)',
                r'(?i)(win2019|ubuntu-20|rhel8|macos-monterey|container-os|cisco-asa)'
            ],
            'obfuscated_indicators': [
                'platform_build', 'os_build', 'compute_spec', 'runtime_system',
                'system_kernel', 'platform_version', 'os_family', 'system_architecture'
            ],
            'contextual_embeddings': [
                'operating system managing computational resources',
                'platform software providing execution environment'
            ]
        },
        'security_systems': {
            'metamorphic_patterns': [
                r'(?i)(crowdstrike|falcon)(\s*(sensor|agent|endpoint))?',
                r'(?i)(microsoft\s*)?(defender|windows\s*defender)',
                r'(?i)(tanium|edr|endpoint\s*detection)',
                r'(?i)(sentinelone|symantec|mcafee|kaspersky)',
                r'(?i)(qualys|rapid7|tenable|nessus)',
                r'(?i)(splunk|elastic|datadog)',
                r'(?i)(mde-sensor|qualys-agent|tanium-client|s1-agent|falco-sensor|amp-connector)',
                r'(?i)(defender-online|crowdstrike-falcon|jamf-protect|cisco-amp)'
            ],
            'obfuscated_indicators': [
                'protection_agent', 'agent_status', 'scanner_engine', 'monitoring_signals',
                'security_posture', 'threat_detection', 'defense_system'
            ],
            'contextual_embeddings': [
                'endpoint protection platform with threat detection',
                'security monitoring system with behavioral analysis'
            ]
        },
        'organizational_units': {
            'metamorphic_patterns': [
                r'(?i)(information\s*technology|it(\s*dept)?|technology\s*services)',
                r'(?i)(human\s*resources|hr(\s*dept)?|people\s*operations)',
                r'(?i)(finance|accounting|financial\s*services)',
                r'(?i)(engineering|software\s*engineering|development|devops)',
                r'(?i)(marketing|sales|business\s*development)',
                r'(?i)(operations|ops|infrastructure|network\s*operations)',
                r'(?i)(corp-it|software-eng|data-intelligence|executive-staff|platform-reliability|network-sec)'
            ],
            'obfuscated_indicators': [
                'dept_assignment', 'billing_tags', 'asset_criticality', 'business_division',
                'organizational_hierarchy', 'reporting_structure', 'team_assignment'
            ],
            'contextual_embeddings': [
                'organizational division with specific business function',
                'departmental structure within corporate hierarchy'
            ]
        },
        'geographic_locations': {
            'metamorphic_patterns': [
                r'(?i)(data\s*center|datacenter|dc)[\s\-_]*([a-z]{2,15})',
                r'(?i)(office|facility|site|location|campus)[\s\-_]*([a-z]{2,15})',
                r'(?i)(cloud|aws|azure|gcp)[\s\-_]*(us|eu|ap|ca)[\s\-_]*(east|west|central|north|south)',
                r'(?i)(americas|emea|apac|asia|europe|north\s*america)',
                r'(?i)(facility-nyc|cloud-aws-use1|dc-chicago|distributed-mobile|gcp-central1|colo-london)',
                r'(?i)(us-east-1|us-west-2|eu-central-1|west-europe|us-central1)'
            ],
            'obfuscated_indicators': [
                'facility_code', 'region_az', 'geo_location', 'site_code',
                'geographic_zone', 'location_identifier', 'deployment_region'
            ],
            'contextual_embeddings': [
                'geographic location with physical infrastructure presence',
                'regional deployment zone for distributed systems'
            ]
        },
        'domain_entities': {
            'metamorphic_patterns': [
                r'(?i)((https?|ftp|ldap|ssh)://)?([a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?\.)*[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?\.[a-z]{2,}',
                r'(?i)[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?\.(local|corp|internal|company|domain)',
                r'(?i)(mail|www|api|cdn|vpn|proxy|gateway)\.[a-z0-9\-\.]+',
                r'(?i)[a-z0-9\-]+\.(cluster\.local|svc|amazonaws\.com|azure\.com)'
            ],
            'obfuscated_indicators': [
                'fqdn_requested', 'service_name', 'target_system_fqdn', 'primary_fqdn',
                'resource_identifier', 'application_endpoint', 'api_gateway'
            ],
            'contextual_embeddings': [
                'hierarchical domain name system identifier',
                'web resource locator with protocol specification'
            ]
        }
    }
