import os
from dotenv import load_dotenv

load_dotenv()

class QuantumConfig:
    ENV = "production"
    
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
        }
    }
