import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
import re
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockBigQueryScanner:
    def __init__(self):
        logger.info("Mock BigQuery scanner initialized")
        
        try:
            from neural_engine_brilliant import BrilliantVisibilityMapper
            self.ml_mapper = BrilliantVisibilityMapper()
            logger.info("Using BrilliantVisibilityMapper")
        except ImportError:
            logger.info("Using built-in pattern matching")
            self.ml_mapper = None
    
    def generate_mock_data(self):
        """Generate realistic mock enterprise data"""
        mock_tables = {
            'enterprise_inventory': {
                'hostname': [
                    'srv-web-01', 'srv-web-02', 'srv-api-01', 'db-prod-mysql-01',
                    'CORP-LAPTOP-0001', 'CORP-LAPTOP-0002', 'WIN-SERVER-001',
                    'k8s-worker-01', 'k8s-worker-02', 'analytics-node-01',
                    'mailsvr-exch-01', 'build-agent-07', 'proxy-01'
                ],
                'ip_address': [
                    '192.168.1.100', '192.168.1.101', '10.0.0.55', '172.16.1.50',
                    '192.168.2.100', '192.168.2.101', '10.1.1.10', '172.31.0.100',
                    '10.244.1.5', '10.244.1.6', '172.20.1.50', '192.168.10.5', '203.0.113.1'
                ],
                'os_version': [
                    'Windows-Server-2019-Standard', 'ubuntu-20.04-lts', 'rhel8-enterprise',
                    'Windows-NT-10.0.19041-x64', 'macos-monterey-12.6', 'debian-11-bullseye',
                    'container-os-optimized-v2023', 'Windows-Server-2022-Datacenter',
                    'Linux-5.4.0-ubuntu-x86_64', 'Darwin-21.6.0-arm64', 'centos7-legacy',
                    'alpine-3.18-minimal', 'fedora-38-workstation'
                ],
                'department': [
                    'engineering-team', 'finance-dept', 'hr-division', 'marketing-group',
                    'it-operations', 'corp-it-messaging', 'software-eng-platform',
                    'data-intelligence-team', 'executive-staff-support', 'platform-reliability-eng',
                    'network-sec-operations', 'cyber-threat-intel', 'sales-enterprise'
                ],
                'location': [
                    'us-east-1', 'datacenter-nyc', 'facility-london', 'dc-chicago-tier3',
                    'aws-us-west-2', 'gcp-europe-west1', 'azure-eastus', 'office-sfo-soma',
                    'colo-frankfurt-suite5', 'cloud-aws-use1', 'distributed-mobile',
                    'remote-workforce', 'field-operations'
                ]
            },
            'security_telemetry': {
                'agent_id': [
                    'UUID-A7B9C3D1', 'UUID-B8C2E4F5', 'UUID-C9D3F6A8', 'UUID-D4E7G9B2',
                    'GUID-12345678', 'GUID-87654321', 'REF-001947', 'REF-002103',
                    'ASSET-12345', 'DEVICE-98765', 'ENDPOINT-001', 'SENSOR-456'
                ],
                'agent_status': [
                    'defender-online-v4.18', 'crowdstrike-falcon-6.45', 'tanium-client-7.4.6',
                    'sentinelone-agent-22.3.1', 'qualys-agent-v4.8', 'mde-sensor-v2.1',
                    's1-agent-v22.3', 'falco-sensor-v0.35', 'amp-connector-v1.8',
                    'jamf-protect-2.8.1', 'symantec-sep-14.3', 'mcafee-epo-5.10'
                ],
                'machine_name': [
                    'WEB-FARM-01', 'WEB-FARM-02', 'DB-PROD-01', 'K8S-WORKER-05',
                    'MOBILE-MGMT-99', 'FW-PERIMETER-01', 'MAIL-EXCHANGE-01',
                    'BUILD-AGENT-03', 'ANALYTICS-NODE-02', 'PROXY-SERVER-01',
                    'FILE-SERVER-01', 'BACKUP-SERVER-01'
                ],
                'security_score': [85, 92, 78, 95, 88, 91, 76, 89, 93, 82, 87, 90]
            },
            'network_flow_logs': {
                'source_ip': [
                    '192.168.1.100', '10.0.0.55', '172.16.1.50', '192.168.2.100',
                    '10.1.1.10', '172.31.0.100', '10.244.1.5', '172.20.1.50',
                    '203.0.113.1', '198.51.100.1', '192.168.10.5', '10.0.1.200'
                ],
                'dest_ip': [
                    '192.168.1.101', '10.0.0.56', '172.16.1.51', '192.168.2.101',
                    '10.1.1.11', '172.31.0.101', '10.244.1.6', '172.20.1.51',
                    '203.0.113.2', '198.51.100.2', '192.168.10.6', '10.0.1.201'
                ],
                'protocol': ['TCP', 'UDP', 'HTTP', 'HTTPS', 'SSH', 'FTP', 'DNS', 'SMTP', 'SNMP', 'ICMP', 'RDP', 'LDAP'],
                'port': [80, 443, 22, 21, 53, 25, 161, 3389, 389, 3306, 5432, 6379]
            },
            'application_logs': {
                'service_name': [
                    'api.company.com', 'www.company.com', 'mail.company.com',
                    'internal.corp.local', 'kubernetes.default.svc.cluster.local',
                    'jenkins.build.corp', 'monitoring.ops.corp', 'wiki.corp.internal',
                    'files.corp.local', 'vpn.company.com', 'docs.company.com', 'support.company.com'
                ],
                'user_agent': [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'curl/7.68.0',
                    'Python-urllib/3.8', 'Go-http-client/1.1', 'Java/11.0.2',
                    'PostmanRuntime/7.28.0', 'Chrome/91.0.4472.124', 'Firefox/89.0',
                    'Safari/14.1.1', 'Edge/91.0.864.59', 'kubectl/v1.21.0', 'helm/v3.6.0'
                ],
                'status_code': [200, 201, 301, 302, 400, 401, 403, 404, 500, 502, 503, 504],
                'response_time': [45, 120, 89, 234, 567, 23, 156, 78, 345, 67, 189, 456]
            }
        }
        
        # Create DataFrames with realistic sizes
        tables = {}
        for table_name, columns in mock_tables.items():
            data = {}
            max_len = max(len(values) for values in columns.values())
            
            for col_name, values in columns.items():
                # Repeat values to create realistic dataset sizes
                expanded_values = []
                for i in range(200):  # 200 rows per table
                    expanded_values.append(random.choice(values))
                data[col_name] = expanded_values
            
            tables[table_name] = pd.DataFrame(data)
        
        return tables
    
    def classify_column(self, data_series, column_name):
        clean_data = data_series.dropna().astype(str).head(100).tolist()
        if not clean_data:
            return None
            
        patterns = {
            'host_coverage': [
                lambda x: bool(re.search(r'[a-zA-Z]+-[a-zA-Z]+-\d+', str(x))),
                lambda x: bool(re.search(r'srv|host|server|node|machine', str(x).lower())),
                lambda x: bool(re.search(r'[A-Z]{3,}-[A-Z]+-\d+', str(x))),
                lambda x: 'host' in column_name.lower() or 'hostname' in column_name.lower() or 'machine' in column_name.lower()
            ],
            'network_coverage': [
                lambda x: bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', str(x))),
                lambda x: bool(re.search(r'eth0|ens3|bond0|wifi', str(x).lower())),
                lambda x: ':' in str(x) and any(c.isdigit() for c in str(x)),
                lambda x: 'ip' in column_name.lower() or 'addr' in column_name.lower() or 'port' in column_name.lower()
            ],
            'security_coverage': [
                lambda x: bool(re.search(r'agent|defender|security|antivirus|falcon|tanium', str(x).lower())),
                lambda x: bool(re.search(r'v\d+\.\d+', str(x))),
                lambda x: bool(re.search(r'UUID-|GUID-|sensor|endpoint', str(x))),
                lambda x: 'agent' in column_name.lower() or 'security' in column_name.lower()
            ],
            'platform_coverage': [
                lambda x: bool(re.search(r'windows|linux|ubuntu|darwin|macos|rhel|centos', str(x).lower())),
                lambda x: bool(re.search(r'build|version|os', str(x).lower())),
                lambda x: 'os' in column_name.lower() or 'platform' in column_name.lower() or 'version' in column_name.lower()
            ],
            'organizational_coverage': [
                lambda x: bool(re.search(r'team|dept|division|group|unit', str(x).lower())),
                lambda x: bool(re.search(r'engineering|finance|hr|marketing|sales', str(x).lower())),
                lambda x: 'dept' in column_name.lower() or 'team' in column_name.lower() or 'org' in column_name.lower()
            ],
            'geographic_coverage': [
                lambda x: bool(re.search(r'us-|eu-|datacenter|facility|office', str(x).lower())),
                lambda x: bool(re.search(r'nyc|london|chicago|aws|gcp|azure', str(x).lower())),
                lambda x: 'location' in column_name.lower() or 'region' in column_name.lower() or 'site' in column_name.lower()
            ],
            'domain_coverage': [
                lambda x: bool(re.search(r'\.[a-z]{2,4}$', str(x).lower())),
                lambda x: bool(re.search(r'\.local|\.corp|\.internal|\.com', str(x).lower())),
                lambda x: bool(re.search(r'api\.|www\.|mail\.|service', str(x).lower())),
                lambda x: 'domain' in column_name.lower() or 'url' in column_name.lower() or 'service' in column_name.lower()
            ]
        }
        
        for metric, checks in patterns.items():
            score = 0
            for check in checks:
                try:
                    matches = sum(1 for sample in clean_data[:50] if check(sample))
                    if matches > len(clean_data) * 0.1:
                        score += 1
                except:
                    pass
            
            if score >= 2:
                return {
                    'metric': metric,
                    'confidence': min(score / len(checks) * 1.2, 1.0),
                    'samples': clean_data[:3],
                    'pattern_type': 'mock_pattern'
                }
        
        return None
    
    def mock_scan_project(self, project_id='mock-project-123'):
        logger.info(f"Starting mock BigQuery scan for project: {project_id}")
        logger.info("Simulating connection to BigQuery...")
        time.sleep(1)
        
        logger.info("Discovering datasets...")
        mock_datasets = ['enterprise_data', 'security_logs', 'network_analytics', 'application_monitoring']
        
        all_mappings = []
        mock_tables = self.generate_mock_data()
        
        for dataset_name in mock_datasets:
            logger.info(f"Scanning dataset: {dataset_name}")
            
            for table_name, df in mock_tables.items():
                logger.info(f"Analyzing table: {dataset_name}.{table_name}")
                time.sleep(0.2)  # Simulate processing time
                
                mappings = []
                
                if self.ml_mapper:
                    try:
                        tables_dict = {table_name: df}
                        ml_mappings = self.ml_mapper.discover_mappings_with_brilliance(tables_dict)
                        
                        for mapping in ml_mappings:
                            mappings.append({
                                'project': project_id,
                                'dataset': dataset_name,
                                'table': table_name,
                                'column': mapping.column_name,
                                'source': f"{project_id}.{dataset_name}.{table_name}.{mapping.column_name}",
                                'metric': mapping.target_metric,
                                'confidence': mapping.entanglement_strength,
                                'pattern_type': mapping.reasoning_graph.get('pattern_type', 'ml_detected'),
                                'detection_methods': mapping.reasoning_graph.get('detection_methods', []),
                                'sample_values': mapping.reasoning_graph.get('sample_values', []),
                                'full_table_name': f"{project_id}.{dataset_name}.{table_name}"
                            })
                    except Exception as ml_error:
                        logger.error(f"ML analysis failed: {ml_error}")
                
                if not mappings:
                    for column_name in df.columns:
                        result = self.classify_column(df[column_name], column_name)
                        if result:
                            mappings.append({
                                'project': project_id,
                                'dataset': dataset_name,
                                'table': table_name,
                                'column': column_name,
                                'source': f"{project_id}.{dataset_name}.{table_name}.{column_name}",
                                'metric': result['metric'],
                                'confidence': result['confidence'],
                                'pattern_type': result['pattern_type'],
                                'detection_methods': ['mock_classification'],
                                'sample_values': result['samples'],
                                'full_table_name': f"{project_id}.{dataset_name}.{table_name}"
                            })
                
                all_mappings.extend(mappings)
                logger.info(f"Found {len(mappings)} mappings for {table_name}")
        
        logger.info(f"Mock scan completed! Total mappings found: {len(all_mappings)}")
        return all_mappings
    
    def save_results(self, mappings, project_id):
        os.makedirs('outputs', exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        high_confidence_mappings = [m for m in mappings if m['confidence'] > 0.7]
        
        report = {
            'scan_metadata': {
                'timestamp': timestamp,
                'project_id': project_id,
                'scan_type': 'MOCK_SCAN',
                'total_mappings': len(mappings),
                'high_confidence_mappings': len(high_confidence_mappings),
                'datasets_scanned': len(set(f"{m['dataset']}" for m in mappings)),
                'tables_scanned': len(set(f"{m['dataset']}.{m['table']}" for m in mappings)),
                'ml_system': 'BrilliantVisibilityMapper' if self.ml_mapper else 'MockPatternMatcher'
            },
            'mappings': sorted(mappings, key=lambda x: x['confidence'], reverse=True),
            'summary_by_metric': self.summarize_by_metric(mappings),
            'summary_by_dataset': self.summarize_by_dataset(mappings)
        }
        
        filename = f'outputs/mock_bigquery_scan_{project_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        
        self.print_summary(report)
        return filename
    
    def summarize_by_metric(self, mappings):
        metrics = {}
        for mapping in mappings:
            metric = mapping['metric']
            if metric not in metrics:
                metrics[metric] = []
            metrics[metric].append(mapping)
        
        return {metric: {
            'count': len(mappings),
            'avg_confidence': sum(m['confidence'] for m in mappings) / len(mappings),
            'datasets': list(set(m['dataset'] for m in mappings)),
            'best_sources': sorted(mappings, key=lambda x: x['confidence'], reverse=True)[:3]
        } for metric, mappings in metrics.items()}
    
    def summarize_by_dataset(self, mappings):
        datasets = {}
        for mapping in mappings:
            dataset = mapping['dataset']
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append(mapping)
        
        return {dataset: {
            'total_mappings': len(mappings),
            'high_confidence': len([m for m in mappings if m['confidence'] > 0.7]),
            'tables': len(set(m['table'] for m in mappings)),
            'metrics': list(set(m['metric'] for m in mappings))
        } for dataset, mappings in datasets.items()}
    
    def print_summary(self, report):
        print("\n" + "="*80)
        print("MOCK BIGQUERY SCAN RESULTS")
        print("="*80)
        
        metadata = report['scan_metadata']
        print(f"Project: {metadata['project_id']} (MOCK)")
        print(f"ML System: {metadata['ml_system']}")
        print(f"Datasets scanned: {metadata['datasets_scanned']}")
        print(f"Tables scanned: {metadata['tables_scanned']}")
        print(f"Total mappings: {metadata['total_mappings']}")
        print(f"High confidence mappings: {metadata['high_confidence_mappings']}")
        
        print("\nVISIBILITY METRICS DISCOVERED:")
        for metric, data in report['summary_by_metric'].items():
            print(f"  {metric}: {data['count']} sources (avg confidence: {data['avg_confidence']:.3f})")
        
        print("\nTOP MAPPINGS:")
        for mapping in report['mappings'][:15]:
            print(f"  {mapping['source']} -> {mapping['metric']} ({mapping['confidence']:.3f})")
            print(f"    Pattern: {mapping['pattern_type']}")
            print(f"    Samples: {mapping['sample_values'][:2]}")
            print()
        
        print("="*80)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Mock BigQuery scanner for testing')
    parser.add_argument('--project', default='mock-enterprise-project', help='Mock project ID')
    args = parser.parse_args()
    
    scanner = MockBigQueryScanner()
    mappings = scanner.mock_scan_project(args.project)
    filename = scanner.save_results(mappings, args.project)
    
    print(f"\nMock scan complete! Results saved to: {filename}")
    print("\nThis was a MOCK scan with simulated data.")
    print("To run against real BigQuery, use the real scanner with proper authentication.")

if __name__ == "__main__":
    main()
