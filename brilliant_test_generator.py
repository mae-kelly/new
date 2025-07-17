import json
import random
import uuid
from datetime import datetime, timedelta

class BrilliantTestGenerator:
    def __init__(self):
        self.companies = ['TechCorp', 'DataSys', 'CloudNet', 'SecureIT', 'InnovateLab']
        self.locations = ['NYC', 'SFO', 'LON', 'TOK', 'SYD', 'FRA', 'SIN']
        self.departments = ['eng', 'ops', 'sec', 'data', 'ml', 'platform', 'infra']
        
    def generate_realistic_dataset(self, size=1000):
        dataset = []
        
        for i in range(size):
            company = random.choice(self.companies).lower()
            dept = random.choice(self.departments)
            location = random.choice(self.locations).lower()
            
            # Generate realistic enterprise data
            row = {
                'hostname': f"srv-{dept}-{random.randint(1,99):02d}.{company}.com",
                'ip_address': f"10.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
                'mac_address': ':'.join(['%02x' % random.randint(0, 255) for _ in range(6)]),
                'asset_id': f"ASSET-{random.randint(100000,999999)}",
                'serial_number': f"SN{random.randint(10000000,99999999)}",
                'os_version': random.choice([
                    'Windows-Server-2019-Standard',
                    'ubuntu-20.04-lts-server',
                    'rhel8-enterprise-v2.1',
                    'macos-monterey-12.6',
                    'debian-11-bullseye'
                ]),
                'security_agent': random.choice([
                    'crowdstrike-falcon-7.05',
                    'microsoft-defender-v4.18',
                    'sentinelone-agent-23.1.2',
                    'qualys-vmdr-v4.9',
                    'tanium-endpoint-v7.4.8'
                ]),
                'department': f"{company}-{dept}-team",
                'location': f"datacenter-{location}-primary",
                'domain_name': f"{random.choice(['api', 'web', 'db', 'cache'])}.{dept}.{company}.internal",
                'last_seen': (datetime.now() - timedelta(hours=random.randint(0,72))).isoformat(),
                'compliance_status': random.choice(['compliant', 'needs_update', 'critical']),
                'vulnerability_score': random.randint(0, 100),
                'network_segment': f"vlan-{random.randint(100,999)}-{dept}",
                'backup_status': random.choice(['backed_up', 'backup_failed', 'no_backup']),
                'patch_level': f"patch-{random.randint(2023,2025)}-{random.randint(1,12):02d}",
                'encryption_status': random.choice(['encrypted', 'unencrypted', 'partial']),
                'access_level': random.choice(['public', 'internal', 'restricted', 'confidential']),
                'service_tier': random.choice(['production', 'staging', 'development', 'testing'])
            }
            dataset.append(row)
        
        return dataset
    
    def save_test_datasets
