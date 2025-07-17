import pandas as pd
import numpy as np
import re
import json
from datetime import datetime

class SimpleVisibilityML:
    def __init__(self):
        self.patterns = {
            'host': [
                r'[a-zA-Z]+-[a-zA-Z]+-\d+',
                r'[A-Z]{3,4}-[A-Z]+-\d+',
                r'UUID-[A-Z0-9]+',
                r'[a-z]+svr|[a-z]+server'
            ],
            'network': [
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
                r'eth0|ens3|bond0|mgmt',
                r':\d{2,5}'
            ],
            'security': [
                r'defender|falcon|agent|sensor',
                r'v\d+\.\d+',
                r'online|client'
            ],
            'platform': [
                r'windows|linux|ubuntu|darwin|rhel|macos',
                r'20\d{2}|v\d{4}',
                r'x86_64|arm64'
            ],
            'org': [
                r'[a-z]+-[a-z]+-[a-z]+',
                r'corp|team|dept|eng'
            ],
            'location': [
                r'us-|eu-|ap-',
                r'datacenter|dc|facility',
                r'aws|gcp|azure'
            ],
            'domain': [
                r'\.[a-z]{2,4}$',
                r'\.local|\.corp|\.internal',
                r'api\.|www\.'
            ]
        }
        
    def classify(self, text, column_name):
        text = str(text).lower()
        scores = {}
        
        for category, pattern_list in self.patterns.items():
            score = 0
            for pattern in pattern_list:
                if re.search(pattern, text):
                    score += 0.3
            
            if category in column_name.lower():
                score += 0.4
            
            scores[category] = min(score, 1.0)
        
        if not scores or max(scores.values()) < 0.3:
            return 'unknown', 0.0
            
        best_category = max(scores, key=scores.get)
        confidence = scores[best_category]
        
        return best_category, confidence
    
    def analyze_table(self, table_name, table_data):
        if not table_data:
            return []
            
        df = pd.DataFrame(table_data)
        mappings = []
        
        visibility_map = {
            'host': 'host_coverage',
            'network': 'network_coverage',
            'security': 'security_coverage',
            'platform': 'platform_coverage',
            'org': 'organizational_coverage',
            'location': 'geographic_coverage',
            'domain': 'domain_coverage'
        }
        
        for column in df.columns:
            if df[column].dtype == 'object':
                samples = df[column].dropna().head(5).tolist()
                if samples:
                    predictions = []
                    confidences = []
                    
                    for sample in samples:
                        category, conf = self.classify(sample, column)
                        if category != 'unknown':
                            predictions.append(category)
                            confidences.append(conf)
                    
                    if predictions:
                        most_common = max(set(predictions), key=predictions.count)
                        avg_confidence = np.mean(confidences)
                        
                        if avg_confidence >= 0.5:
                            mappings.append({
                                'source': f'{table_name}.{column}',
                                'metric': visibility_map.get(most_common, 'unknown_coverage'),
                                'confidence': float(avg_confidence),
                                'pattern_type': most_common,
                                'samples': samples[:3]
                            })
        
        return mappings

ml_engine = SimpleVisibilityML()

# Test with sample data
test_data = {
    'enterprise_inventory': [
        {'hostname': 'srv-web-01', 'ip_addr': '192.168.1.100', 'os_build': 'ubuntu-20.04'},
        {'hostname': 'db-prod-mysql', 'ip_addr': '10.0.0.55', 'os_build': 'rhel8-optimized'},
        {'hostname': 'CORP-LAPTOP-01', 'ip_addr': '172.16.1.50', 'os_build': 'windows-2019'}
    ],
    'security_telemetry': [
        {'sensor_id': 'UUID-A7B9C3D1', 'agent_status': 'defender-online-v4.18', 'machine_name': 'WEB-FARM-02'},
        {'sensor_id': 'UUID-B8C2E4F5', 'agent_status': 'falcon-sensor-6.45', 'machine_name': 'DB-PROD-01'},
        {'sensor_id': 'UUID-C9D3F6A8', 'agent_status': 'tanium-client-7.4.6', 'machine_name': 'K8S-WORKER-05'}
    ]
}

print("🧠 SIMPLE ML VISIBILITY ANALYSIS")
print("="*50)

all_mappings = []
for table_name, table_data in test_data.items():
    mappings = ml_engine.analyze_table(table_name, table_data)
    all_mappings.extend(mappings)

all_mappings.sort(key=lambda x: x['confidence'], reverse=True)

for mapping in all_mappings:
    print(f"✓ {mapping['source']} -> {mapping['metric']}")
    print(f"  Confidence: {mapping['confidence']:.3f}")
    print(f"  Pattern: {mapping['pattern_type']}")
    print(f"  Samples: {mapping['samples']}")
    print()

print(f"Total mappings found: {len(all_mappings)}")
print(f"High confidence (>0.7): {len([m for m in all_mappings if m['confidence'] > 0.7])}")

with open('models/results.json', 'w') as f:
    json.dump({
        'mappings': all_mappings,
        'timestamp': datetime.now().isoformat(),
        'total_mappings': len(all_mappings)
    }, f, indent=2)

print("Results saved to models/results.json")
