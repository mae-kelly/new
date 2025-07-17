import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import joblib
import json
from datetime import datetime

class InsaneAccuracyML:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,3))
        self.model = None
        self.feature_names = []
        
    def generate_massive_training_data(self):
        patterns = {
            'host_identifiers': [
                # Traditional patterns
                'srv-web-01', 'srv-web-02', 'srv-web-03', 'srv-api-01', 'srv-api-02',
                'db-prod-mysql-01', 'db-prod-mysql-02', 'db-test-postgres-01',
                'app-server-01', 'app-server-02', 'app-server-03',
                'mail-exchange-01', 'mail-exchange-02', 'mail-relay-01',
                'file-server-01', 'file-server-02', 'backup-server-01',
                'build-agent-01', 'build-agent-02', 'build-agent-03',
                'analytics-node-01', 'analytics-node-02', 'analytics-worker-01',
                'k8s-master-01', 'k8s-worker-01', 'k8s-worker-02', 'k8s-worker-03',
                'web-farm-01', 'web-farm-02', 'web-farm-03',
                'load-balancer-01', 'load-balancer-02', 'proxy-01', 'proxy-02',
                
                # Corporate patterns
                'CORP-LAPTOP-0001', 'CORP-LAPTOP-0002', 'CORP-DESKTOP-0001',
                'CORP-SERVER-0001', 'CORP-SERVER-0002', 'CORP-WORKSTATION-0001',
                'WIN-LAPTOP-001', 'WIN-LAPTOP-002', 'WIN-SERVER-001',
                'MAC-LAPTOP-001', 'MAC-LAPTOP-002', 'MAC-DESKTOP-001',
                'LIN-SERVER-001', 'LIN-SERVER-002', 'LIN-WORKSTATION-001',
                
                # Cloud patterns
                'aws-ec2-web-01', 'aws-ec2-db-01', 'aws-ec2-app-01',
                'gcp-compute-web-01', 'gcp-compute-api-01', 'gcp-k8s-node-01',
                'azure-vm-web-01', 'azure-vm-db-01', 'azure-aks-node-01',
                
                # UUID patterns
                'UUID-A7B9C3D1', 'UUID-B8C2E4F5', 'UUID-C9D3F6A8', 'UUID-D4E7G9B2',
                'GUID-12345678', 'GUID-87654321', 'GUID-ABCDEFGH',
                
                # Reference patterns
                'REF-001947', 'REF-002103', 'REF-002456', 'REF-003891',
                'ASSET-001', 'ASSET-002', 'ASSET-003', 'DEVICE-001',
                
                # Container patterns
                'docker-container-web-123', 'docker-container-api-456',
                'pod-web-app-789', 'pod-database-012', 'pod-redis-345'
            ],
            
            'network_entities': [
                # IPv4 addresses
                '192.168.1.100', '192.168.1.101', '192.168.2.50', '192.168.10.1',
                '10.0.0.1', '10.0.0.10', '10.1.1.1', '10.10.10.10',
                '172.16.0.1', '172.16.1.100', '172.31.0.55', '172.20.1.50',
                '203.0.113.1', '203.0.113.50', '198.51.100.1', '198.51.100.100',
                
                # IPv4 with ports
                '192.168.1.100:80', '192.168.1.100:443', '192.168.1.100:22',
                '10.0.0.1:3306', '10.0.0.1:5432', '10.0.0.1:6379',
                '172.16.1.50:8080', '172.16.1.50:9000', '172.16.1.50:5000',
                
                # IPv4 with subnets
                '192.168.1.0/24', '10.0.0.0/16', '172.16.0.0/12',
                '192.168.0.0/16', '10.10.0.0/24', '172.31.0.0/16',
                
                # IPv6 addresses
                '2001:db8::1', '2001:db8:85a3::8a2e:370:7334', 'fe80::1',
                '::1', '2001:4860:4860::8888', '2001:4860:4860::8844',
                
                # Interface configurations
                'eth0:192.168.1.50', 'ens3:10.1.5.88', 'bond0:172.31.5.100',
                'wlan0:192.168.1.200', 'lo:127.0.0.1', 'docker0:172.17.0.1',
                'br0:192.168.100.1', 'tun0:10.8.0.1', 'tap0:10.9.0.1',
                
                # Network interface names
                'eth0', 'eth1', 'ens3', 'ens33', 'bond0', 'bond1',
                'wlan0', 'wifi0', 'mgmt', 'backup', 'storage',
                'cni0', 'flannel.1', 'weave', 'calico', 'cilium0'
            ],
            
            'security_systems': [
                # Endpoint agents
                'defender-online-v4.18', 'defender-offline-v4.17', 'defender-updating-v4.19',
                'crowdstrike-falcon-6.45', 'crowdstrike-falcon-6.44', 'crowdstrike-falcon-6.46',
                'sentinelone-agent-22.3.1', 'sentinelone-agent-22.2.5', 'sentinelone-agent-23.1.0',
                'tanium-client-7.4.6', 'tanium-client-7.4.5', 'tanium-client-7.5.0',
                'qualys-agent-v4.8', 'qualys-agent-v4.7', 'qualys-agent-v4.9',
                'symantec-sep-14.3', 'mcafee-epo-5.10', 'kaspersky-kes-11.5',
                
                # Security sensors
                'mde-sensor-v2.1', 'mde-sensor-v2.0', 'mde-sensor-v2.2',
                's1-agent-v22.3', 's1-agent-v22.2', 's1-agent-v23.1',
                'falco-sensor-v0.35', 'falco-sensor-v0.34', 'falco-sensor-v0.36',
                'amp-connector-v1.8', 'amp-connector-v1.7', 'amp-connector-v1.9',
                'jamf-protect-2.8.1', 'jamf-protect-2.8.0', 'jamf-protect-2.9.0',
                
                # Vulnerability scanners
                'nessus-agent-8.15.0', 'rapid7-agent-3.2.1', 'qualys-scanner-4.8.2',
                'openvas-scanner-21.4', 'nexpose-engine-6.6.82', 'acunetix-scanner-14.5',
                
                # Status indicators
                'online', 'offline', 'updating', 'error', 'disconnected', 'active',
                'protected', 'unprotected', 'scanning', 'idle', 'quarantined'
            ],
            
            'platform_systems': [
                # Windows versions
                'Windows-NT-10.0.17763-x64', 'Windows-NT-10.0.19041-x64', 'Windows-NT-10.0.22000-x64',
                'Windows-Server-2019-Standard', 'Windows-Server-2022-Datacenter', 'Windows-Server-2016-Standard',
                'win2019-std-build-v4.2', 'win2022-dc-build-v1.0', 'win10-enterprise-v21h2',
                
                # Linux distributions
                'Linux-5.4.0-ubuntu-x86_64', 'Linux-5.15.0-ubuntu-x86_64', 'Linux-6.1.0-debian-x86_64',
                'ubuntu-20.04-dev-template', 'ubuntu-22.04-lts', 'ubuntu-18.04-legacy',
                'rhel8-db-optimized-v2.1', 'rhel9-base-v1.0', 'centos7-legacy-v3.5',
                'debian-11-bullseye', 'debian-12-bookworm', 'fedora-38-workstation',
                'alpine-3.18-minimal', 'arch-linux-rolling', 'opensuse-leap-15.5',
                
                # macOS versions
                'Darwin-21.6.0-arm64', 'Darwin-22.1.0-arm64', 'Darwin-20.6.0-x86_64',
                'macos-monterey-corp-image', 'macos-ventura-12.6', 'macos-big-sur-11.7',
                
                # Container platforms
                'container-os-optimized-v2023', 'coreos-stable-3815.2.1', 'flatcar-linux-3602.2.0',
                'kubernetes-node-v1.28', 'docker-desktop-4.24.0', 'podman-4.7.0',
                
                # Virtualization
                'vmware-esxi-7.0', 'vmware-esxi-8.0', 'hyper-v-2022',
                'citrix-xenserver-8.2', 'proxmox-ve-8.0', 'kvm-qemu-7.2',
                
                # Embedded/IoT
                'cisco-ios-15.1-M12a', 'cisco-asa-fw-9.18', 'palo-alto-panos-11.0',
                'fortinet-fortigate-7.4', 'juniper-junos-22.4', 'openwrt-23.05'
            ],
            
            'organizational_units': [
                # Department patterns
                'corp-it-messaging', 'corp-it-infrastructure', 'corp-it-security',
                'software-eng-platform', 'software-eng-backend', 'software-eng-frontend',
                'data-intelligence-team', 'data-analytics-group', 'data-science-lab',
                'finance-accounting', 'finance-treasury', 'finance-procurement',
                'hr-talent-acquisition', 'hr-people-ops', 'hr-learning-dev',
                'marketing-digital', 'marketing-content', 'marketing-events',
                'sales-enterprise', 'sales-smb', 'sales-inside',
                'operations-facilities', 'operations-supply-chain', 'operations-logistics',
                'legal-compliance', 'legal-contracts', 'legal-ip',
                'executive-staff-support', 'executive-communications', 'executive-strategy',
                
                # Team patterns
                'platform-reliability-eng', 'site-reliability-eng', 'devops-automation',
                'network-sec-operations', 'cyber-threat-intel', 'incident-response',
                'product-management', 'product-design', 'user-experience',
                'quality-assurance', 'test-automation', 'performance-eng',
                'business-intelligence', 'data-warehousing', 'machine-learning',
                
                # Project codes
                'proj-alpha-team', 'proj-beta-squad', 'proj-gamma-group',
                'initiative-cloud', 'initiative-mobile', 'initiative-ai',
                'workstream-security', 'workstream-compliance', 'workstream-migration'
            ],
            
            'geographic_locations': [
                # Data centers
                'facility-nyc-dc1', 'facility-nyc-dc2', 'facility-sfo-dc1',
                'datacenter-chicago-tier3', 'datacenter-dallas-tier4', 'datacenter-miami-tier2',
                'dc-virginia-primary', 'dc-oregon-secondary', 'dc-california-dr',
                
                # Cloud regions
                'cloud-aws-use1', 'cloud-aws-usw2', 'cloud-aws-euc1',
                'aws-us-east-1', 'aws-us-west-2', 'aws-eu-central-1',
                'gcp-us-central1', 'gcp-europe-west1', 'gcp-asia-southeast1',
                'azure-eastus', 'azure-westeurope', 'azure-southeastasia',
                
                # Geographic codes
                'us-east-coast', 'us-west-coast', 'us-central',
                'eu-central-datacenter', 'eu-west-primary', 'eu-north-backup',
                'asia-pacific-zone', 'asia-southeast-region', 'asia-northeast-hub',
                'americas-north', 'americas-south', 'emea-primary',
                
                # Office locations
                'office-nyc-manhattan', 'office-sfo-soma', 'office-london-canary',
                'campus-seattle-main', 'campus-austin-tech', 'campus-dublin-emea',
                'site-singapore-apac', 'site-sydney-pacific', 'site-tokyo-jp',
                
                # Colo facilities
                'colo-london-cage12', 'colo-frankfurt-suite5', 'colo-tokyo-rack3',
                'hosting-amsterdam-nl', 'hosting-toronto-ca', 'hosting-mumbai-in',
                
                # Mobile/Distributed
                'distributed-mobile', 'remote-workforce', 'field-operations',
                'edge-location-01', 'cdn-pop-west', 'cdn-pop-east'
            ],
            
            'domain_entities': [
                # Corporate domains
                'api.company.com', 'www.company.com', 'mail.company.com',
                'intranet.company.com', 'vpn.company.com', 'files.company.com',
                'wiki.company.com', 'docs.company.com', 'support.company.com',
                
                # Internal domains
                'internal.corp.local', 'mail.corp.local', 'files.corp.local',
                'wiki.corp.internal', 'jenkins.corp.internal', 'git.corp.internal',
                'monitoring.corp.internal', 'logs.corp.internal', 'metrics.corp.internal',
                
                # Service domains
                'api-gateway.prod.internal', 'auth-service.prod.internal', 'user-service.prod.internal',
                'payment-service.prod.internal', 'notification-service.prod.internal', 'search-service.prod.internal',
                'frontend-app.dmz.corp', 'admin-panel.mgmt.corp', 'dashboard.ops.corp',
                
                # Kubernetes services
                'kubernetes.default.svc.cluster.local', 'kube-dns.kube-system.svc.cluster.local',
                'ingress-nginx.ingress-nginx.svc.cluster.local', 'prometheus.monitoring.svc.cluster.local',
                'grafana.monitoring.svc.cluster.local', 'elasticsearch.logging.svc.cluster.local',
                
                # Cloud services
                'backup.s3.amazonaws.com', 'logs.cloudwatch.amazonaws.com', 'secrets.vault.azure.com',
                'storage.googleapis.com', 'bigquery.googleapis.com', 'firestore.googleapis.com',
                
                # CDN and external
                'cdn.jquery.com', 'fonts.googleapis.com', 'ajax.cloudflare.com',
                'cdn.jsdelivr.net', 'unpkg.com', 'cdnjs.cloudflare.com',
                
                # Development
                'dev-api.staging.corp', 'test-app.qa.corp', 'demo.sandbox.corp',
                'localhost:3000', 'localhost:8080', '127.0.0.1:5000'
            ]
        }
        
        # Generate massive dataset with variations
        X, y = [], []
        
        for category, base_patterns in patterns.items():
            # Original patterns
            for pattern in base_patterns:
                X.append(pattern)
                y.append(category)
            
            # Generate variations for each pattern
            for pattern in base_patterns[:20]:  # Use first 20 for variations
                # Number variations
                for i in range(1, 100):
                    if any(char.isdigit() for char in pattern):
                        # Replace existing numbers
                        varied = re.sub(r'\d+', str(i), pattern)
                        X.append(varied)
                        y.append(category)
                    else:
                        # Add numbers
                        X.append(f"{pattern}-{i:02d}")
                        y.append(category)
                
                # Letter variations
                for letter in 'abcdefghijklmnopqrstuvwxyz':
                    if '-' in pattern:
                        varied = pattern.replace('-', f'-{letter}-')
                        X.append(varied)
                        y.append(category)
                
                # Version variations
                for major in range(1, 10):
                    for minor in range(0, 10):
                        if 'v' in pattern.lower():
                            varied = re.sub(r'v\d+\.\d+', f'v{major}.{minor}', pattern, flags=re.IGNORECASE)
                            X.append(varied)
                            y.append(category)
        
        # Add noise data
        noise_patterns = [
            'random_string_123', 'test_data_456', 'unknown_value_789',
            'null', 'undefined', 'empty', 'missing', 'n/a', 'none',
            'placeholder_text', 'sample_data', 'dummy_value',
            '12345', 'abcdef', 'qwerty', 'password', 'admin'
        ]
        
        for noise in noise_patterns * 50:
            X.append(noise)
            y.append('unknown')
        
        print(f"Generated {len(X)} training samples across {len(set(y))} categories")
        return X, y
    
    def extract_advanced_features(self, texts, column_names=None):
        if column_names is None:
            column_names = ['unknown'] * len(texts)
        
        # Text features using TF-IDF
        tfidf_features = self.vectorizer.fit_transform(texts).toarray()
        
        # Advanced manual features
        manual_features = []
        
        for i, text in enumerate(texts):
            col_name = column_names[i].lower()
            text_lower = str(text).lower()
            
            features = [
                # Length features
                len(text),
                len(text.split('-')),
                len(text.split('_')),
                len(text.split('.')),
                len(text.split(':')),
                len(text.split('/')),
                
                # Character type ratios
                sum(c.isupper() for c in text) / max(len(text), 1),
                sum(c.islower() for c in text) / max(len(text), 1),
                sum(c.isdigit() for c in text) / max(len(text), 1),
                sum(c in '-_.' for c in text) / max(len(text), 1),
                sum(c in ':/' for c in text) / max(len(text), 1),
                
                # Pattern detection
                bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text)),  # IP
                bool(re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', text_lower)),  # UUID
                bool(re.search(r'v\d+\.\d+', text_lower)),  # Version
                bool(re.search(r'[a-z]+-[a-z]+-\d+', text_lower)),  # Service pattern
                bool(re.search(r'[A-Z]{3,4}-[A-Z]+-\d+', text)),  # Corp pattern
                bool(re.search(r'\.[a-z]{2,4}$', text_lower)),  # Domain ending
                bool(re.search(r'^\d+$', text)),  # Pure number
                
                # Keyword presence
                'srv' in text_lower, 'server' in text_lower, 'host' in text_lower,
                'db' in text_lower, 'database' in text_lower, 'mysql' in text_lower,
                'web' in text_lower, 'api' in text_lower, 'app' in text_lower,
                'prod' in text_lower, 'test' in text_lower, 'dev' in text_lower,
                'corp' in text_lower, 'company' in text_lower, 'internal' in text_lower,
                'agent' in text_lower, 'sensor' in text_lower, 'client' in text_lower,
                'defender' in text_lower, 'falcon' in text_lower, 'security' in text_lower,
                'windows' in text_lower, 'linux' in text_lower, 'ubuntu' in text_lower,
                'aws' in text_lower, 'gcp' in text_lower, 'azure' in text_lower,
                'docker' in text_lower, 'k8s' in text_lower, 'kubernetes' in text_lower,
                
                # Column name hints
                'host' in col_name, 'name' in col_name, 'id' in col_name,
                'addr' in col_name, 'ip' in col_name, 'network' in col_name,
                'agent' in col_name, 'sensor' in col_name, 'security' in col_name,
                'os' in col_name, 'platform' in col_name, 'system' in col_name,
                'dept' in col_name, 'team' in col_name, 'org' in col_name,
                'location' in col_name, 'region' in col_name, 'facility' in col_name,
                'domain' in col_name, 'fqdn' in col_name, 'service' in col_name,
                
                # Advanced patterns
                text.count('-'), text.count('_'), text.count('.'), 
                text.count(':'), text.count('/'), text.count('\\'),
                len([c for c in text if c.isdigit()]),
                len([c for c in text if c.isupper()]),
                len([c for c in text if c.islower()]),
                
                # Position of first digit
                next((i for i, c in enumerate(text) if c.isdigit()), len(text)),
                # Position of first uppercase
                next((i for i, c in enumerate(text) if c.isupper()), len(text)),
                # Position of first special char
                next((i for i, c in enumerate(text) if c in '-_.:/'), len(text)),
            ]
            
            manual_features.append(features)
        
        # Combine TF-IDF and manual features
        manual_features = np.array(manual_features)
        combined_features = np.hstack([tfidf_features, manual_features])
        
        print(f"Feature matrix shape: {combined_features.shape}")
        return combined_features
    
    def train_insane_model(self):
        print("🚀 Training INSANE ACCURACY model...")
        
        # Generate massive training data
        X_text, y = self.generate_massive_training_data()
        
        # Extract features
        X = self.extract_advanced_features(X_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Try multiple algorithms with hyperparameter tuning
        models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        }
        
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model, param_grids[name], 
                cv=5, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluate on test set
            test_score = grid_search.score(X_test, y_test)
            
            print(f"{name} best CV score: {grid_search.best_score_:.4f}")
            print(f"{name} test score: {test_score:.4f}")
            print(f"{name} best params: {grid_search.best_params_}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = grid_search.best_estimator_
                best_name = name
        
        self.model = best_model
        
        # Final evaluation
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Best Model: {best_name}")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        joblib.dump(self.model, 'models/insane_model.pkl')
        joblib.dump(self.vectorizer, 'models/insane_vectorizer.pkl')
        
        with open('models/insane_results.json', 'w') as f:
            json.dump({
                'best_model': best_name,
                'accuracy': float(accuracy),
                'test_samples': len(y_test),
                'training_samples': len(y_train),
                'feature_count': X.shape[1],
                'categories': list(set(y)),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        return accuracy
    
    def predict_insane(self, texts, column_names=None):
        if self.model is None:
            self.model = joblib.load('models/insane_model.pkl')
            self.vectorizer = joblib.load('models/insane_vectorizer.pkl')
        
        # Extract same features as training
        if column_names is None:
            column_names = ['unknown'] * len(texts)
        
        # Use pre-fitted vectorizer
        tfidf_features = self.vectorizer.transform(texts).toarray()
        
        # Manual features (same as training)
        manual_features = []
        for i, text in enumerate(texts):
            col_name = column_names[i].lower()
            text_lower = str(text).lower()
            
            features = [
                len(text), len(text.split('-')), len(text.split('_')), len(text.split('.')),
                len(text.split(':')), len(text.split('/')),
                sum(c.isupper() for c in text) / max(len(text), 1),
                sum(c.islower() for c in text) / max(len(text), 1),
                sum(c.isdigit() for c in text) / max(len(text), 1),
                sum(c in '-_.' for c in text) / max(len(text), 1),
                sum(c in ':/' for c in text) / max(len(text), 1),
                bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text)),
                bool(re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', text_lower)),
                bool(re.search(r'v\d+\.\d+', text_lower)),
                bool(re.search(r'[a-z]+-[a-z]+-\d+', text_lower)),
                bool(re.search(r'[A-Z]{3,4}-[A-Z]+-\d+', text)),
                bool(re.search(r'\.[a-z]{2,4}$', text_lower)),
                bool(re.search(r'^\d+$', text)),
                'srv' in text_lower, 'server' in text_lower, 'host' in text_lower,
                'db' in text_lower, 'database' in text_lower, 'mysql' in text_lower,
                'web' in text_lower, 'api' in text_lower, 'app' in text_lower,
                'prod' in text_lower, 'test' in text_lower, 'dev' in text_lower,
                'corp' in text_lower, 'company' in text_lower, 'internal' in text_lower,
                'agent' in text_lower, 'sensor' in text_lower, 'client' in text_lower,
                'defender' in text_lower, 'falcon' in text_lower, 'security' in text_lower,
                'windows' in text_lower, 'linux' in text_lower, 'ubuntu' in text_lower,
                'aws' in text_lower, 'gcp' in text_lower, 'azure' in text_lower,
                'docker' in text_lower, 'k8s' in text_lower, 'kubernetes' in text_lower,
                'host' in col_name, 'name' in col_name, 'id' in col_name,
                'addr' in col_name, 'ip' in col_name, 'network' in col_name,
                'agent' in col_name, 'sensor' in col_name, 'security' in col_name,
                'os' in col_name, 'platform' in col_name, 'system' in col_name,
                'dept' in col_name, 'team' in col_name, 'org' in col_name,
                'location' in col_name, 'region' in col_name, 'facility' in col_name,
                'domain' in col_name, 'fqdn' in col_name, 'service' in col_name,
                text.count('-'), text.count('_'), text.count('.'),
                text.count(':'), text.count('/'), text.count('\\'),
                len([c for c in text if c.isdigit()]),
                len([c for c in text if c.isupper()]),
                len([c for c in text if c.islower()]),
                next((i for i, c in enumerate(text) if c.isdigit()), len(text)),
                next((i for i, c in enumerate(text) if c.isupper()), len(text)),
                next((i for i, c in enumerate(text) if c in '-_.:/'), len(text)),
            ]
            manual_features.append(features)
        
        manual_features = np.array(manual_features)
        combined_features = np.hstack([tfidf_features, manual_features])
        
        predictions = self.model.predict(combined_features)
        probabilities = self.model.predict_proba(combined_features)
        
        return predictions, probabilities

if __name__ == '__main__':
    ml = InsaneAccuracyML()
    accuracy = ml.train_insane_model()
    
    if accuracy >= 0.95:
        print(f"\n🔥 INSANE ACCURACY ACHIEVED: {accuracy*100:.2f}%! 🔥")
    elif accuracy >= 0.90:
        print(f"\n⚡ EXCELLENT ACCURACY: {accuracy*100:.2f}%! ⚡")
    else:
        print(f"\n📈 Good start: {accuracy*100:.2f}% - optimizing further...")
