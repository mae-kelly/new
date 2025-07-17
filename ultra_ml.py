import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import json
from datetime import datetime

class UltraLeanML:
    def __init__(self):
        self.model = None
        
    def extract_features(self, text, column_name):
        text = str(text).lower()
        return [
            len(text),
            text.count('-'),
            text.count('.'),
            text.count(':'),
            sum(c.isdigit() for c in text) / max(len(text), 1),
            'srv' in text,
            'db' in text,
            'prod' in text,
            'agent' in text,
            'windows' in text,
            'linux' in text,
            'corp' in text,
            bool(re.search(r'\d+\.\d+\.\d+\.\d+', text)),
            'id' in column_name.lower(),
            'addr' in column_name.lower(),
            'host' in column_name.lower()
        ]
    
    def train(self):
        patterns = {
            'host': ['srv-web-01', 'db-prod-mysql', 'CORP-LAPTOP-01'],
            'network': ['192.168.1.100', '10.0.0.1', 'eth0:172.16.1.1'],
            'security': ['defender-v4.18', 'agent-online', 'falcon-sensor'],
            'platform': ['ubuntu-20.04', 'windows-2019', 'rhel8'],
            'org': ['engineering-team', 'finance-dept', 'it-ops'],
            'location': ['us-east-1', 'datacenter-nyc', 'cloud-aws'],
            'domain': ['api.company.com', 'internal.corp', 'service.net']
        }
        
        X, y = [], []
        
        for category, examples in patterns.items():
            for example in examples * 100:
                features = self.extract_features(example, f'test_{category}_col')
                X.append(features)
                y.append(category)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_score = accuracy_score(y_test, rf_model.predict(X_test))
        
        xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_score = accuracy_score(y_test, xgb_model.predict(X_test))
        
        if xgb_score > rf_score:
            self.model = xgb_model
            best_score = xgb_score
            best_name = 'xgboost'
        else:
            self.model = rf_model
            best_score = rf_score
            best_name = 'random_forest'
        
        joblib.dump(self.model, 'models/best_model.pkl')
        
        result = {
            'best_model': best_name,
            'accuracy': float(best_score),
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f'Training complete: {best_name} with {best_score:.3f} accuracy')
        return result
    
    def predict(self, table_data, table_name):
        if not self.model:
            self.model = joblib.load('models/best_model.pkl')
        
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
        
        for col in df.columns:
            if df[col].dtype == 'object':
                samples = df[col].dropna().head(5).tolist()
                if samples:
                    features = [self.extract_features(s, col) for s in samples]
                    predictions = self.model.predict(features)
                    probs = self.model.predict_proba(features)
                    
                    most_common = max(set(predictions), key=list(predictions).count)
                    avg_confidence = np.mean([max(p) for p in probs])
                    
                    if avg_confidence > 0.6:
                        mappings.append({
                            'source': f'{table_name}.{col}',
                            'metric': visibility_map.get(most_common, 'unknown_coverage'),
                            'confidence': min(float(avg_confidence), 1.0),
                            'pattern_type': most_common,
                            'samples': samples[:3]
                        })
        
        return mappings

if __name__ == '__main__':
    ml = UltraLeanML()
    ml.train()
