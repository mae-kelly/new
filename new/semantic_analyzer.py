#!/usr/bin/env python3

import json
from pathlib import Path
import logging
from typing import Dict, List, Any
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import spacy

class SemanticFieldAnalyzer:
    def __init__(self, classification_results_path: str = "field_classifications.json"):
        self.classification_results_path = Path(classification_results_path)
        
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
        
        self.semantic_analysis = {
            "field_meanings": {},
            "business_purposes": {},
            "data_relationships": {},
            "semantic_clusters": {}
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def analyze_all_fields(self):
        if not self.classification_results_path.exists():
            self.logger.error(f"Classification results not found: {self.classification_results_path}")
            return None
        
        with open(self.classification_results_path, 'r') as f:
            classification_data = json.load(f)
        
        self.logger.info("Starting semantic field analysis...")
        
        for field_key, classification in classification_data.items():
            semantic_info = self.analyze_field_semantics(field_key, classification)
            self.semantic_analysis["field_meanings"][field_key] = semantic_info
        
        self.analyze_business_purposes(classification_data)
        self.discover_data_relationships(classification_data)
        self.create_semantic_clusters(classification_data)
        
        self.save_semantic_analysis()
        return self.semantic_analysis

    def analyze_field_semantics(self, field_key: str, classification: Dict) -> Dict:
        field_name = classification.get("field_name", "")
        table_context = classification.get("table_context", "")
        log_role = classification.get("log_role", "")
        
        semantic_info = {
            "primary_meaning": self.extract_primary_meaning(field_name, table_context),
            "cybersecurity_purpose": self.infer_cybersecurity_purpose(field_name, log_role),
            "data_content_type": self.classify_data_content(field_name, classification.get("field_type", "")),
            "operational_significance": self.assess_operational_significance(field_name, log_role),
            "threat_detection_relevance": self.evaluate_threat_detection_relevance(field_name, log_role)
        }
        
        return semantic_info

    def extract_primary_meaning(self, field_name: str, table_context: str) -> str:
        field_desc = f"Field named '{field_name}' in table '{table_context}'"
        
        meaning_labels = [
            "stores network traffic information",
            "contains authentication data",
            "holds system event details",
            "tracks user activity",
            "records security incidents",
            "monitors application behavior",
            "stores configuration settings",
            "contains threat intelligence",
            "tracks asset information",
            "holds compliance data"
        ]
        
        classification = self.classifier(field_desc, meaning_labels)
        return classification["labels"][0]

    def infer_cybersecurity_purpose(self, field_name: str, log_role: str) -> str:
        purpose_mapping = {
            "network": "monitors network traffic for security policy enforcement and threat detection",
            "endpoint": "tracks endpoint activities for malware detection and system monitoring",
            "cloud": "monitors cloud infrastructure for security compliance and threat detection",
            "application": "tracks application behavior for security monitoring and performance analysis",
            "identity_authentication": "manages user authentication and access control for security"
        }
        
        return purpose_mapping.get(log_role, f"supports {log_role} security monitoring")

    def classify_data_content(self, field_name: str, field_type: str) -> str:
        field_lower = field_name.lower()
        
        content_types = {
            "ip_address": ["ip", "addr", "address"],
            "timestamp": ["time", "date", "timestamp", "created", "modified"],
            "identifier": ["id", "uuid", "key", "guid"],
            "user_info": ["user", "username", "account", "email"],
            "network_info": ["port", "protocol", "dns", "url"],
            "system_info": ["host", "system", "device", "machine"],
            "security_event": ["event", "alert", "incident", "threat"],
            "file_info": ["file", "path", "filename", "directory"]
        }
        
        for content_type, keywords in content_types.items():
            if any(keyword in field_lower for keyword in keywords):
                return content_type
        
        return f"general_{field_type.lower()}"

    def assess_operational_significance(self, field_name: str, log_role: str) -> str:
        field_lower = field_name.lower()
        
        significance_keywords = {
            "critical": ["id", "key", "primary", "main", "core"],
            "high": ["time", "user", "system", "event", "alert"],
            "medium": ["description", "details", "info", "data"],
            "low": ["comment", "note", "misc", "other"]
        }
        
        for level, keywords in significance_keywords.items():
            if any(keyword in field_lower for keyword in keywords):
                return f"{level} significance for {log_role} operations"
        
        return f"standard significance for {log_role} operations"

    def evaluate_threat_detection_relevance(self, field_name: str, log_role: str) -> str:
        field_lower = field_name.lower()
        
        threat_keywords = {
            "high": ["malware", "attack", "threat", "intrusion", "breach"],
            "medium": ["suspicious", "anomaly", "violation", "unauthorized"],
            "standard": ["event", "log", "activity", "behavior"]
        }
        
        for relevance, keywords in threat_keywords.items():
            if any(keyword in field_lower for keyword in keywords):
                return f"{relevance} relevance for threat detection"
        
        return "contributes to general threat detection capabilities"

    def analyze_business_purposes(self, classification_data: Dict):
        for field_key, classification in classification_data.items():
            log_role = classification.get("log_role", "")
            
            business_purpose = {
                "primary_business_value": self.assess_business_value(log_role),
                "compliance_support": self.determine_compliance_value(log_role),
                "risk_mitigation": self.evaluate_risk_mitigation(log_role),
                "operational_efficiency": f"enhances {log_role} operational visibility"
            }
            
            self.semantic_analysis["business_purposes"][field_key] = business_purpose

    def assess_business_value(self, log_role: str) -> str:
        value_map = {
            "network": "protects business infrastructure through network security monitoring",
            "endpoint": "secures business endpoints and prevents data breaches",
            "cloud": "ensures cloud infrastructure security and compliance",
            "application": "protects business applications and customer data",
            "identity_authentication": "secures business access and prevents unauthorized use"
        }
        return value_map.get(log_role, "supports business security objectives")

    def determine_compliance_value(self, log_role: str) -> str:
        compliance_map = {
            "network": "supports PCI DSS and network security compliance",
            "endpoint": "enables GDPR and endpoint security compliance",
            "identity_authentication": "satisfies SOX and access control mandates",
            "application": "supports OWASP and application security standards",
            "cloud": "ensures cloud security compliance and governance"
        }
        return compliance_map.get(log_role, "supports general compliance requirements")

    def evaluate_risk_mitigation(self, log_role: str) -> str:
        risk_map = {
            "network": "mitigates network intrusion and data exfiltration risks",
            "endpoint": "reduces endpoint compromise and malware risks",
            "identity_authentication": "prevents unauthorized access and insider threats",
            "application": "reduces application vulnerabilities and attack risks",
            "cloud": "mitigates cloud security and configuration risks"
        }
        return risk_map.get(log_role, f"contributes to {log_role} risk reduction")

    def discover_data_relationships(self, classification_data: Dict):
        relationships = {}
        
        for field_key, classification in classification_data.items():
            field_name = classification.get("field_name", "")
            log_role = classification.get("log_role", "")
            
            relationships[field_key] = {
                "similar_fields": self.find_similar_fields(field_name, classification_data),
                "related_log_roles": self.find_related_roles(log_role, classification_data),
                "dependency_fields": self.identify_dependencies(field_name),
                "correlation_potential": self.assess_correlation_potential(field_name, log_role)
            }
        
        self.semantic_analysis["data_relationships"] = relationships

    def find_similar_fields(self, field_name: str, classification_data: Dict) -> List[str]:
        similar_fields = []
        field_embedding = self.sentence_model.encode([field_name])
        
        for other_key, other_classification in classification_data.items():
            other_name = other_classification.get("field_name", "")
            if other_name != field_name:
                other_embedding = self.sentence_model.encode([other_name])
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(field_embedding, other_embedding)[0][0]
                
                if similarity > 0.7:
                    similar_fields.append(other_name)
        
        return similar_fields[:5]

    def find_related_roles(self, log_role: str, classification_data: Dict) -> List[str]:
        related_roles = set()
        
        for classification in classification_data.values():
            other_role = classification.get("log_role", "")
            if other_role != log_role and other_role != "unknown":
                related_roles.add(other_role)
        
        return list(related_roles)[:3]

    def identify_dependencies(self, field_name: str) -> List[str]:
        field_lower = field_name.lower()
        
        dependency_map = {
            "source": ["destination", "target", "protocol"],
            "user": ["session", "authentication", "domain"],
            "event": ["timestamp", "severity", "source"],
            "file": ["path", "hash", "size"],
            "process": ["pid", "user", "command"]
        }
        
        for key, deps in dependency_map.items():
            if key in field_lower:
                return deps
        
        return []

    def assess_correlation_potential(self, field_name: str, log_role: str) -> str:
        field_lower = field_name.lower()
        
        if any(keyword in field_lower for keyword in ["id", "key", "timestamp", "user"]):
            return "high correlation potential"
        elif any(keyword in field_lower for keyword in ["ip", "host", "session"]):
            return "medium correlation potential"
        else:
            return "low correlation potential"

    def create_semantic_clusters(self, classification_data: Dict):
        from sklearn.cluster import KMeans
        import numpy as np
        
        field_names = [cls.get("field_name", "") for cls in classification_data.values()]
        
        if len(field_names) > 5:
            embeddings = self.sentence_model.encode(field_names)
            
            n_clusters = min(8, len(field_names) // 3)
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clustering.fit_predict(embeddings)
            
            clusters = {}
            for i, (field_key, classification) in enumerate(classification_data.items()):
                cluster_id = f"cluster_{cluster_labels[i]}"
                
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                
                clusters[cluster_id].append({
                    "field_key": field_key,
                    "field_name": classification.get("field_name", ""),
                    "log_role": classification.get("log_role", "")
                })
            
            self.semantic_analysis["semantic_clusters"] = clusters

    def save_semantic_analysis(self):
        output_file = Path("semantic_field_analysis.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.semantic_analysis, f, indent=2, default=str)
        
        self.logger.info(f"Semantic analysis saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    classification_results = sys.argv[1] if len(sys.argv) > 1 else "field_classifications.json"
    
    analyzer = SemanticFieldAnalyzer(classification_results)
    results = analyzer.analyze_all_fields()
    
    print(f"\nSemantic Analysis Summary:")
    print(f"- Field meanings analyzed: {len(results['field_meanings'])}")
    print(f"- Business purposes identified: {len(results['business_purposes'])}")
    print(f"- Data relationships mapped: {len(results['data_relationships'])}")
    print(f"- Semantic clusters created: {len(results['semantic_clusters'])}")