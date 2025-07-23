#!/usr/bin/env python3

import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Any
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class CybersecurityFieldClassifier:
    def __init__(self, discovery_results_path: str = "bigquery_discovery_results.json"):
        self.discovery_results_path = Path(discovery_results_path)
        
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        self.cybersecurity_log_roles = {
            "network": ["firewall_traffic", "ids_ips", "ndr", "proxy", "dns", "waf"],
            "endpoint": ["os_logs", "edr", "dlp", "fim"],
            "cloud": ["cloud_event", "cloud_load_balancer", "cloud_config", "theom", "wiz", "cloud_security"],
            "application": ["web_logs", "api_gateway"],
            "identity_authentication": ["authentication_attempts", "privilege_escalation", "identity_lifecycle"]
        }
        
        self.common_data_fields = {
            "network": ["ip_source", "ip_target", "protocol", "detection_signature", "port", "dns_record_fqdn", "http_headers"],
            "endpoint": ["system_name", "ip", "filename"],
            "cloud": ["instance_id", "resource_id", "vpc_id"],
            "application": ["user_agent", "session_id", "endpoint_path"],
            "identity_authentication": ["username", "domain", "group"]
        }
        
        self.visibility_factors = {
            "network": ["url_fqdn_coverage", "cmdb_asset_visibility", "network_zones_spans", "ipam_public_ip_coverage", 
                       "geolocation", "vpc", "log_ingest_volume"],
            "endpoint": ["cmdb_asset_visibility", "crowdstrike_agent_coverage", "log_ingest_volume"],
            "cloud": ["vpc", "ipam_public_ip_coverage", "url_fqdn_coverage", "crowdstrike_agent_coverage"],
            "application": ["url_fqdn_coverage", "control_coverage", "domain", "internal", "external"],
            "identity_authentication": ["domain", "internal", "external", "controls"]
        }
        
        self.field_classifications = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def classify_all_fields(self):
        if not self.discovery_results_path.exists():
            self.logger.error(f"Discovery results not found: {self.discovery_results_path}")
            return None
        
        with open(self.discovery_results_path, 'r') as f:
            discovery_data = json.load(f)
        
        self.logger.info("Starting field classification...")
        
        for dataset_id, dataset_info in discovery_data.get("all_datasets", {}).items():
            for table_name, table_info in dataset_info.get("tables", {}).items():
                for field in table_info.get("fields", []):
                    field_key = f"{dataset_id}.{table_name}.{field['column_name']}"
                    
                    classification = self.classify_field(
                        field["column_name"],
                        field.get("data_type", ""),
                        f"{dataset_id}.{table_name}",
                        field
                    )
                    
                    self.field_classifications[field_key] = classification
        
        self.save_classifications()
        return self.field_classifications

    def classify_field(self, field_name: str, field_type: str, table_context: str, field_info: Dict) -> Dict:
        classification = {
            "field_name": field_name,
            "field_type": field_type,
            "table_context": table_context,
            "log_role": self.classify_log_role(field_name, table_context),
            "log_type": self.classify_log_type(field_name, table_context),
            "common_field_type": self.classify_common_field_type(field_name),
            "visibility_factors": self.classify_visibility_factors(field_name),
            "infrastructure_type": self.classify_infrastructure_type(table_context),
            "privacy_level": self.classify_privacy_level(field_name),
            "is_partitioning": field_info.get("is_partitioning_column", "NO") == "YES",
            "is_clustering": field_info.get("clustering_ordinal_position") is not None
        }
        
        return classification

    def classify_log_role(self, field_name: str, table_context: str) -> str:
        combined_text = f"{field_name} {table_context}".lower()
        
        role_keywords = {
            "network": ["ip", "port", "protocol", "dns", "url", "firewall", "proxy", "traffic", "packet"],
            "endpoint": ["host", "system", "file", "process", "registry", "event", "endpoint"],
            "cloud": ["cloud", "instance", "resource", "vpc", "gcp", "aws", "azure"],
            "application": ["app", "web", "api", "http", "request", "response", "session"],
            "identity_authentication": ["user", "auth", "login", "credential", "identity", "account"]
        }
        
        for role, keywords in role_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return role
        
        log_role_labels = ["network", "endpoint", "cloud", "application", "identity_authentication"]
        classification = self.classifier(combined_text, log_role_labels)
        return classification["labels"][0]

    def classify_log_type(self, field_name: str, table_context: str) -> str:
        combined_text = f"{field_name} {table_context}".lower()
        
        for role, log_types in self.cybersecurity_log_roles.items():
            for log_type in log_types:
                log_type_keywords = log_type.split('_')
                if any(keyword in combined_text for keyword in log_type_keywords):
                    return log_type
        
        return "unknown"

    def classify_common_field_type(self, field_name: str) -> str:
        field_lower = field_name.lower()
        
        for category, field_types in self.common_data_fields.items():
            for field_type in field_types:
                field_type_parts = field_type.split('_')
                if any(part in field_lower for part in field_type_parts):
                    return field_type
        
        return "unknown"

    def classify_visibility_factors(self, field_name: str) -> List[str]:
        field_lower = field_name.lower()
        applicable_factors = []
        
        if any(keyword in field_lower for keyword in ['url', 'fqdn', 'domain', 'hostname']):
            applicable_factors.append("url_fqdn_coverage")
        
        if any(keyword in field_lower for keyword in ['asset', 'host', 'system', 'device']):
            applicable_factors.append("cmdb_asset_visibility")
        
        if any(keyword in field_lower for keyword in ['ip', 'address']):
            applicable_factors.append("ipam_public_ip_coverage")
        
        if any(keyword in field_lower for keyword in ['zone', 'span', 'segment']):
            applicable_factors.append("network_zones_spans")
        
        if any(keyword in field_lower for keyword in ['geo', 'location', 'country', 'region']):
            applicable_factors.append("geolocation")
        
        if any(keyword in field_lower for keyword in ['vpc', 'cloud']):
            applicable_factors.append("vpc")
        
        if any(keyword in field_lower for keyword in ['log', 'event', 'record']):
            applicable_factors.append("log_ingest_volume")
        
        return applicable_factors

    def classify_infrastructure_type(self, table_context: str) -> str:
        context_lower = table_context.lower()
        
        if any(keyword in context_lower for keyword in ['cloud', 'gcp', 'aws', 'azure']):
            return "cloud"
        elif any(keyword in context_lower for keyword in ['saas', 'service']):
            return "saas"
        elif any(keyword in context_lower for keyword in ['api', 'gateway']):
            return "api"
        else:
            return "on_prem"

    def classify_privacy_level(self, field_name: str) -> str:
        field_lower = field_name.lower()
        
        pii_indicators = ["email", "phone", "ssn", "name", "address", "birth", "passport", "license"]
        if any(indicator in field_lower for indicator in pii_indicators):
            return "restricted"
        
        sensitive_indicators = ["password", "secret", "token", "key", "hash", "credential"]
        if any(indicator in field_lower for indicator in sensitive_indicators):
            return "confidential"
        
        if any(keyword in field_lower for keyword in ['user', 'person', 'individual']):
            return "internal"
        
        return "public"

    def save_classifications(self):
        output_file = Path("field_classifications.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.field_classifications, f, indent=2, default=str)
        
        self.logger.info(f"Field classifications saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    discovery_results = sys.argv[1] if len(sys.argv) > 1 else "bigquery_discovery_results.json"
    
    classifier = CybersecurityFieldClassifier(discovery_results)
    results = classifier.classify_all_fields()
    
    print(f"\nField Classification Summary:")
    print(f"- Total fields classified: {len(results)}")
    
    log_roles = {}
    for classification in results.values():
        role = classification["log_role"]
        log_roles[role] = log_roles.get(role, 0) + 1
    
    print(f"- Log role distribution: {log_roles}")