#!/usr/bin/env python3

import json
import os
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Any
from google.cloud import bigquery
from google.oauth2 import service_account
from transformers import pipeline
import re
import ssl
import certifi

class ContentAnalyzer:
    def __init__(self, project_root: str = ".", discovery_results_path: str = "bigquery_discovery_results.json"):
        self.project_root = Path(project_root)
        self.discovery_results_path = Path(discovery_results_path)
        
        # Configure SSL/TLS using the actual certificates in the repo
        ssl_cert_path = os.path.join(self.project_root, "..", "ssl", "nexia.1dc.com.crt")
        ssl_key_path = os.path.join(self.project_root, "..", "ssl", "nexia.1dc.com.key")
        
        # Set SSL environment variables to use the repo's certificates
        if os.path.exists(ssl_cert_path):
            os.environ['SSL_CERT_FILE'] = ssl_cert_path
            os.environ['REQUESTS_CA_BUNDLE'] = ssl_cert_path
            os.environ['CURL_CA_BUNDLE'] = ssl_cert_path
        else:
            # Fallback to certifi if repo certs not found
            os.environ['SSL_CERT_FILE'] = certifi.where()
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            os.environ['CURL_CA_BUNDLE'] = certifi.where()
        
        # Disable SSL warnings
        import urllib3
        urllib3.disable_warnings()
        
        # Create custom SSL context if we have the key file
        if os.path.exists(ssl_key_path) and os.path.exists(ssl_cert_path):
            ssl_context = ssl.create_default_context()
            ssl_context.load_cert_chain(ssl_cert_path, ssl_key_path)
        else:
            ssl._create_default_https_context = ssl._create_unverified_context
        
        SERVICE_ACCOUNT_FILE = os.path.join(self.project_root, "gcp_prod_key.json")
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        project = "chronicle-flow"
        clientBQ = bigquery.Client(project=project, credentials=credentials)
        self.clientBQ = clientBQ
        
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        
        self.content_analysis = {
            "table_content_analysis": {},
            "field_value_patterns": {},
            "data_quality_metrics": {},
            "semantic_content_mapping": {}
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def runBQQuery(self, query, params=None):
        if params:
            df = self.clientBQ.query(query, job_config=params).to_dataframe()
        else:
            df = self.clientBQ.query(query).to_dataframe()
        return df

    def analyze_all_content(self):
        if not self.discovery_results_path.exists():
            self.logger.error(f"Discovery results not found: {self.discovery_results_path}")
            return None
        
        with open(self.discovery_results_path, 'r') as f:
            discovery_data = json.load(f)
        
        self.logger.info("Starting comprehensive content analysis...")
        
        for dataset_id, dataset_info in discovery_data.get("all_datasets", {}).items():
            for table_name, table_info in dataset_info.get("tables", {}).items():
                if table_info.get("row_count", 0) > 0:
                    self.analyze_table_content(dataset_id, table_name, table_info)
        
        self.save_content_analysis()
        return self.content_analysis

    def analyze_table_content(self, dataset_id: str, table_name: str, table_info: Dict):
        table_key = f"{dataset_id}.{table_name}"
        
        try:
            sample_query = f"""
            SELECT *
            FROM `chronicle-flow.{dataset_id}.{table_name}`
            ORDER BY RAND()
            LIMIT 50
            """
            
            sample_df = self.runBQQuery(sample_query)
            
            if not sample_df.empty:
                content_analysis = {
                    "table_key": table_key,
                    "sample_size": len(sample_df),
                    "field_analyses": {},
                    "content_patterns": self.identify_content_patterns(sample_df),
                    "semantic_themes": self.extract_semantic_themes(sample_df),
                    "data_quality": self.assess_data_quality(sample_df),
                    "entity_extractions": self.extract_entities(sample_df)
                }
                
                for column in sample_df.columns:
                    field_analysis = self.analyze_field_content(sample_df[column], column)
                    content_analysis["field_analyses"][column] = field_analysis
                
                self.content_analysis["table_content_analysis"][table_key] = content_analysis
                self.logger.info(f"Analyzed content for {table_key}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing content for {dataset_id}.{table_name}: {e}")

    def analyze_field_content(self, field_series: pd.Series, field_name: str) -> Dict:
        field_analysis = {
            "field_name": field_name,
            "data_type": str(field_series.dtype),
            "null_count": field_series.isnull().sum(),
            "unique_count": field_series.nunique(),
            "completeness": 1 - (field_series.isnull().sum() / len(field_series)),
            "value_patterns": self.analyze_value_patterns(field_series),
            "content_classification": self.classify_field_content(field_series, field_name),
            "sample_values": self.get_sample_values(field_series)
        }
        
        return field_analysis

    def analyze_value_patterns(self, field_series: pd.Series) -> Dict:
        patterns = {
            "common_patterns": [],
            "format_consistency": "unknown",
            "value_distribution": "unknown"
        }
        
        non_null_values = field_series.dropna().astype(str)
        
        if len(non_null_values) > 0:
            pattern_checks = {
                "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                "ip_address": r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
                "uuid": r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                "timestamp": r'^\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}',
                "url": r'^https?://[^\s]+$',
                "phone": r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
            }
            
            for pattern_name, pattern_regex in pattern_checks.items():
                matches = non_null_values.str.match(pattern_regex, case=False).sum()
                if matches > len(non_null_values) * 0.8:
                    patterns["common_patterns"].append(pattern_name)
            
            unique_ratio = field_series.nunique() / len(field_series)
            if unique_ratio > 0.95:
                patterns["value_distribution"] = "highly_unique"
            elif unique_ratio > 0.5:
                patterns["value_distribution"] = "moderately_unique"
            else:
                patterns["value_distribution"] = "repetitive"
            
            value_lengths = non_null_values.str.len()
            if value_lengths.std() < value_lengths.mean() * 0.1:
                patterns["format_consistency"] = "consistent"
            else:
                patterns["format_consistency"] = "variable"
        
        return patterns

    def classify_field_content(self, field_series: pd.Series, field_name: str) -> str:
        non_null_values = field_series.dropna().astype(str)
        
        if len(non_null_values) == 0:
            return "empty_field"
        
        sample_text = " ".join(non_null_values.head(10).tolist())
        combined_text = f"{field_name} {sample_text}"
        
        content_labels = [
            "network_security_data",
            "user_authentication_data", 
            "system_event_data",
            "application_performance_data",
            "threat_intelligence_data",
            "compliance_audit_data",
            "infrastructure_monitoring_data",
            "business_analytics_data"
        ]
        
        classification = self.classifier(combined_text[:500], content_labels)
        return classification["labels"][0]

    def get_sample_values(self, field_series: pd.Series) -> List[str]:
        non_null_values = field_series.dropna()
        
        if len(non_null_values) > 0:
            sample_size = min(5, len(non_null_values))
            return [str(val) for val in non_null_values.head(sample_size).tolist()]
        
        return []

    def identify_content_patterns(self, sample_df: pd.DataFrame) -> Dict:
        patterns = {
            "table_structure": "unknown",
            "temporal_patterns": [],
            "categorical_patterns": [],
            "numerical_patterns": []
        }
        
        if len(sample_df.columns) > 20:
            patterns["table_structure"] = "wide_table"
        elif len(sample_df.columns) < 5:
            patterns["table_structure"] = "narrow_table"
        else:
            patterns["table_structure"] = "standard_table"
        
        for column in sample_df.columns:
            column_lower = column.lower()
            
            if any(keyword in column_lower for keyword in ['time', 'date', 'timestamp', 'created', 'modified']):
                patterns["temporal_patterns"].append(column)
            
            if sample_df[column].dtype == 'object' and sample_df[column].nunique() < len(sample_df) * 0.5:
                patterns["categorical_patterns"].append(column)
            
            if pd.api.types.is_numeric_dtype(sample_df[column]):
                patterns["numerical_patterns"].append(column)
        
        return patterns

    def extract_semantic_themes(self, sample_df: pd.DataFrame) -> List[str]:
        text_columns = []
        
        for column in sample_df.columns:
            if sample_df[column].dtype == 'object':
                sample_text = " ".join(sample_df[column].dropna().astype(str).head(20).tolist())
                if len(sample_text) > 50:
                    text_columns.append(sample_text)
        
        if text_columns:
            combined_text = " ".join(text_columns)[:1000]
            
            theme_labels = [
                "cybersecurity_monitoring",
                "network_traffic_analysis", 
                "user_behavior_tracking",
                "system_performance_monitoring",
                "threat_detection_response",
                "compliance_reporting",
                "infrastructure_management",
                "business_intelligence"
            ]
            
            classification = self.classifier(combined_text, theme_labels)
            return classification["labels"][:3]
        
        return []

    def assess_data_quality(self, sample_df: pd.DataFrame) -> Dict:
        quality_metrics = {
            "completeness": {},
            "consistency": {},
            "validity": {},
            "overall_score": 0
        }
        
        total_cells = sample_df.shape[0] * sample_df.shape[1]
        null_cells = sample_df.isnull().sum().sum()
        
        quality_metrics["completeness"]["overall"] = 1 - (null_cells / total_cells)
        
        for column in sample_df.columns:
            completeness = 1 - (sample_df[column].isnull().sum() / len(sample_df))
            quality_metrics["completeness"][column] = completeness
            
            if sample_df[column].dtype == 'object':
                non_null_values = sample_df[column].dropna().astype(str)
                if len(non_null_values) > 0:
                    avg_length = non_null_values.str.len().mean()
                    std_length = non_null_values.str.len().std()
                    consistency = 1 - (std_length / avg_length) if avg_length > 0 else 0
                    quality_metrics["consistency"][column] = min(consistency, 1)
        
        quality_scores = []
        quality_scores.append(quality_metrics["completeness"]["overall"])
        
        if quality_metrics["consistency"]:
            quality_scores.append(sum(quality_metrics["consistency"].values()) / len(quality_metrics["consistency"]))
        
        quality_metrics["overall_score"] = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return quality_metrics

    def extract_entities(self, sample_df: pd.DataFrame) -> Dict:
        entities = {
            "organizations": [],
            "persons": [],
            "locations": [],
            "technical_terms": []
        }
        
        text_columns = []
        for column in sample_df.columns:
            if sample_df[column].dtype == 'object':
                sample_text = " ".join(sample_df[column].dropna().astype(str).head(10).tolist())
                text_columns.append(sample_text)
        
        if text_columns:
            combined_text = " ".join(text_columns)[:1000]
            
            try:
                ner_results = self.ner_pipeline(combined_text)
                
                for entity in ner_results:
                    entity_text = entity["word"]
                    entity_label = entity["entity_group"]
                    
                    if entity_label == "ORG":
                        entities["organizations"].append(entity_text)
                    elif entity_label == "PER":
                        entities["persons"].append(entity_text)
                    elif entity_label == "LOC":
                        entities["locations"].append(entity_text)
                
                entities["organizations"] = list(set(entities["organizations"]))[:5]
                entities["persons"] = list(set(entities["persons"]))[:5]
                entities["locations"] = list(set(entities["locations"]))[:5]
                
            except Exception as e:
                self.logger.warning(f"Error extracting entities: {e}")
        
        return entities

    def save_content_analysis(self):
        output_file = Path("content_analysis_results.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.content_analysis, f, indent=2, default=str)
        
        self.logger.info(f"Content analysis saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else "."
    discovery_results = sys.argv[2] if len(sys.argv) > 2 else "bigquery_discovery_results.json"
    
    analyzer = ContentAnalyzer(project_root, discovery_results)
    results = analyzer.analyze_all_content()
    
    print(f"\nContent Analysis Summary:")
    print(f"- Tables analyzed: {len(results['table_content_analysis'])}")
    
    quality_scores = []
    for table_analysis in results['table_content_analysis'].values():
        overall_score = table_analysis.get('data_quality', {}).get('overall_score', 0)
        if overall_score > 0:
            quality_scores.append(overall_score)
    
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        print(f"- Average data quality score: {avg_quality:.2f}")
    
    print(f"- Analysis complete")