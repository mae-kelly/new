import pandas as pd
import re
import json
import base64
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from urllib.parse import unquote

@dataclass
class QuantumDetection:
    source: str
    metric: str
    confidence: float
    evidence: List[str]
    extraction_method: str

class QuantumAO1Engine:
    def __init__(self):
        self.visibility_extractors = {
            'network_presence': {
                'ipv4': r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)',
                'ipv6': r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}',
                'fqdn': r'[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*',
                'ports': r':\d{1,5}\b'
            },
            'endpoint_identity': {
                'hostname_enterprise': r'\b[A-Z]{2,4}-[A-Z0-9]+-\d+\b',
                'hostname_workstation': r'\bWS-[A-Z0-9]+-\d+\b',
                'hostname_server': r'\bSRV-[A-Z0-9]+-\d+\b',
                'device_guid': r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
            },
            'identity_context': {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'domain_user': r'\b[A-Za-z0-9_-]+\\[A-Za-z0-9_.-]+\b',
                'logon_event': r'\b(?:4624|4625|4648|4672|4768|4769|4776|4778|4779)\b'
            },
            'application_telemetry': {
                'http_status': r'\b[1-5][0-9]{2}\b',
                'api_endpoint': r'/api/[a-zA-Z0-9/_-]+',
                'url_path': r'https?://[^\s]+'
            },
            'cloud_infrastructure': {
                'aws_instance': r'\bi-[0-9a-f]{8,17}\b',
                'aws_volume': r'\bvol-[0-9a-f]{8,17}\b',
                'region_code': r'\b(?:us|eu|ap|ca)-(?:east|west|central|north|south)-[0-9][a-z]?\b'
            }
        }
    
    def quantum_extract(self, data_series: pd.Series, source_ref: str) -> List[QuantumDetection]:
        detections = []
        raw_content = self._extract_all_content(data_series)
        
        for category, patterns in self.visibility_extractors.items():
            for pattern_name, pattern in patterns.items():
                matches = self._find_pattern_matches(raw_content, pattern)
                if matches:
                    confidence = self._calculate_confidence(matches, len(raw_content))
                    if confidence > 0.3:
                        detections.append(QuantumDetection(
                            source=source_ref,
                            metric=category,
                            confidence=confidence,
                            evidence=matches[:5],
                            extraction_method=pattern_name
                        ))
        
        return sorted(detections, key=lambda x: x.confidence, reverse=True)
    
    def _extract_all_content(self, series: pd.Series) -> List[str]:
        content = []
        for value in series.dropna().head(1000):
            str_val = str(value)
            content.append(str_val)
            
            if self._is_json(str_val):
                content.extend(self._extract_json(str_val))
            elif self._is_xml(str_val):
                content.extend(self._extract_xml(str_val))
            elif self._is_base64(str_val):
                decoded = self._decode_base64(str_val)
                if decoded:
                    content.append(decoded)
        
        return content
    
    def _is_json(self, value: str) -> bool:
        return (value.strip().startswith('{') and value.strip().endswith('}')) or \
               (value.strip().startswith('[') and value.strip().endswith(']'))
    
    def _extract_json(self, value: str) -> List[str]:
        try:
            data = json.loads(value)
            extracted = []
            self._traverse_json(data, extracted)
            return extracted
        except:
            return []
    
    def _traverse_json(self, obj: Any, extracted: List[str]):
        if isinstance(obj, dict):
            for k, v in obj.items():
                extracted.append(str(k))
                self._traverse_json(v, extracted)
        elif isinstance(obj, list):
            for item in obj:
                self._traverse_json(item, extracted)
        else:
            extracted.append(str(obj))
    
    def _is_xml(self, value: str) -> bool:
        return value.strip().startswith('<') and value.strip().endswith('>')
    
    def _extract_xml(self, value: str) -> List[str]:
        try:
            root = ET.fromstring(value)
            content = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    content.append(elem.text.strip())
                content.append(elem.tag)
            return content
        except:
            return []
    
    def _is_base64(self, value: str) -> bool:
        if len(value) < 4 or len(value) % 4 != 0:
            return False
        try:
            base64.b64decode(value, validate=True)
            return True
        except:
            return False
    
    def _decode_base64(self, value: str) -> Optional[str]:
        try:
            return base64.b64decode(value).decode('utf-8', errors='ignore')
        except:
            return None
    
    def _find_pattern_matches(self, content: List[str], pattern: str) -> List[str]:
        matches = []
        for text in content:
            try:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found[:10])
            except:
                continue
        return list(set(matches))
    
    def _calculate_confidence(self, matches: List[str], total_content: int) -> float:
        if not matches or total_content == 0:
            return 0.0
        
        base_confidence = min(len(matches) / max(total_content * 0.01, 1), 1.0)
        uniqueness_bonus = len(set(matches)) / max(len(matches), 1)
        
        return min(base_confidence * uniqueness_bonus, 1.0)

from industry_patterns import IndustryPatternEngine

class QuantumAO1Engine:
    def __init__(self):
        self.visibility_extractors = {
            'network_presence': {
                'ipv4': r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)',
                'ipv6': r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}',
                'fqdn': r'[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*',
                'ports': r':\d{1,5}\b'
            },
            'endpoint_identity': {
                'hostname_enterprise': r'\b[A-Z]{2,4}-[A-Z0-9]+-\d+\b',
                'hostname_workstation': r'\bWS-[A-Z0-9]+-\d+\b',
                'hostname_server': r'\bSRV-[A-Z0-9]+-\d+\b',
                'device_guid': r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
            },
            'identity_context': {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'domain_user': r'\b[A-Za-z0-9_-]+\\[A-Za-z0-9_.-]+\b',
                'logon_event': r'\b(?:4624|4625|4648|4672|4768|4769|4776|4778|4779)\b'
            },
            'application_telemetry': {
                'http_status': r'\b[1-5][0-9]{2}\b',
                'api_endpoint': r'/api/[a-zA-Z0-9/_-]+',
                'url_path': r'https?://[^\s]+'
            },
            'cloud_infrastructure': {
                'aws_instance': r'\bi-[0-9a-f]{8,17}\b',
                'aws_volume': r'\bvol-[0-9a-f]{8,17}\b',
                'region_code': r'\b(?:us|eu|ap|ca)-(?:east|west|central|north|south)-[0-9][a-z]?\b'
            }
        }
        self.industry_engine = IndustryPatternEngine()
    
    def quantum_extract_enhanced(self, data_series, source_ref: str):
        base_detections = self.quantum_extract(data_series, source_ref)
        industry_context = self.industry_engine.detect_industry_context(base_detections)
        enhanced_detections = self.industry_engine.enhance_detections_with_industry_context(
            base_detections, industry_context
        )
        return enhanced_detections, industry_context

from industry_patterns import IndustryPatternEngine

class QuantumAO1Engine:
    def __init__(self):
        self.visibility_extractors = {
            'network_presence': {
                'ipv4': r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)',
                'ipv6': r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}',
                'fqdn': r'[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*',
                'ports': r':\d{1,5}\b'
            },
            'endpoint_identity': {
                'hostname_enterprise': r'\b[A-Z]{2,4}-[A-Z0-9]+-\d+\b',
                'hostname_workstation': r'\bWS-[A-Z0-9]+-\d+\b',
                'hostname_server': r'\bSRV-[A-Z0-9]+-\d+\b',
                'device_guid': r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
            },
            'identity_context': {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'domain_user': r'\b[A-Za-z0-9_-]+\\[A-Za-z0-9_.-]+\b',
                'logon_event': r'\b(?:4624|4625|4648|4672|4768|4769|4776|4778|4779)\b'
            },
            'application_telemetry': {
                'http_status': r'\b[1-5][0-9]{2}\b',
                'api_endpoint': r'/api/[a-zA-Z0-9/_-]+',
                'url_path': r'https?://[^\s]+'
            },
            'cloud_infrastructure': {
                'aws_instance': r'\bi-[0-9a-f]{8,17}\b',
                'aws_volume': r'\bvol-[0-9a-f]{8,17}\b',
                'region_code': r'\b(?:us|eu|ap|ca)-(?:east|west|central|north|south)-[0-9][a-z]?\b'
            }
        }
        self.industry_engine = IndustryPatternEngine()
    
    def quantum_extract_enhanced(self, data_series, source_ref: str):
        base_detections = self.quantum_extract(data_series, source_ref)
        industry_context = self.industry_engine.detect_industry_context(base_detections)
        enhanced_detections = self.industry_engine.enhance_detections_with_industry_context(
            base_detections, industry_context
        )
        return enhanced_detections, industry_context
