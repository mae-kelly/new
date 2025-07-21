import re
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class IndustryPattern:
    name: str
    pattern: str
    category: str
    confidence_multiplier: float
    description: str

class IndustryPatternEngine:
    def __init__(self):
        self.industry_patterns = {
            'financial': [
                IndustryPattern('swift_code', r'\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b', 'identity_context', 1.5, 'SWIFT bank codes'),
                IndustryPattern('iban', r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b', 'identity_context', 1.8, 'IBAN numbers'),
                IndustryPattern('card_number', r'\b4[0-9]{12}(?:[0-9]{3})?\b', 'identity_context', 2.0, 'Credit card numbers'),
                IndustryPattern('trading_symbol', r'\b[A-Z]{1,5}:[A-Z]{1,8}\b', 'application_telemetry', 1.3, 'Trading symbols'),
            ],
            'healthcare': [
                IndustryPattern('patient_id', r'\bPT[0-9]{6,8}\b', 'identity_context', 1.7, 'Patient identifiers'),
                IndustryPattern('medical_record', r'\bMRN[0-9]{6,10}\b', 'identity_context', 1.8, 'Medical record numbers'),
                IndustryPattern('npi', r'\b[0-9]{10}\b', 'identity_context', 1.6, 'National Provider Identifier'),
                IndustryPattern('phi_marker', r'\b(?:SSN|DOB|MRN|PHI|HIPAA)\b', 'identity_context', 1.4, 'PHI indicators'),
            ],
            'retail': [
                IndustryPattern('customer_id', r'\bCUST[0-9]{6,10}\b', 'identity_context', 1.4, 'Customer IDs'),
                IndustryPattern('order_id', r'\bORD[0-9A-F]{6,12}\b', 'application_telemetry', 1.3, 'Order identifiers'),
                IndustryPattern('sku', r'\b[A-Z]{2,4}[0-9]{4,8}\b', 'application_telemetry', 1.2, 'Product SKUs'),
                IndustryPattern('barcode', r'\b[0-9]{8,14}\b', 'application_telemetry', 1.1, 'Product barcodes'),
            ],
            'manufacturing': [
                IndustryPattern('asset_tag', r'\bAT[0-9]{6,10}\b', 'endpoint_identity', 1.6, 'Asset tags'),
                IndustryPattern('serial_number', r'\bSN[A-Z0-9]{8,12}\b', 'endpoint_identity', 1.5, 'Serial numbers'),
                IndustryPattern('lot_number', r'\bLOT[0-9A-F]{6,10}\b', 'application_telemetry', 1.3, 'Manufacturing lots'),
            ],
            'energy': [
                IndustryPattern('meter_id', r'\bMTR[0-9]{6,12}\b', 'endpoint_identity', 1.5, 'Meter identifiers'),
                IndustryPattern('scada_tag', r'\b[A-Z]{2,4}_[0-9]{3,6}\b', 'endpoint_identity', 1.7, 'SCADA tags'),
                IndustryPattern('grid_node', r'\bGN[0-9]{4,8}\b', 'network_presence', 1.6, 'Grid nodes'),
            ]
        }
        
        self.compliance_patterns = {
            'pci_dss': [
                IndustryPattern('card_data', r'\b4[0-9]{12}(?:[0-9]{3})?\b', 'identity_context', 2.0, 'Payment card data'),
                IndustryPattern('cardholder_data', r'\bCHD[0-9]{6,10}\b', 'identity_context', 1.8, 'Cardholder data'),
            ],
            'sox': [
                IndustryPattern('financial_control', r'\bFC[0-9]{4,8}\b', 'application_telemetry', 1.5, 'Financial controls'),
                IndustryPattern('audit_trail', r'\bAT[0-9]{6,10}\b', 'application_telemetry', 1.6, 'Audit trails'),
            ],
            'gdpr': [
                IndustryPattern('eu_citizen_id', r'\b[A-Z]{2}[0-9]{6,12}\b', 'identity_context', 1.7, 'EU citizen IDs'),
                IndustryPattern('consent_record', r'\bCON[0-9A-F]{8,12}\b', 'identity_context', 1.5, 'Consent records'),
            ]
        }

    def detect_industry_context(self, detections: List[Any]) -> Dict[str, Any]:
        industry_scores = {}
        
        for industry, patterns in self.industry_patterns.items():
            score = 0
            pattern_matches = 0
            
            for detection in detections:
                for pattern in patterns:
                    if any(re.search(pattern.pattern, str(evidence), re.IGNORECASE) 
                           for evidence in detection.evidence):
                        score += pattern.confidence_multiplier
                        pattern_matches += 1
            
            if pattern_matches > 0:
                industry_scores[industry] = {
                    'score': score,
                    'pattern_matches': pattern_matches,
                    'confidence': min(score / 10.0, 1.0)
                }
        
        detected_industry = max(industry_scores.items(), key=lambda x: x[1]['score'])[0] if industry_scores else 'generic'
        
        return {
            'detected_industry': detected_industry,
            'industry_scores': industry_scores,
            'compliance_relevance': self._assess_compliance_relevance(detected_industry),
            'recommended_patterns': self._get_recommended_patterns(detected_industry)
        }

    def _assess_compliance_relevance(self, industry: str) -> List[str]:
        compliance_mapping = {
            'financial': ['SOX', 'PCI_DSS', 'GDPR', 'Basel_III'],
            'healthcare': ['HIPAA', 'GDPR', 'HITECH'],
            'retail': ['PCI_DSS', 'GDPR', 'CCPA'],
            'manufacturing': ['ISO27001', 'NIST', 'GDPR'],
            'energy': ['NERC_CIP', 'ICS_CERT', 'NIST']
        }
        return compliance_mapping.get(industry, ['ISO27001', 'NIST'])

    def _get_recommended_patterns(self, industry: str) -> List[Dict]:
        patterns = self.industry_patterns.get(industry, [])
        return [{'name': p.name, 'description': p.description, 'category': p.category} for p in patterns]

    def enhance_detections_with_industry_context(self, detections: List[Any], industry_context: Dict) -> List[Any]:
        enhanced = []
        industry = industry_context['detected_industry']
        patterns = self.industry_patterns.get(industry, [])
        
        for detection in detections:
            enhanced_detection = detection
            
            for pattern in patterns:
                if pattern.category == detection.metric:
                    for evidence in detection.evidence:
                        if re.search(pattern.pattern, str(evidence), re.IGNORECASE):
                            enhanced_detection.confidence *= pattern.confidence_multiplier
                            enhanced_detection.confidence = min(enhanced_detection.confidence, 1.0)
                            break
            
            enhanced.append(enhanced_detection)
        
        return enhanced
