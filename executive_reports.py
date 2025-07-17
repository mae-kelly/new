import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

class ExecutiveReportGenerator:
    def __init__(self):
        self.compliance_frameworks = {
            'SOC2': {
                'requirements': ['network_monitoring', 'access_logging', 'system_monitoring', 'change_management'],
                'description': 'Service Organization Control 2',
                'focus_areas': ['security', 'availability', 'processing_integrity']
            },
            'ISO27001': {
                'requirements': ['asset_inventory', 'access_control', 'incident_detection', 'risk_management'],
                'description': 'International Organization for Standardization 27001',
                'focus_areas': ['information_security', 'risk_management', 'business_continuity']
            },
            'NIST_CSF': {
                'requirements': ['identify', 'protect', 'detect', 'respond', 'recover'],
                'description': 'NIST Cybersecurity Framework',
                'focus_areas': ['cybersecurity', 'risk_management', 'incident_response']
            },
            'PCI_DSS': {
                'requirements': ['network_segmentation', 'access_control', 'monitoring', 'encryption'],
                'description': 'Payment Card Industry Data Security Standard',
                'focus_areas': ['payment_security', 'data_protection', 'access_control']
            }
        }
        
        self.risk_thresholds = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }

    def generate_executive_summary(self, scan_results: Dict) -> Dict:
        overall_score = self._calculate_overall_ao1_score(scan_results)
        key_findings = self._extract_key_findings(scan_results)
        business_impact = self._assess_business_impact(scan_results)
        
        return {
            'executive_summary': {
                'ao1_readiness_score': overall_score,
                'readiness_status': self._get_readiness_status(overall_score),
                'key_findings': key_findings,
                'business_impact': business_impact,
                'investment_recommendation': self._generate_investment_recommendation(overall_score, scan_results),
                'timeline_estimate': self._estimate_implementation_timeline(overall_score),
                'budget_estimate': self._estimate_budget_requirements(scan_results)
            },
            'compliance_assessment': self._map_to_compliance_frameworks(scan_results),
            'risk_analysis': self._analyze_security_risks(scan_results),
            'implementation_roadmap': self._create_implementation_roadmap(scan_results),
            'technical_details': self._summarize_technical_findings(scan_results),
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'scan_scope': f"{len(scan_results.get('visibility_detections', []))} data sources analyzed",
                'confidence_level': 'high' if overall_score > 0.7 else 'medium' if overall_score > 0.4 else 'low'
            }
        }

    def _calculate_overall_ao1_score(self, scan_results: Dict) -> float:
        ao1_assessment = scan_results.get('ao1_assessment', {})
        overall_readiness = ao1_assessment.get('overall_ao1_readiness', {})
        return overall_readiness.get('percentage', 0) / 100.0

    def _get_readiness_status(self, score: float) -> str:
        if score >= 0.8:
            return 'READY'
        elif score >= 0.6:
            return 'NEARLY_READY'
        elif score >= 0.4:
            return 'PARTIAL'
        else:
            return 'NOT_READY'

    def _extract_key_findings(self, scan_results: Dict) -> List[Dict]:
        findings = []
        
        detections = scan_results.get('visibility_detections', [])
        high_confidence_detections = [d for d in detections if d.get('confidence', 0) > 0.8]
        
        coverage_breakdown = scan_results.get('ao1_assessment', {}).get('coverage_breakdown', {})
        
        findings.append({
            'finding': f"Discovered {len(high_confidence_detections)} high-confidence security data sources",
            'impact': 'positive',
            'detail': f"Strong visibility foundation across {len(set(d.get('project') for d in high_confidence_detections))} projects"
        })
        
        weak_areas = [area for area, data in coverage_breakdown.items() if data.get('status') == 'NOT_READY']
        if weak_areas:
            findings.append({
                'finding': f"Gaps identified in {len(weak_areas)} visibility areas",
                'impact': 'negative',
                'detail': f"Areas needing attention: {', '.join(weak_areas)}"
            })
        
        projects_with_data = len(set(d.get('project') for d in detections))
        if projects_with_data > 10:
            findings.append({
                'finding': f"Comprehensive data coverage across {projects_with_data} projects",
                'impact': 'positive',
                'detail': "Enterprise-wide visibility achieved"
            })
        
        return findings[:5]

    def _assess_business_impact(self, scan_results: Dict) -> Dict:
        overall_score = self._calculate_overall_ao1_score(scan_results)
        detections_count = len(scan_results.get('visibility_detections', []))
        
        threat_detection_improvement = min(overall_score * 100, 95)
        incident_response_improvement = min((overall_score * 0.8) * 100, 90)
        compliance_readiness = min((overall_score * 0.9) * 100, 92)
        
        return {
            'threat_detection_improvement': f"{threat_detection_improvement:.0f}%",
            'incident_response_time_reduction': f"{incident_response_improvement:.0f}%",
            'compliance_readiness': f"{compliance_readiness:.0f}%",
            'estimated_risk_reduction': self._calculate_risk_reduction(overall_score),
            'roi_timeframe': '6-12 months' if overall_score > 0.6 else '12-18 months'
        }

    def _calculate_risk_reduction(self, ao1_score: float) -> str:
        base_risk_reduction = ao1_score * 70
        return f"{base_risk_reduction:.0f}%"

    def _generate_investment_recommendation(self, score: float, scan_results: Dict) -> str:
        if score >= 0.8:
            return "Proceed with AO1 implementation immediately. Strong data foundation detected."
        elif score >= 0.6:
            return "Begin AO1 implementation with focused data collection improvements in identified gaps."
        elif score >= 0.4:
            return "Invest in data infrastructure improvements before full AO1 implementation."
        else:
            return "Significant data collection improvements required before AO1 implementation."

    def _estimate_implementation_timeline(self, score: float) -> Dict:
        if score >= 0.8:
            return {"phase1": "2-4 weeks", "full_implementation": "8-12 weeks", "total": "3 months"}
        elif score >= 0.6:
            return {"phase1": "4-6 weeks", "full_implementation": "12-16 weeks", "total": "4-5 months"}
        elif score >= 0.4:
            return {"phase1": "6-8 weeks", "full_implementation": "16-24 weeks", "total": "6-8 months"}
        else:
            return {"phase1": "8-12 weeks", "full_implementation": "24-36 weeks", "total": "9-12 months"}

    def _estimate_budget_requirements(self, scan_results: Dict) -> Dict:
        projects_count = len(set(d.get('project') for d in scan_results.get('visibility_detections', [])))
        base_cost = min(projects_count * 5000, 200000)
        
        return {
            "initial_setup": f"${base_cost:,}",
            "annual_operational": f"${base_cost * 0.3:,.0f}",
            "total_first_year": f"${base_cost * 1.3:,.0f}"
        }

    def _map_to_compliance_frameworks(self, scan_results: Dict) -> Dict:
        compliance_scores = {}
        coverage_breakdown = scan_results.get('ao1_assessment', {}).get('coverage_breakdown', {})
        
        for framework, config in self.compliance_frameworks.items():
            met_requirements = 0
            total_requirements = len(config['requirements'])
            
            for requirement in config['requirements']:
                if self._requirement_satisfied(requirement, coverage_breakdown):
                    met_requirements += 1
            
            score = (met_requirements / total_requirements) * 100
            
            compliance_scores[framework] = {
                'score': score,
                'status': 'COMPLIANT' if met_requirements == total_requirements else 'PARTIAL' if met_requirements > 0 else 'NON_COMPLIANT',
                'met_requirements': met_requirements,
                'total_requirements': total_requirements,
                'gaps': [req for req in config['requirements'] if not self._requirement_satisfied(req, coverage_breakdown)],
                'description': config['description'],
                'focus_areas': config['focus_areas']
            }
        
        return compliance_scores

    def _requirement_satisfied(self, requirement: str, coverage_breakdown: Dict) -> bool:
        requirement_mapping = {
            'network_monitoring': 'network_coverage',
            'access_logging': 'identity_coverage',
            'system_monitoring': 'endpoint_coverage',
            'change_management': 'application_coverage',
            'asset_inventory': 'endpoint_coverage',
            'access_control': 'identity_coverage',
            'incident_detection': 'application_coverage',
            'risk_management': 'application_coverage',
            'identify': 'endpoint_coverage',
            'protect': 'identity_coverage',
            'detect': 'application_coverage',
            'respond': 'application_coverage',
            'recover': 'cloud_coverage',
            'network_segmentation': 'network_coverage',
            'monitoring': 'application_coverage',
            'encryption': 'application_coverage'
        }
        
        mapped_area = requirement_mapping.get(requirement)
        if not mapped_area:
            return False
        
        area_data = coverage_breakdown.get(mapped_area, {})
        return area_data.get('status') in ['READY', 'PARTIAL']

    def _analyze_security_risks(self, scan_results: Dict) -> Dict:
        coverage_breakdown = scan_results.get('ao1_assessment', {}).get('coverage_breakdown', {})
        overall_score = self._calculate_overall_ao1_score(scan_results)
        
        risks = []
        
        for area, data in coverage_breakdown.items():
            if data.get('status') == 'NOT_READY':
                risk_level = 'HIGH' if area in ['network_coverage', 'identity_coverage'] else 'MEDIUM'
                risks.append({
                    'area': area,
                    'risk_level': risk_level,
                    'description': f"Limited visibility in {area.replace('_', ' ')}",
                    'impact': 'Could delay threat detection and incident response'
                })
        
        return {
            'overall_risk_level': 'LOW' if overall_score > 0.7 else 'MEDIUM' if overall_score > 0.4 else 'HIGH',
            'identified_risks': risks,
            'risk_mitigation_priority': self._prioritize_risk_mitigation(risks),
            'recommended_actions': self._generate_risk_actions(risks)
        }

    def _prioritize_risk_mitigation(self, risks: List[Dict]) -> List[str]:
        high_risks = [r['area'] for r in risks if r['risk_level'] == 'HIGH']
        medium_risks = [r['area'] for r in risks if r['risk_level'] == 'MEDIUM']
        return high_risks + medium_risks

    def _generate_risk_actions(self, risks: List[Dict]) -> List[str]:
        actions = []
        for risk in risks[:3]:
            area = risk['area'].replace('_coverage', '').replace('_', ' ')
            actions.append(f"Enhance {area} data collection and monitoring")
        return actions

    def _create_implementation_roadmap(self, scan_results: Dict) -> Dict:
        coverage_breakdown = scan_results.get('ao1_assessment', {}).get('coverage_breakdown', {})
        overall_score = self._calculate_overall_ao1_score(scan_results)
        
        phases = []
        
        if overall_score < 0.6:
            phases.append({
                'phase': 'Foundation',
                'duration': '4-8 weeks',
                'objectives': ['Establish baseline data collection', 'Implement core monitoring'],
                'success_criteria': 'Achieve 60% AO1 readiness score'
            })
        
        phases.append({
            'phase': 'Enhancement',
            'duration': '6-10 weeks',
            'objectives': ['Fill identified gaps', 'Optimize detection rules'],
            'success_criteria': 'Achieve 80% AO1 readiness score'
        })
        
        phases.append({
            'phase': 'Optimization',
            'duration': '4-6 weeks',
            'objectives': ['Fine-tune detection logic', 'Implement automation'],
            'success_criteria': 'Achieve operational excellence'
        })
        
        return {
            'phases': phases,
            'total_timeline': f"{sum(int(p['duration'].split('-')[1].split()[0]) for p in phases)} weeks maximum",
            'critical_dependencies': self._identify_dependencies(coverage_breakdown),
            'success_metrics': self._define_success_metrics()
        }

    def _identify_dependencies(self, coverage_breakdown: Dict) -> List[str]:
        dependencies = []
        not_ready_areas = [area for area, data in coverage_breakdown.items() if data.get('status') == 'NOT_READY']
        
        if 'network_coverage' in not_ready_areas:
            dependencies.append('Network monitoring infrastructure')
        if 'identity_coverage' in not_ready_areas:
            dependencies.append('Identity and access management integration')
        if 'endpoint_coverage' in not_ready_areas:
            dependencies.append('Endpoint detection and response tools')
        
        return dependencies

    def _define_success_metrics(self) -> List[str]:
        return [
            'AO1 readiness score > 85%',
            'All 5 visibility areas in READY status',
            'Mean time to detection < 15 minutes',
            'False positive rate < 5%'
        ]

    def _summarize_technical_findings(self, scan_results: Dict) -> Dict:
        detections = scan_results.get('visibility_detections', [])
        
        return {
            'total_data_sources': len(detections),
            'high_confidence_sources': len([d for d in detections if d.get('confidence', 0) > 0.8]),
            'projects_analyzed': len(set(d.get('project') for d in detections)),
            'coverage_distribution': self._calculate_coverage_distribution(detections),
            'data_quality_assessment': self._assess_data_quality(detections)
        }

    def _calculate_coverage_distribution(self, detections: List[Dict]) -> Dict:
        distribution = {}
        for detection in detections:
            metric = detection.get('metric', 'unknown')
            if metric not in distribution:
                distribution[metric] = 0
            distribution[metric] += 1
        return distribution

    def _assess_data_quality(self, detections: List[Dict]) -> str:
        if not detections:
            return 'insufficient_data'
        
        avg_confidence = np.mean([d.get('confidence', 0) for d in detections])
        
        if avg_confidence > 0.8:
            return 'excellent'
        elif avg_confidence > 0.6:
            return 'good'
        elif avg_confidence > 0.4:
            return 'fair'
        else:
            return 'poor'
