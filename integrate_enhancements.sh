#!/bin/bash

set -e

echo "Integrating AO1 Scanner Enhancements..."

echo "Adding WebSocket support to API..."
cat >> api.py << 'PYTHON'

from realtime_api import manager, get_scan_progress_realtime

@app.websocket("/ws/scan/{job_id}")
async def websocket_scan_progress(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        while True:
            progress = await get_scan_progress_realtime(job_id)
            if progress:
                await manager.send_personal_message(json.dumps(progress), websocket)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)

@app.get("/scans/{job_id}/progress")
async def get_scan_progress_http(job_id: str, current_user = Depends(get_current_user)):
    progress = await get_scan_progress_realtime(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found")
    return progress

@app.get("/dashboard/stats")
async def get_dashboard_stats(current_user = Depends(get_current_user)):
    from realtime_api import get_dashboard_stats
    return await get_dashboard_stats(current_user)

@app.get("/reports/executive/{job_id}")
async def get_executive_report(job_id: str, current_user = Depends(get_current_user)):
    from executive_reports import ExecutiveReportGenerator
    
    job = db.get_scan_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    results = job_manager.get_job_results(job_id)
    scan_results = {"visibility_detections": results, "ao1_assessment": {}}
    
    generator = ExecutiveReportGenerator()
    report = generator.generate_executive_summary(scan_results)
    return report

@app.get("/performance/summary")
async def get_performance_summary(current_user = Depends(get_current_user)):
    from performance_monitor import performance_monitor
    return performance_monitor.get_performance_summary()

PYTHON

echo "Enhancing quantum engine with industry patterns..."
cat >> quantum_engine.py << 'PYTHON'

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
PYTHON

echo "Adding resilience to scanner..."
cat >> content_value_scanner.py << 'PYTHON'

from resilience_manager import resilience_manager
from performance_monitor import performance_monitor

class ValueContentScanner:
    async def _scan_table_values_resilient(self, project_id: str, dataset_id: str, table_id: str):
        resource_id = f"{project_id}.{dataset_id}.{table_id}"
        operation_id = f"scan_{int(time.time())}_{resource_id}"
        
        metrics = performance_monitor.start_operation(operation_id, "table_scan", resource_id)
        
        try:
            async def scan_operation():
                return await self._scan_table_values(project_id, dataset_id, table_id)
            
            result = await resilience_manager.execute_with_resilience(
                scan_operation, resource_id, "table_scan"
            )
            
            performance_monitor.update_operation(operation_id, 
                rows_scanned=len
