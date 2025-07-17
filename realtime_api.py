from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
from typing import Dict, List
import time

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.job_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if job_id:
            if job_id not in self.job_connections:
                self.job_connections[job_id] = []
            self.job_connections[job_id].append(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str = None):
        self.active_connections.remove(websocket)
        if job_id and job_id in self.job_connections:
            self.job_connections[job_id].remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_job(self, message: str, job_id: str):
        if job_id in self.job_connections:
            for connection in self.job_connections[job_id]:
                await connection.send_text(message)

manager = ConnectionManager()

async def get_scan_progress_realtime(job_id: str):
    job = db.get_scan_job(job_id)
    if not job:
        return None
    
    return {
        "job_id": job_id,
        "progress_percentage": min((job.processed_tables / max(job.total_tables, 1)) * 100, 100),
        "current_status": job.status,
        "tables_processed": job.processed_tables,
        "total_tables": job.total_tables,
        "detections_found": job.mappings_found,
        "timestamp": time.time(),
        "estimated_completion": time.time() + ((job.total_tables - job.processed_tables) * 2)
    }

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
    user_jobs = db.get_user_jobs(current_user.id, limit=10)
    recent_results = []
    for job in user_jobs:
        if job.status == "completed":
            results = db.get_job_results(job.id)
            recent_results.extend(results)
    
    coverage_by_area = {}
    for result in recent_results:
        area = result.metric_type
        if area not in coverage_by_area:
            coverage_by_area[area] = []
        coverage_by_area[area].append(result.confidence_score)
    
    area_averages = {area: sum(scores)/len(scores) for area, scores in coverage_by_area.items() if scores}
    
    return {
        "total_scans": len(user_jobs),
        "active_scans": len([j for j in user_jobs if j.status == "running"]),
        "total_detections": sum(j.mappings_found for j in user_jobs),
        "coverage_areas": area_averages,
        "recent_activity": [{"job_id": j.id, "status": j.status, "started": j.started_at} for j in user_jobs[:5]]
    }
