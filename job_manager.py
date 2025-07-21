import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from database import db, ScanJob, ScanResult, AuditLog
from secure_scanner import SecureBigQueryScanner
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class JobManager:
    def __init__(self):
        self.active_jobs = {}
        
    async def create_scan_job(self, user_id: str, project_ids: List[str]) -> str:
        job = ScanJob(
            id=str(uuid.uuid4()),
            user_id=user_id,
            project_ids=project_ids,
            status="pending"
        )
        
        job_id = db.create_scan_job(job)
        
        audit = AuditLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            action="scan_job_created",
            resource_type="scan_job",
            resource_id=job_id,
            details={"project_ids": project_ids}
        )
        db.log_audit(audit)
        
        asyncio.create_task(self._execute_scan_job(job_id, user_id))
        
        return job_id
    
    async def _execute_scan_job(self, job_id: str, user_id: str):
        try:
            db.update_scan_job(job_id, {"status": "running"})
            
            scanner = SecureBigQueryScanner(user_id)
            results = await scanner.scan_user_projects()
            
            total_mappings = len(results.get('mappings', []))
            
            db.update_scan_job(job_id, {
                "total_tables": total_mappings,
                "processed_tables": total_mappings,
                "mappings_found": total_mappings
            })
            
            for mapping in results.get('mappings', []):
                result = ScanResult(
                    id=str(uuid.uuid4()),
                    job_id=job_id,
                    project_id=mapping.get('project', ''),
                    dataset_id=mapping.get('dataset', ''),
                    table_id=mapping.get('table', ''),
                    column_name=mapping.get('column'),
                    metric_type=mapping.get('metric'),
                    confidence_score=mapping.get('confidence', 0.0),
                    pattern_type=mapping.get('pattern_type'),
                    detection_methods=mapping.get('detection_methods', [])
                )
                db.create_scan_result(result)
            
            results_path = f"results/{job_id}.json"
            Path("results").mkdir(exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            db.update_scan_job(job_id, {
                "results_path": results_path,
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            db.update_scan_job(job_id, {
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.utcnow().isoformat()
            })
            
        finally:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        job = db.get_scan_job(job_id)
        if not job:
            return None
            
        return {
            "id": job.id,
            "status": job.status,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "total_tables": job.total_tables,
            "processed_tables": job.processed_tables,
            "mappings_found": job.mappings_found,
            "error_message": job.error_message
        }
    
    def get_user_jobs(self, user_id: str) -> List[Dict]:
        jobs = db.get_user_jobs(user_id)
        return [
            {
                "id": job.id,
                "status": job.status,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "mappings_found": job.mappings_found
            }
            for job in jobs
        ]
    
    def get_job_results(self, job_id: str) -> List[Dict]:
        results = db.get_job_results(job_id)
        return [
            {
                "id": result.id,
                "project_id": result.project_id,
                "dataset_id": result.dataset_id,
                "table_id": result.table_id,
                "column_name": result.column_name,
                "metric_type": result.metric_type,
                "confidence_score": result.confidence_score,
                "pattern_type": result.pattern_type,
                "detection_methods": result.detection_methods
            }
            for result in results
        ]

job_manager = JobManager()
