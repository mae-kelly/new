from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import logging
import uuid
from datetime import datetime, timezone
from auth import AuthManager, ProjectAccess
from database import db, User, ScanJob, ProjectAccess as ProjectAccessModel, AuditLog
from job_manager import job_manager
from redis_client import cache_client
from secure_config import SecurityConfig
import structlog

logger = structlog.get_logger()

app = FastAPI(title="BigQuery Visibility Scanner", version="1.0.0")
security = HTTPBearer(auto_error=False)
auth_manager = AuthManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ScanRequest(BaseModel):
    project_ids: List[str]

class ProjectAccessGrant(BaseModel):
    user_email: str
    project_id: str
    access_level: str = "read"

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required"
        )
    
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    
    user = db.get_user_by_id(payload.get('user_id'))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user

def log_audit(user_id: str, action: str, resource_type: str = None, resource_id: str = None, 
              details: Dict = None, request: Request = None):
    audit = AuditLog(
        id=str(uuid.uuid4()),
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details or {},
        ip_address=request.client.host if request else None,
        user_agent=request.headers.get("user-agent") if request else None
    )
    db.log_audit(audit)

@app.post("/auth/register")
async def register(user_data: UserCreate, request: Request):
    if not auth_manager.verify_domain(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email domain not allowed"
        )
    
    existing_user = db.get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    hashed_password = auth_manager.hash_password(user_data.password)
    user = User(
        id=str(uuid.uuid4()),
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name
    )
    
    user_id = db.create_user(user)
    token = auth_manager.create_token(user_id, user_data.email)
    
    log_audit(user_id, "user_registered", "user", user_id, {"email": user_data.email}, request)
    
    return {"access_token": token, "token_type": "bearer", "user_id": user_id}

@app.post("/auth/login")
async def login(login_data: UserLogin, request: Request):
    user = db.get_user_by_email(login_data.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    if not auth_manager.verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    token = auth_manager.create_token(user.id, user.email)
    
    db.conn.execute("UPDATE users SET last_login = ? WHERE id = ?", 
                   [datetime.now(timezone.utc).isoformat(), user.id])
    
    log_audit(user.id, "user_login", "user", user.id, {"email": user.email}, request)
    
    return {"access_token": token, "token_type": "bearer", "user_id": user.id}

@app.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at
    }

@app.post("/scans")
async def create_scan(
    scan_request: ScanRequest,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    if not cache_client.rate_limit_check(current_user.id, "create_scan", limit=3, window=3600):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded - max 3 scans per hour"
        )
    
    if len(scan_request.project_ids) > SecurityConfig.MAX_PROJECTS_PER_SCAN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many projects requested. Max: {SecurityConfig.MAX_PROJECTS_PER_SCAN}"
        )
    
    job_id = await job_manager.create_scan_job(current_user.id, scan_request.project_ids)
    
    log_audit(current_user.id, "scan_created", "scan_job", job_id, 
              {"project_ids": scan_request.project_ids}, request)
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/scans/{job_id}")
async def get_scan_status(job_id: str, current_user: User = Depends(get_current_user)):
    job = db.get_scan_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return job_manager.get_job_status(job_id)

@app.get("/scans/{job_id}/results")
async def get_scan_results(job_id: str, current_user: User = Depends(get_current_user)):
    job = db.get_scan_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job not completed"
        )
    
    return job_manager.get_job_results(job_id)

@app.get("/scans")
async def list_scans(current_user: User = Depends(get_current_user)):
    return job_manager.get_user_jobs(current_user.id)

@app.get("/projects")
async def list_projects(current_user: User = Depends(get_current_user)):
    user_projects = db.get_user_projects(current_user.id)
    return {"projects": user_projects}

@app.post("/admin/project-access")
async def grant_project_access(
    access_grant: ProjectAccessGrant,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    target_user = db.get_user_by_email(access_grant.user_email)
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if db.check_project_access(target_user.id, access_grant.project_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already has access to this project"
        )
    
    access = ProjectAccessModel(
        id=str(uuid.uuid4()),
        user_id=target_user.id,
        project_id=access_grant.project_id,
        access_level=access_grant.access_level,
        granted_by=current_user.id
    )
    
    access_id = db.grant_project_access(access)
    
    log_audit(current_user.id, "project_access_granted", "project_access", access_id,
              {"target_user": target_user.email, "project_id": access_grant.project_id}, request)
    
    return {"message": "Access granted", "access_id": access_id}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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

