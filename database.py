import duckdb
import uuid
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

DATABASE_PATH = os.getenv('DATABASE_PATH', 'scanner.duckdb')

@dataclass
class User:
    id: str
    email: str
    hashed_password: str
    full_name: str
    is_active: bool = True
    created_at: str = None
    last_login: str = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

@dataclass
class ScanJob:
    id: str
    user_id: str
    status: str = "pending"
    project_ids: List[str] = None
    started_at: str = None
    completed_at: str = None
    error_message: str = None
    results_path: str = None
    total_tables: int = 0
    processed_tables: int = 0
    mappings_found: int = 0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()
        if not self.project_ids:
            self.project_ids = []

@dataclass
class ProjectAccess:
    id: str
    user_id: str
    project_id: str
    access_level: str = "read"
    granted_at: str = None
    granted_by: str = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.granted_at:
            self.granted_at = datetime.now(timezone.utc).isoformat()

@dataclass
class ScanResult:
    id: str
    job_id: str
    project_id: str
    dataset_id: str
    table_id: str
    column_name: str = None
    metric_type: str = None
    confidence_score: float = 0.0
    pattern_type: str = None
    detection_methods: List[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.detection_methods:
            self.detection_methods = []

@dataclass
class AuditLog:
    id: str
    user_id: str = None
    action: str = None
    resource_type: str = None
    resource_id: str = None
    details: Dict = None
    ip_address: str = None
    user_agent: str = None
    timestamp: str = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.details:
            self.details = {}

class DuckDBManager:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        self.conn = duckdb.connect(self.db_path)
        self._create_tables()
    
    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR PRIMARY KEY,
                email VARCHAR UNIQUE NOT NULL,
                hashed_password VARCHAR NOT NULL,
                full_name VARCHAR,
                is_active BOOLEAN DEFAULT true,
                created_at TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_jobs (
                id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                status VARCHAR DEFAULT 'pending',
                project_ids JSON,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                results_path VARCHAR,
                total_tables INTEGER DEFAULT 0,
                processed_tables INTEGER DEFAULT 0,
                mappings_found INTEGER DEFAULT 0
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS project_access (
                id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                project_id VARCHAR NOT NULL,
                access_level VARCHAR DEFAULT 'read',
                granted_at TIMESTAMP,
                granted_by VARCHAR
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_results (
                id VARCHAR PRIMARY KEY,
                job_id VARCHAR NOT NULL,
                project_id VARCHAR NOT NULL,
                dataset_id VARCHAR NOT NULL,
                table_id VARCHAR NOT NULL,
                column_name VARCHAR,
                metric_type VARCHAR,
                confidence_score DOUBLE,
                pattern_type VARCHAR,
                detection_methods JSON,
                created_at TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id VARCHAR PRIMARY KEY,
                user_id VARCHAR,
                action VARCHAR,
                resource_type VARCHAR,
                resource_id VARCHAR,
                details JSON,
                ip_address VARCHAR,
                user_agent VARCHAR,
                timestamp TIMESTAMP
            )
        """)
    
    def create_user(self, user: User) -> str:
        self.conn.execute("""
            INSERT INTO users (id, email, hashed_password, full_name, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [user.id, user.email, user.hashed_password, user.full_name, user.is_active, user.created_at])
        return user.id
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        result = self.conn.execute("SELECT * FROM users WHERE email = ?", [email]).fetchone()
        if result:
            return User(*result)
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        result = self.conn.execute("SELECT * FROM users WHERE id = ?", [user_id]).fetchone()
        if result:
            return User(*result)
        return None
    
    def create_scan_job(self, job: ScanJob) -> str:
        self.conn.execute("""
            INSERT INTO scan_jobs (id, user_id, status, project_ids, started_at, total_tables, processed_tables, mappings_found)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [job.id, job.user_id, job.status, json.dumps(job.project_ids), job.started_at, job.total_tables, job.processed_tables, job.mappings_found])
        return job.id
    
    def update_scan_job(self, job_id: str, updates: Dict[str, Any]):
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [job_id]
        self.conn.execute(f"UPDATE scan_jobs SET {set_clause} WHERE id = ?", values)
    
    def get_scan_job(self, job_id: str) -> Optional[ScanJob]:
        result = self.conn.execute("SELECT * FROM scan_jobs WHERE id = ?", [job_id]).fetchone()
        if result:
            data = list(result)
            if data[3]:
                data[3] = json.loads(data[3])
            return ScanJob(*data)
        return None
    
    def get_user_jobs(self, user_id: str, limit: int = 50) -> List[ScanJob]:
        results = self.conn.execute("""
            SELECT * FROM scan_jobs WHERE user_id = ? 
            ORDER BY started_at DESC LIMIT ?
        """, [user_id, limit]).fetchall()
        
        jobs = []
        for result in results:
            data = list(result)
            if data[3]:
                data[3] = json.loads(data[3])
            jobs.append(ScanJob(*data))
        return jobs
    
    def create_scan_result(self, result: ScanResult) -> str:
        self.conn.execute("""
            INSERT INTO scan_results (id, job_id, project_id, dataset_id, table_id, column_name, metric_type, confidence_score, pattern_type, detection_methods, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [result.id, result.job_id, result.project_id, result.dataset_id, result.table_id, result.column_name, result.metric_type, result.confidence_score, result.pattern_type, json.dumps(result.detection_methods), result.created_at])
        return result.id
    
    def get_job_results(self, job_id: str) -> List[ScanResult]:
        results = self.conn.execute("SELECT * FROM scan_results WHERE job_id = ?", [job_id]).fetchall()
        scan_results = []
        for result in results:
            data = list(result)
            if data[9]:
                data[9] = json.loads(data[9])
            scan_results.append(ScanResult(*data))
        return scan_results
    
    def grant_project_access(self, access: ProjectAccess) -> str:
        self.conn.execute("""
            INSERT INTO project_access (id, user_id, project_id, access_level, granted_at, granted_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [access.id, access.user_id, access.project_id, access.access_level, access.granted_at, access.granted_by])
        return access.id
    
    def check_project_access(self, user_id: str, project_id: str) -> bool:
        result = self.conn.execute("""
            SELECT COUNT(*) FROM project_access 
            WHERE user_id = ? AND project_id = ?
        """, [user_id, project_id]).fetchone()
        return result[0] > 0
    
    def get_user_projects(self, user_id: str) -> List[str]:
        results = self.conn.execute("""
            SELECT project_id FROM project_access WHERE user_id = ?
        """, [user_id]).fetchall()
        return [r[0] for r in results]
    
    def log_audit(self, log: AuditLog) -> str:
        self.conn.execute("""
            INSERT INTO audit_logs (id, user_id, action, resource_type, resource_id, details, ip_address, user_agent, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [log.id, log.user_id, log.action, log.resource_type, log.resource_id, json.dumps(log.details), log.ip_address, log.user_agent, log.timestamp])
        return log.id
    
    def close(self):
        if self.conn:
            self.conn.close()

db = DuckDBManager()
