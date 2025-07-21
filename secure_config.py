import os
import secrets
from typing import Optional, List

class SecurityConfig:
    JWT_SECRET = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
    GOOGLE_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    ALLOWED_DOMAINS = [d.strip() for d in os.getenv('ALLOWED_DOMAINS', '').split(',') if d.strip()]
    MAX_PROJECTS_PER_SCAN = int(os.getenv('MAX_PROJECTS_PER_SCAN', '50'))
    MAX_TABLES_PER_PROJECT = int(os.getenv('MAX_TABLES_PER_PROJECT', '200'))
    SCAN_TIMEOUT_MINUTES = int(os.getenv('SCAN_TIMEOUT_MINUTES', '60'))
    
    @staticmethod
    def validate_config():
        if not SecurityConfig.GOOGLE_CREDENTIALS_PATH:
            if os.getenv('ENVIRONMENT') == 'production':
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS must be set")

class DataPrivacy:
    @staticmethod
    def sanitize_evidence(evidence: List[str]) -> List[str]:
        return [f"<redacted:{len(str(e))}>" for e in evidence[:3]]
    
    @staticmethod
    def safe_table_reference(project: str, dataset: str, table: str) -> str:
        return f"{project}.{dataset}.{table}"

class PerformanceConfig:
    MAX_CONCURRENT_SCANS = 8
    QUERY_TIMEOUT_MS = 180000
    MAX_BYTES_BILLED = 50 * 1024 * 1024
    SAMPLE_RATE = 0.002
    MAX_ROWS_PER_TABLE = 1000
    
    RESILIENCE_SETTINGS = {
        'max_retries': 3,
        'backoff_factor': 2,
        'circuit_breaker_threshold': 5,
        'rate_limit_per_second': 10
    }
