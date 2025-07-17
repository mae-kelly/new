import os
import secrets
from typing import Optional

class SecurityConfig:
    JWT_SECRET = os.getenv('JWT_SECRET_KEY', 'fallback-secret-for-dev')
    GOOGLE_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    ALLOWED_DOMAINS = [d.strip() for d in os.getenv('ALLOWED_DOMAINS', '').split(',') if d.strip()]
    MAX_PROJECTS_PER_SCAN = int(os.getenv('MAX_PROJECTS_PER_SCAN', '10'))
    MAX_TABLES_PER_PROJECT = int(os.getenv('MAX_TABLES_PER_PROJECT', '100'))
    SCAN_TIMEOUT_MINUTES = int(os.getenv('SCAN_TIMEOUT_MINUTES', '30'))
    
    @staticmethod
    def validate_config():
        if not SecurityConfig.JWT_SECRET or SecurityConfig.JWT_SECRET == 'fallback-secret-for-dev':
            if os.getenv('ENVIRONMENT') == 'production':
                raise ValueError("JWT_SECRET_KEY must be set in production")

class DataPrivacy:
    @staticmethod
    def anonymize_value(value):
        if not value or len(str(value)) < 3:
            return "<redacted>"
        return f"<redacted:{len(str(value))}chars>"
    
    @staticmethod
    def sanitize_column_sample(values):
        return [DataPrivacy.anonymize_value(v) for v in values[:3]]
    
    @staticmethod
    def safe_table_reference(project, dataset, table):
        return f"{project}.{dataset}.{table}"
