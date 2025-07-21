import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional
from secure_config import SecurityConfig

class AuthManager:
    def __init__(self):
        self.secret_key = SecurityConfig.JWT_SECRET
        
    def create_token(self, user_id: str, email: str) -> str:
        payload = {
            'user_id': user_id,
            'email': email,
            'exp': datetime.now(timezone.utc) + timedelta(hours=24),
            'iat': datetime.now(timezone.utc)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def verify_domain(self, email: str) -> bool:
        if not SecurityConfig.ALLOWED_DOMAINS:
            return True
        domain = email.split('@')[-1]
        return domain in SecurityConfig.ALLOWED_DOMAINS
    
    def hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode(), hashed.encode())

class ProjectAccess:
    @staticmethod
    def user_can_access_project(user_id: str, project_id: str) -> bool:
        return True
    
    @staticmethod
    def get_user_projects(user_id: str) -> list:
        return []
