import json
import time
from typing import Optional, Dict, Any

class InMemoryCache:
    def __init__(self):
        self.cache = {}
        self.expiry = {}
        
    def _is_expired(self, key: str) -> bool:
        if key not in self.expiry:
            return False
        return time.time() > self.expiry[key]
    
    def _cleanup_expired(self):
        expired_keys = [k for k in self.expiry if self._is_expired(k)]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.expiry.pop(key, None)
    
    def set_job_status(self, job_id: str, status: Dict[str, Any], ttl: int = 3600):
        key = f"job_status:{job_id}"
        self.cache[key] = json.dumps(status, default=str)
        self.expiry[key] = time.time() + ttl
        self._cleanup_expired()
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        key = f"job_status:{job_id}"
        if key in self.cache and not self._is_expired(key):
            return json.loads(self.cache[key])
        return None
    
    def set_user_session(self, session_id: str, user_data: Dict, ttl: int = 86400):
        key = f"session:{session_id}"
        self.cache[key] = json.dumps(user_data, default=str)
        self.expiry[key] = time.time() + ttl
        self._cleanup_expired()
    
    def get_user_session(self, session_id: str) -> Optional[Dict]:
        key = f"session:{session_id}"
        if key in self.cache and not self._is_expired(key):
            return json.loads(self.cache[key])
        return None
    
    def invalidate_session(self, session_id: str):
        key = f"session:{session_id}"
        self.cache.pop(key, None)
        self.expiry.pop(key, None)
    
    def rate_limit_check(self, user_id: str, action: str, limit: int = 10, window: int = 3600) -> bool:
        key = f"rate_limit:{user_id}:{action}"
        current_time = time.time()
        
        if key not in self.cache:
            self.cache[key] = [current_time]
            self.expiry[key] = current_time + window
            return True
        
        timestamps = json.loads(self.cache[key])
        timestamps = [t for t in timestamps if current_time - t < window]
        
        if len(timestamps) >= limit:
            return False
        
        timestamps.append(current_time)
        self.cache[key] = json.dumps(timestamps)
        self.expiry[key] = current_time + window
        return True

cache_client = InMemoryCache()
