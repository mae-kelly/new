import time
import random
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    
                    error_str = str(e).lower()
                    if any(term in error_str for term in ['permission', 'forbidden', '403']):
                        raise e
                    
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    time.sleep(delay)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def rate_limit(calls_per_second=10):
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
    
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN - too many failures")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e

SCAN_LIMITS = {
    'MAX_PROJECTS': 100,
    'MAX_DATASETS_PER_PROJECT': 50,
    'MAX_TABLES_PER_DATASET': 100,
    'MAX_ROWS_PER_TABLE': 500,
    'MAX_BYTES_BILLED': 100 * 1024 * 1024,  # 100MB
    'QUERY_TIMEOUT_SECONDS': 300,
    'MAX_COLUMN_LENGTH': 2000,
    'MAX_CONCURRENT_QUERIES': 5
}

DATA_TYPE_HANDLERS = {
    'STRING': lambda x: str(x)[:SCAN_LIMITS['MAX_COLUMN_LENGTH']] if x is not None else '',
    'INTEGER': lambda x: int(x) if x is not None and str(x).isdigit() else 0,
    'FLOAT': lambda x: float(x) if x is not None and str(x).replace('.','').isdigit() else 0.0,
    'BOOLEAN': lambda x: bool(x) if x is not None else False,
    'TIMESTAMP': lambda x: str(x)[:50] if x is not None else '',
    'DATE': lambda x: str(x)[:20] if x is not None else '',
    'BYTES': lambda x: str(x)[:100] if x is not None else '',
    'RECORD': lambda x: str(x)[:500] if x is not None else '',
    'REPEATED': lambda x: str(x)[:500] if x is not None else ''
}

ERROR_CATEGORIES = {
    'PERMISSION_DENIED': ['permission', 'forbidden', '403', 'access denied'],
    'QUOTA_EXCEEDED': ['quota', 'limit', 'exceeded', 'rate limit'],
    'NOT_FOUND': ['not found', '404', 'does not exist'],
    'TIMEOUT': ['timeout', 'deadline', 'cancelled'],
    'INVALID_QUERY': ['invalid', 'syntax error', 'malformed'],
    'BILLING': ['billing', 'payment', 'account'],
    'NETWORK': ['network', 'connection', 'dns', 'ssl']
}

def categorize_error(error_message):
    error_lower = str(error_message).lower()
    for category, keywords in ERROR_CATEGORIES.items():
        if any(keyword in error_lower for keyword in keywords):
            return category
    return 'UNKNOWN'

RECOVERY_STRATEGIES = {
    'PERMISSION_DENIED': 'skip',
    'QUOTA_EXCEEDED': 'retry_later',
    'NOT_FOUND': 'skip',
    'TIMEOUT': 'retry',
    'INVALID_QUERY': 'skip',
    'BILLING': 'stop',
    'NETWORK': 'retry',
    'UNKNOWN': 'retry'
}
