import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import json
from contextlib import asynccontextmanager
import signal
import sys

class ErrorType(Enum):
    BIGQUERY_ERROR = "bigquery_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    PERMISSION_ERROR = "permission_error"
    QUOTA_ERROR = "quota_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    GENERAL_ERROR = "general_error"

@dataclass
class ErrorContext:
    error_type: ErrorType
    message: str
    retry_count: int
    max_retries: int
    backoff_seconds: float
    context: Dict[str, Any]

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        
    def call(self, func):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func()
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

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    
                    error_type = classify_error(e)
                    if error_type in [ErrorType.PERMISSION_ERROR, ErrorType.QUOTA_ERROR]:
                        raise e
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    await asyncio.sleep(delay)
                    
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def classify_error(error: Exception) -> ErrorType:
    error_str = str(error).lower()
    
    if "permission" in error_str or "forbidden" in error_str:
        return ErrorType.PERMISSION_ERROR
    elif "quota" in error_str or "rate limit" in error_str:
        return ErrorType.QUOTA_ERROR
    elif "timeout" in error_str:
        return ErrorType.TIMEOUT_ERROR
    elif "network" in error_str or "connection" in error_str:
        return ErrorType.NETWORK_ERROR
    elif "bigquery" in error_str:
        return ErrorType.BIGQUERY_ERROR
    else:
        return ErrorType.GENERAL_ERROR

class ErrorHandler:
    def __init__(self):
        self.circuit_breakers = {}
        
    def get_circuit_breaker(self, key: str) -> CircuitBreaker:
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = CircuitBreaker()
        return self.circuit_breakers[key]
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]):
        error_type = classify_error(error)
        
        error_data = {
            "error_type": error_type.value,
            "message": str(error),
            "context": context,
            "timestamp": time.time()
        }
        
        logging.error(f"Error handled: {error_type.value}", extra=error_data)
        
        if error_type == ErrorType.QUOTA_ERROR:
            await self.handle_quota_error(error, context)
        elif error_type == ErrorType.PERMISSION_ERROR:
            await self.handle_permission_error(error, context)
        elif error_type == ErrorType.RATE_LIMIT_ERROR:
            await self.handle_rate_limit_error(error, context)

    async def handle_quota_error(self, error: Exception, context: Dict[str, Any]):
        await asyncio.sleep(300)
        
    async def handle_permission_error(self, error: Exception, context: Dict[str, Any]):
        raise error
        
    async def handle_rate_limit_error(self, error: Exception, context: Dict[str, Any]):
        await asyncio.sleep(60)

class GracefulShutdown:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.tasks = []
        
    def register_task(self, task):
        self.tasks.append(task)
        
    async def shutdown(self):
        logging.info("Graceful shutdown initiated")
        self.shutdown_event.set()
        
        for task in self.tasks:
            if not task.done():
                task.cancel()
                
        await asyncio.gather(*self.tasks, return_exceptions=True)
        logging.info("Graceful shutdown complete")

shutdown_handler = GracefulShutdown()

def signal_handler(signum, frame):
    logging.info(f"Received signal {signum}")
    asyncio.create_task(shutdown_handler.shutdown())

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@asynccontextmanager
async def timeout_context(seconds: int):
    try:
        await asyncio.wait_for(asyncio.sleep(0), timeout=seconds)
        yield
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

async def health_check() -> bool:
    try:
        return True
    except Exception:
        return False

async def ready_check() -> bool:
    try:
        return True
    except Exception:
        return False
