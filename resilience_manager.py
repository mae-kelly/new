import asyncio
import time
import random
import logging
from typing import Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    timeout: int = 60
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0
    success_count: int = 0

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

class ScanResilienceManager:
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failure_patterns: Dict[str, Dict] = {}
        self.retry_configs: Dict[str, RetryConfig] = {
            'quota_exceeded': RetryConfig(max_retries=3, base_delay=30, max_delay=300),
            'permission_denied': RetryConfig(max_retries=1, base_delay=0.1, max_delay=0.1),
            'network_timeout': RetryConfig(max_retries=5, base_delay=2, max_delay=60),
            'internal_error': RetryConfig(max_retries=3, base_delay=5, max_delay=120),
            'default': RetryConfig()
        }

    def get_circuit_breaker(self, resource_id: str) -> CircuitBreaker:
        if resource_id not in self.circuit_breakers:
            self.circuit_breakers[resource_id] = CircuitBreaker()
        return self.circuit_breakers[resource_id]

    async def execute_with_resilience(self, operation: Callable, resource_id: str, operation_type: str = 'default'):
        circuit_breaker = self.get_circuit_breaker(resource_id)
        retry_config = self.retry_configs.get(operation_type, self.retry_configs['default'])
        
        if circuit_breaker.state == CircuitState.OPEN:
            if time.time() - circuit_breaker.last_failure_time > circuit_breaker.timeout:
                circuit_breaker.state = CircuitState.HALF_OPEN
                circuit_breaker.success_count = 0
            else:
                raise Exception(f"Circuit breaker OPEN for {resource_id}")

        last_exception = None
        
        for attempt in range(retry_config.max_retries + 1):
            try:
                result = await operation()
                self._record_success(resource_id)
                return result
                
            except Exception as e:
                last_exception = e
                error_type = self._classify_error(e)
                self._record_failure(resource_id, e, error_type)
                
                if attempt == retry_config.max_retries:
                    break
                    
                if error_type == 'permission_denied':
                    break
                    
                await self._adaptive_backoff(retry_config, attempt, error_type)
        
        raise last_exception

    def _classify_error(self, error: Exception) -> str:
        error_msg = str(error).lower()
        
        if any(keyword in error_msg for keyword in ['quota', 'limit', 'rate limit', '429']):
            return 'quota_exceeded'
        elif any(keyword in error_msg for keyword in ['permission', 'forbidden', '403', 'access denied']):
            return 'permission_denied'
        elif any(keyword in error_msg for keyword in ['timeout', 'connection', 'network']):
            return 'network_timeout'
        elif any(keyword in error_msg for keyword in ['500', 'internal', 'server error']):
            return 'internal_error'
        else:
            return 'unknown_error'

    def _record_success(self, resource_id: str):
        circuit_breaker = self.get_circuit_breaker(resource_id)
        
        if circuit_breaker.state == CircuitState.HALF_OPEN:
            circuit_breaker.success_count += 1
            if circuit_breaker.success_count >= 3:
                circuit_breaker.state = CircuitState.CLOSED
                circuit_breaker.failure_count = 0
        elif circuit_breaker.state == CircuitState.CLOSED:
            circuit_breaker.failure_count = max(0, circuit_breaker.failure_count - 1)

    def _record_failure(self, resource_id: str, error: Exception, error_type: str):
        circuit_breaker = self.get_circuit_breaker(resource_id)
        circuit_breaker.failure_count += 1
        circuit_breaker.last_failure_time = time.time()
        
        if circuit_breaker.failure_count >= circuit_breaker.failure_threshold:
            circuit_breaker.state = CircuitState.OPEN
        
        if resource_id not in self.failure_patterns:
            self.failure_patterns[resource_id] = {}
        
        if error_type not in self.failure_patterns[resource_id]:
            self.failure_patterns[resource_id][error_type] = 0
        self.failure_patterns[resource_id][error_type] += 1

    async def _adaptive_backoff(self, retry_config: RetryConfig, attempt: int, error_type: str):
        delay = min(
            retry_config.base_delay * (retry_config.exponential_base ** attempt),
            retry_config.max_delay
        )
        
        if retry_config.jitter:
            jitter = random.uniform(0, delay * 0.1)
            delay += jitter
        
        if error_type == 'quota_exceeded':
            delay *= 2
        elif error_type == 'network_timeout':
            delay *= 1.5
        
        await asyncio.sleep(delay)

    def get_resilience_stats(self) -> Dict[str, Any]:
        stats = {
            'circuit_breakers': {},
            'failure_patterns': self.failure_patterns,
            'total_resources': len(self.circuit_breakers)
        }
        
        for resource_id, cb in self.circuit_breakers.items():
            stats['circuit_breakers'][resource_id] = {
                'state': cb.state.value,
                'failure_count': cb.failure_count,
                'last_failure': cb.last_failure_time
            }
        
        return stats

resilience_manager = ScanResilienceManager()
