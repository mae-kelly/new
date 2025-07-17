import asyncio
import pytest
from error_handling import ErrorHandler, retry_with_backoff, CircuitBreaker, classify_error, ErrorType

@pytest.mark.asyncio
async def test_retry_with_backoff():
    attempts = 0
    
    @retry_with_backoff(max_retries=2)
    async def failing_function():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception("Temporary failure")
        return "success"
    
    result = await failing_function()
    assert result == "success"
    assert attempts == 3

def test_circuit_breaker():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    
    def failing_function():
        raise Exception("Always fails")
    
    with pytest.raises(Exception):
        cb.call(failing_function)
    
    with pytest.raises(Exception):
        cb.call(failing_function)
    
    assert cb.state == "OPEN"
    
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        cb.call(failing_function)

def test_error_classification():
    assert classify_error(Exception("Permission denied")) == ErrorType.PERMISSION_ERROR
    assert classify_error(Exception("Quota exceeded")) == ErrorType.QUOTA_ERROR
    assert classify_error(Exception("Connection timeout")) == ErrorType.TIMEOUT_ERROR
    assert classify_error(Exception("Network error")) == ErrorType.NETWORK_ERROR
    assert classify_error(Exception("BigQuery error")) == ErrorType.BIGQUERY_ERROR
    assert classify_error(Exception("Unknown error")) == ErrorType.GENERAL_ERROR

if __name__ == "__main__":
    pytest.main([__file__])
