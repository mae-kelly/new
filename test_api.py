import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import tempfile
import uuid

@pytest.fixture(scope="function")
def test_db():
    test_db_path = f"/tmp/test_scanner_{uuid.uuid4().hex}.duckdb"
    
    if os.path.exists(test_db_path):
        os.unlink(test_db_path)
    
    with patch.dict(os.environ, {
        'DATABASE_PATH': test_db_path,
        'JWT_SECRET_KEY': 'test-secret-key-for-testing-only-very-long-secret',
        'ALLOWED_DOMAINS': 'test.com',
        'GOOGLE_APPLICATION_CREDENTIALS': 'test-creds.json'
    }):
        import importlib
        if 'database' in locals():
            importlib.reload(database)
        from database import DuckDBManager
        
        db = DuckDBManager(test_db_path)
        yield db
        
        try:
            db.close()
        except:
            pass
    
    if os.path.exists(test_db_path):
        try:
            os.unlink(test_db_path)
        except:
            pass

@pytest.fixture(scope="function")
def client(test_db):
    with patch.dict(os.environ, {
        'JWT_SECRET_KEY': 'test-secret-key-for-testing-only-very-long-secret',
        'ALLOWED_DOMAINS': 'test.com',
        'GOOGLE_APPLICATION_CREDENTIALS': 'test-creds.json'
    }):
        with patch('api.db', test_db):
            import importlib
            if 'api' in locals():
                importlib.reload(api)
            from api import app
            
            test_client = TestClient(app)
            yield test_client

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_register_invalid_domain(client):
    response = client.post("/auth/register", json={
        "email": "test@invalid.com",
        "password": "password123",
        "full_name": "Test User"
    })
    assert response.status_code == 400
    assert "domain not allowed" in response.json()["detail"]

def test_register_valid_domain(client):
    response = client.post("/auth/register", json={
        "email": "test@test.com",
        "password": "password123",
        "full_name": "Test User"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_protected_endpoint_no_auth():
    with patch.dict(os.environ, {
        'JWT_SECRET_KEY': 'test-secret-key-for-testing-only-very-long-secret',
        'ALLOWED_DOMAINS': 'test.com',
        'GOOGLE_APPLICATION_CREDENTIALS': 'test-creds.json'
    }):
        from api import app
        client = TestClient(app)
        response = client.get("/auth/me")
        assert response.status_code == 401
