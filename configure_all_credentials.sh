#!/bin/bash

echo "🔐 Configuring AO1 Scanner with Complete Credential Set"
echo "======================================================"

create_jwt_secret() {
    if command -v openssl &> /dev/null; then
        openssl rand -hex 32
    elif command -v python3 &> /dev/null; then
        python3 -c "import secrets; print(secrets.token_hex(32))"
    else
        echo "jwt_secret_$(date +%s)_$(whoami)" | tr -d '\n' | base64 | head -c 64
    fi
}

mkdir -p credentials

echo "Step 1: Creating BigQuery service account JSON..."

if [ ! -f "credentials/bigquery-service-account.json" ]; then
    echo "Please provide your BigQuery credentials:"
    echo ""
    
    read -p "Project ID: " PROJECT_ID
    read -p "Client Email: " CLIENT_EMAIL
    read -p "Client ID: " CLIENT_ID
    read -p "Private Key ID: " PRIVATE_KEY_ID
    echo "Private Key (paste the full key including -----BEGIN/END PRIVATE KEY-----): "
    read -r PRIVATE_KEY
    
    cat > credentials/bigquery-service-account.json << JSON
{
  "type": "service_account",
  "project_id": "$PROJECT_ID",
  "private_key_id": "$PRIVATE_KEY_ID",
  "private_key": "$PRIVATE_KEY",
  "client_email": "$CLIENT_EMAIL",
  "client_id": "$CLIENT_ID",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/$CLIENT_EMAIL"
}
JSON
    
    echo "✅ Created BigQuery service account JSON"
else
    echo "✅ BigQuery service account JSON already exists"
fi

echo ""
echo "Step 2: Configuring environment variables..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from template"
fi

JWT_SECRET=$(create_jwt_secret)

cat >> .env << ENV

# Enhanced AO1 Scanner Configuration
JWT_SECRET_KEY=$JWT_SECRET
GOOGLE_APPLICATION_CREDENTIALS=credentials/bigquery-service-account.json

# Chronicle SIEM Integration
CHRONICLE_API_KEY=${CHRONICLE_API_KEY:-}
CHRONICLE_SECRET_KEY=${CHRONICLE_SECRET_KEY:-}
CHRONICLE_FEED_ID=${CHRONICLE_FEED_ID:-}
CHRONICLE_ENDPOINT=${CHRONICLE_ENDPOINT:-https://backstory.googleapis.com}

# GraphQL Endpoint
GRAPHQL_ENDPOINT=${GRAPHQL_ENDPOINT:-}

# Advanced Features
ENABLE_SIEM_INTEGRATION=true
ENABLE_GRAPHQL_QUERIES=true
ENABLE_CHRONICLE_EXPORT=true
ENABLE_REAL_TIME_STREAMING=true

# Performance Tuning
MAX_CONCURRENT_SCANS=8
BIGQUERY_TIMEOUT_SECONDS=300
CHRONICLE_BATCH_SIZE=1000
RETRY_ATTEMPTS=3

ENV

echo "✅ Enhanced .env configuration created"

echo ""
echo "Step 3: Creating Chronicle integration credentials..."

if [ ! -z "$CHRONICLE_API_KEY" ]; then
    cat > credentials/chronicle-config.json << JSON
{
  "api_key": "$CHRONICLE_API_KEY",
  "secret_key": "$CHRONICLE_SECRET_KEY",
  "feed_id": "$CHRONICLE_FEED_ID",
  "endpoint": "$CHRONICLE_ENDPOINT",
  "batch_size": 1000,
  "timeout_seconds": 30
}
JSON
    echo "✅ Chronicle configuration created"
else
    echo "⚠️  Chronicle credentials not provided. Set them manually in .env if needed"
fi

echo ""
echo "Step 4: Creating enhanced Kubernetes secrets..."

if command -v base64 &> /dev/null; then
    if [ "$(uname)" = "Darwin" ]; then
        GCP_CREDS_B64=$(base64 credentials/bigquery-service-account.json)
        JWT_B64=$(echo -n "$JWT_SECRET" | base64)
        CHRONICLE_B64=$([ -f "credentials/chronicle-config.json" ] && base64 credentials/chronicle-config.json || echo "")
    else
        GCP_CREDS_B64=$(base64 -w 0 credentials/bigquery-service-account.json)
        JWT_B64=$(echo -n "$JWT_SECRET" | base64 -w 0)
        CHRONICLE_B64=$([ -f "credentials/chronicle-config.json" ] && base64 -w 0 credentials/chronicle-config.json || echo "")
    fi
    
    cat > k8s/secret-configured.yaml << YAML
apiVersion: v1
kind: Secret
metadata:
  name: ao1-scanner-secrets
  namespace: ao1-scanner
type: Opaque
data:
  jwt-secret: $JWT_B64
  gcp-credentials: $GCP_CREDS_B64
  chronicle-config: $CHRONICLE_B64
  chronicle-api-key: $(echo -n "${CHRONICLE_API_KEY:-}" | base64 ${BASE64_FLAGS:-})
  graphql-endpoint: $(echo -n "${GRAPHQL_ENDPOINT:-}" | base64 ${BASE64_FLAGS:-})
YAML
    
    echo "✅ Enhanced Kubernetes secrets configured"
fi

echo ""
echo "Step 5: Creating credential validation script..."

cat > validate_credentials.py << 'PYTHON'
#!/usr/bin/env python3

import json
import os
import sys
from google.cloud import bigquery
from google.auth import default

def validate_bigquery():
    try:
        credentials, project = default()
        client = bigquery.Client(credentials=credentials, project=project)
        
        # Test query
        query = "SELECT 1 as test"
        job = client.query(query)
        list(job.result())
        
        print("✅ BigQuery connection successful")
        print(f"   Project: {project}")
        return True
    except Exception as e:
        print(f"❌ BigQuery connection failed: {e}")
        return False

def validate_chronicle():
    if not os.getenv('CHRONICLE_API_KEY'):
        print("⚠️  Chronicle credentials not configured")
        return True
    
    try:
        import requests
        
        api_key = os.getenv('CHRONICLE_API_KEY')
        endpoint = os.getenv('CHRONICLE_ENDPOINT', 'https://backstory.googleapis.com')
        
        # Test Chronicle API connectivity
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.get(f"{endpoint}/v1/tools/siemsettings", headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("✅ Chronicle connection successful")
            return True
        else:
            print(f"❌ Chronicle connection failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Chronicle validation failed: {e}")
        return False

def validate_graphql():
    endpoint = os.getenv('GRAPHQL_ENDPOINT')
    if not endpoint:
        print("⚠️  GraphQL endpoint not configured")
        return True
    
    try:
        import requests
        
        # Simple introspection query
        query = {"query": "{ __schema { queryType { name } } }"}
        response = requests.post(endpoint, json=query, timeout=10)
        
        if response.status_code == 200:
            print("✅ GraphQL endpoint accessible")
            return True
        else:
            print(f"❌ GraphQL endpoint failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ GraphQL validation failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Validating AO1 Scanner Credentials")
    print("====================================")
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials/bigquery-service-account.json'
    
    results = []
    results.append(validate_bigquery())
    results.append(validate_chronicle())
    results.append(validate_graphql())
    
    if all(results):
        print("\n🎉 All credentials validated successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some credential validations failed")
        print("Check the errors above and update your configuration")
        sys.exit(1)
PYTHON

chmod +x validate_credentials.py

echo "✅ Credential validation script created"

echo ""
echo "🎯 Complete credential configuration finished!"
echo ""
echo "Next steps:"
echo "1. Update any missing values in .env file"
echo "2. Run: python3 validate_credentials.py"
echo "3. Deploy: ./deploy.sh docker"
echo ""
echo "Files created:"
echo "• credentials/bigquery-service-account.json"
echo "• credentials/chronicle-config.json (if Chronicle keys provided)"
echo "• k8s/secret-configured.yaml"
echo "• validate_credentials.py"
echo "• Enhanced .env file"
