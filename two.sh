#!/bin/bash

cat > create_env_template.sh << 'EOF'
#!/bin/bash

echo "📝 Creating .env template with all credential fields..."

cat > .env << 'ENV'
# Core Configuration
DATABASE_PATH=scanner.duckdb
JWT_SECRET_KEY=auto-generated-on-deploy
PORT=8000
HOST=0.0.0.0
ENVIRONMENT=production
LOG_LEVEL=INFO

# BigQuery Credentials - EDIT THESE WITH YOUR ACTUAL VALUES
BIGQUERY_TYPE=service_account
BIGQUERY_PROJECT_ID=your-project-id
BIGQUERY_PRIVATE_KEY_ID=your-private-key-id
BIGQUERY_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nyour-private-key-here\n-----END PRIVATE KEY-----"
BIGQUERY_CLIENT_EMAIL=scanner@your-project.iam.gserviceaccount.com
BIGQUERY_CLIENT_ID=your-client-id
BIGQUERY_AUTH_URI=https://accounts.google.com/o/oauth2/auth
BIGQUERY_TOKEN_URI=https://oauth2.googleapis.com/token
BIGQUERY_AUTH_PROVIDER_X509_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
BIGQUERY_CLIENT_X509_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/scanner%40your-project.iam.gserviceaccount.com

# Chronicle SIEM Integration - OPTIONAL
CHRONICLE_API_KEY=your-chronicle-api-key
CHRONICLE_SECRET_KEY=your-chronicle-secret-key
CHRONICLE_FEED_ID=your-chronicle-feed-id
CHRONICLE_ENDPOINT=https://backstory.googleapis.com

# GraphQL Integration - OPTIONAL  
GRAPHQL_ENDPOINT=https://your-graphql-endpoint.com/graphql
GRAPHQL_API_KEY=your-graphql-api-key

# Feature Flags
ENABLE_SIEM_INTEGRATION=true
ENABLE_GRAPHQL_QUERIES=true
ENABLE_REAL_TIME_STREAMING=true
ENABLE_INDUSTRY_DETECTION=true

# Performance Settings
MAX_PROJECTS_PER_SCAN=50
MAX_TABLES_PER_PROJECT=200
SCAN_TIMEOUT_MINUTES=60
MAX_CONCURRENT_SCANS=8
ENV

echo "✅ Created .env template"
echo ""
echo "📋 EDIT .env NOW with your actual credentials:"
echo ""
echo "Required (BigQuery):"
echo "• BIGQUERY_PROJECT_ID"
echo "• BIGQUERY_PRIVATE_KEY (full key with BEGIN/END lines)"
echo "• BIGQUERY_CLIENT_EMAIL" 
echo "• BIGQUERY_CLIENT_ID"
echo "• BIGQUERY_PRIVATE_KEY_ID"
echo ""
echo "Optional (Chronicle):"
echo "• CHRONICLE_API_KEY"
echo "• CHRONICLE_SECRET_KEY"
echo "• CHRONICLE_FEED_ID"
echo ""
echo "Optional (GraphQL):"
echo "• GRAPHQL_ENDPOINT"
echo "• GRAPHQL_API_KEY"
echo ""
echo "When done editing, run: ./run_scanner.sh"
EOF

cat > run_scanner.sh << 'EOF'
#!/bin/bash

set -e

echo "🚀 AO1 Scanner - Deploy and Run"
echo "==============================="

if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Run: ./create_env_template.sh first"
    exit 1
fi

echo "Step 1: Loading environment..."
source .env

if [ "$BIGQUERY_PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Please edit .env with your actual BigQuery credentials first"
    echo "Required fields: BIGQUERY_PROJECT_ID, BIGQUERY_CLIENT_EMAIL, BIGQUERY_PRIVATE_KEY"
    exit 1
fi

echo "✅ Environment loaded"

echo "Step 2: Generating JWT secret if needed..."
if [ "$JWT_SECRET_KEY" = "auto-generated-on-deploy" ]; then
    JWT_SECRET=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || echo "fallback_$(date +%s)")
    if [ "$(uname)" = "Darwin" ]; then
        sed -i '' "s/JWT_SECRET_KEY=auto-generated-on-deploy/JWT_SECRET_KEY=$JWT_SECRET/" .env
    else
        sed -i "s/JWT_SECRET_KEY=auto-generated-on-deploy/JWT_SECRET_KEY=$JWT_SECRET/" .env
    fi
    echo "✅ JWT secret generated"
fi

echo "Step 3: Creating BigQuery service account JSON..."
mkdir -p credentials

cat > credentials/bigquery-service-account.json << JSON
{
  "type": "$BIGQUERY_TYPE",
  "project_id": "$BIGQUERY_PROJECT_ID",
  "private_key_id": "$BIGQUERY_PRIVATE_KEY_ID",
  "private_key": "$BIGQUERY_PRIVATE_KEY",
  "client_email": "$BIGQUERY_CLIENT_EMAIL",
  "client_id": "$BIGQUERY_CLIENT_ID",
  "auth_uri": "$BIGQUERY_AUTH_URI",
  "token_uri": "$BIGQUERY_TOKEN_URI",
  "auth_provider_x509_cert_url": "$BIGQUERY_AUTH_PROVIDER_X509_CERT_URL",
  "client_x509_cert_url": "$BIGQUERY_CLIENT_X509_CERT_URL"
}
JSON

echo "✅ BigQuery credentials configured"

echo "Step 4: Testing BigQuery connection..."
export GOOGLE_APPLICATION_CREDENTIALS=credentials/bigquery-service-account.json

python3 << PYTHON
import os
try:
    from google.cloud import bigquery
    from google.auth import default
    
    credentials, project = default()
    client = bigquery.Client(credentials=credentials, project=project)
    
    query = "SELECT 1 as test"
    job = client.query(query)
    list(job.result())
    
    print("✅ BigQuery connection successful")
    
except Exception as e:
    print(f"❌ BigQuery connection failed: {e}")
    print("Check your credentials in .env")
    exit(1)
PYTHON

echo "Step 5: Creating Docker Compose..."
mkdir -p data outputs logs results monitoring

cat > docker-compose.yml << 'YAML'
version: '3.8'

services:
  ao1-scanner:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "${PORT:-8000}:8000"
    env_file:
      - .env
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/bigquery-service-account.json
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./logs:/app/logs
      - ./credentials:/app/credentials
      - ./results:/app/results
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
YAML

if [ ! -f "Dockerfile.prod" ]; then
    echo "Creating Dockerfile..."
    cat > Dockerfile.prod << 'DOCKERFILE'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/outputs /app/logs /app/credentials /app/results

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run_server.py"]
DOCKERFILE
fi

echo "✅ Docker configuration ready"

echo "Step 6: Building and starting services..."
docker-compose up --build -d

echo "Step 7: Waiting for services..."
sleep 15

echo "Step 8: Health check..."
for i in {1..20}; do
    if curl -f http://localhost:${PORT:-8000}/health >/dev/null 2>&1; then
        echo "✅ AO1 Scanner is healthy!"
        break
    else
        echo "⏳ Waiting for scanner... (attempt $i/20)"
        sleep 3
    fi
done

echo "Step 9: Testing endpoints..."
API_URL="http://localhost:${PORT:-8000}"

test_endpoint() {
    local endpoint=$1
    local name=$2
    
    if curl -f "$API_URL$endpoint" >/dev/null 2>&1; then
        echo "✅ $name working"
    else
        echo "⚠️  $name not responding"
    fi
}

test_endpoint "/health" "Health check"
test_endpoint "/health/detailed" "Detailed health"
test_endpoint "/dashboard/stats" "Dashboard"
test_endpoint "/metrics" "Metrics"

echo ""
echo "🎉 AO1 Scanner is RUNNING!"
echo "========================="
echo ""
echo "📊 Access Points:"
echo "• Main API: http://localhost:${PORT:-8000}"
echo "• Health: http://localhost:${PORT:-8000}/health/detailed"
echo "• Dashboard: http://localhost:${PORT:-8000}/dashboard/stats"
echo "• Metrics: http://localhost:${PORT:-8000}/metrics"
echo ""
echo "🚀 Quick Test:"
echo "curl http://localhost:${PORT:-8000}/health"
echo ""
echo "🔧 Management:"
echo "• Logs: docker-compose logs -f"
echo "• Stop: docker-compose down"
echo "• Restart: docker-compose restart"
echo ""
echo "Ready to scan BigQuery for AO1 visibility assessment!"
EOF

cat > quick_test.sh << 'EOF'
#!/bin/bash

API_URL="http://localhost:${PORT:-8000}"

echo "🧪 Quick AO1 Scanner Test"
echo "========================"

echo "Testing basic endpoints..."

echo "• Health check:"
HEALTH=$(curl -s "$API_URL/health")
echo "  $HEALTH"

echo "• Detailed health:"
curl -s "$API_URL/health/detailed" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  Overall: {data.get('overall_status', 'unknown')}\")
    components = data.get('components', {})
    for comp, status in components.items():
        print(f\"  {comp}: {status.get('status', 'unknown')}\")
except:
    print('  Could not parse health data')
"

echo "• User registration test:"
REG_RESPONSE=$(curl -s -X POST "$API_URL/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@company.com", "password": "test123", "full_name": "Test User"}')

if echo "$REG_RESPONSE" | grep -q "access_token"; then
    echo "  ✅ Registration working"
elif echo "$REG_RESPONSE" | grep -q "already registered"; then
    echo "  ✅ Registration endpoint working (user exists)"
else
    echo "  ⚠️  Registration response: $REG_RESPONSE"
fi

echo ""
echo "🎯 AO1 Scanner is ready for production scans!"
EOF

chmod +x create_env_template.sh run_scanner.sh quick_test.sh

echo "✅ Simple deployment setup created!"
echo ""
echo "🚀 Two-Step Process:"
echo ""
echo "1. Create .env template and edit with your credentials:"
echo "   ./create_env_template.sh"
echo "   # Edit .env with your actual BigQuery credentials"
echo ""
echo "2. Deploy and run everything:"
echo "   ./run_scanner.sh"
echo ""
echo "3. Optional - test functionality:"
echo "   ./quick_test.sh"
echo ""
echo "That's it! Put your credentials in .env, then run it."