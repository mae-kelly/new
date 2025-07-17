#!/bin/bash

cat > fix_build_issues.sh << 'EOF'
#!/bin/bash

echo "🔧 Fixing Docker build and Python dependency issues..."

echo "Step 1: Creating fixed requirements.txt..."
cat > requirements.txt << 'REQS'
# Core web framework
fastapi==0.104.1
uvicorn[standard]==0.24.0

# Google Cloud dependencies
google-cloud-bigquery==3.12.0
google-cloud-resource-manager==1.10.4
google-auth==2.23.4

# Authentication and security
PyJWT==2.8.0
bcrypt==4.1.1

# Database
duckdb==0.9.2

# Data validation
pydantic[email]==2.5.0

# Configuration
python-dotenv==1.0.0

# Logging
structlog==23.2.0

# HTTP client
httpx==0.25.2

# Data processing
pandas>=2.1.0,<2.3.0
numpy>=1.25.0,<2.0.0

# Additional dependencies for enhanced features
redis>=5.0.0
prometheus-client>=0.19.0
psutil>=5.9.0
requests>=2.31.0

# Development tools (optional)
pytest>=7.4.0
pytest-asyncio>=0.21.0
REQS

echo "✅ Fixed requirements.txt created"

echo "Step 2: Creating optimized Dockerfile..."
cat > Dockerfile.fixed << 'DOCKERFILE'
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with verbose output for debugging
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/outputs /app/logs /app/credentials /app/results

# Create non-root user
RUN groupadd -r scanner && useradd -r -g scanner scanner
RUN chown -R scanner:scanner /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Switch to non-root user
USER scanner

# Start application
CMD ["python", "run_server.py"]
DOCKERFILE

echo "✅ Optimized Dockerfile created"

echo "Step 3: Creating simplified Docker Compose..."
cat > docker-compose.simple.yml << 'YAML'
version: '3.8'

services:
  ao1-scanner:
    build:
      context: .
      dockerfile: Dockerfile.fixed
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
      retries: 5
      start_period: 60s
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

volumes:
  redis-data:
YAML

echo "✅ Simplified Docker Compose created"

echo "Step 4: Creating build troubleshooting script..."
cat > debug_build.sh << 'SCRIPT'
#!/bin/bash

echo "🔍 Docker Build Debug Information"
echo "================================"

echo "Docker version:"
docker --version

echo ""
echo "Python version in container:"
docker run --rm python:3.11-slim python --version

echo ""
echo "Available disk space:"
df -h

echo ""
echo "Docker system info:"
docker system df

echo ""
echo "Cleaning up old builds..."
docker system prune -f

echo ""
echo "Building with verbose output..."
docker build -f Dockerfile.fixed -t ao1-scanner:debug . --no-cache --progress=plain

echo ""
echo "Testing container startup..."
docker run --rm -d --name ao1-test ao1-scanner:debug

sleep 10

echo "Container logs:"
docker logs ao1-test

echo "Stopping test container..."
docker stop ao1-test

echo "✅ Debug build complete"
SCRIPT

chmod +x debug_build.sh

echo "✅ Build troubleshooting script created"

echo "Step 5: Creating alternative lightweight build..."
cat > Dockerfile.lightweight << 'DOCKERFILE'
FROM python:3.11-alpine

# Install system dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    curl \
    postgresql-dev

WORKDIR /app

# Install core dependencies only
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    google-cloud-bigquery==3.12.0 \
    google-auth==2.23.4 \
    pydantic[email]==2.5.0 \
    python-dotenv==1.0.0 \
    duckdb==0.9.2 \
    PyJWT==2.8.0 \
    bcrypt==4.1.1

# Copy application
COPY . .

# Create directories
RUN mkdir -p data outputs logs credentials results

EXPOSE 8000

CMD ["python", "run_server.py"]
DOCKERFILE

echo "✅ Lightweight Alpine Dockerfile created"

echo "Step 6: Creating no-build option..."
cat > run_without_docker.sh << 'SCRIPT'
#!/bin/bash

echo "🏃 Running AO1 Scanner without Docker"
echo "====================================="

if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Run ./create_env_template.sh first"
    exit 1
fi

echo "Step 1: Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Step 2: Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Step 3: Loading environment..."
source .env

echo "Step 4: Creating BigQuery credentials..."
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

export GOOGLE_APPLICATION_CREDENTIALS=credentials/bigquery-service-account.json

echo "Step 5: Testing BigQuery connection..."
python3 -c "
from google.cloud import bigquery
from google.auth import default
try:
    credentials, project = default()
    client = bigquery.Client(credentials=credentials, project=project)
    query = 'SELECT 1 as test'
    job = client.query(query)
    list(job.result())
    print('✅ BigQuery connection successful')
except Exception as e:
    print(f'❌ BigQuery connection failed: {e}')
    exit(1)
"

echo "Step 6: Starting AO1 Scanner..."
mkdir -p data outputs logs results

echo "Starting server on port ${PORT:-8000}..."
python run_server.py &

SERVER_PID=$!

sleep 5

echo "Step 7: Testing server..."
if curl -f http://localhost:${PORT:-8000}/health >/dev/null 2>&1; then
    echo "✅ AO1 Scanner is running!"
    echo ""
    echo "📊 Access Points:"
    echo "• API: http://localhost:${PORT:-8000}"
    echo "• Health: http://localhost:${PORT:-8000}/health"
    echo "• Dashboard: http://localhost:${PORT:-8000}/dashboard/stats"
    echo ""
    echo "Press Ctrl+C to stop the server"
    wait $SERVER_PID
else
    echo "❌ Server failed to start"
    kill $SERVER_PID
    exit 1
fi
SCRIPT

chmod +x run_without_docker.sh

echo "✅ No-Docker option created"
EOF

cat > quick_fix_and_run.sh << 'EOF'
#!/bin/bash

echo "🚀 Quick Fix and Run AO1 Scanner"
echo "================================"

# Apply fixes
./fix_build_issues.sh

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    ./create_env_template.sh
    echo ""
    echo "❌ Please edit .env with your BigQuery credentials first!"
    echo "Then run this script again."
    exit 1
fi

source .env

if [ "$BIGQUERY_PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Please edit .env with your actual BigQuery credentials"
    exit 1
fi

echo ""
echo "Choose deployment method:"
echo "1. Docker (recommended)"
echo "2. Docker with debug build"
echo "3. Lightweight Alpine Docker"
echo "4. No Docker (Python virtual env)"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Building with fixed Dockerfile..."
        docker build -f Dockerfile.fixed -t ao1-scanner:latest .
        docker-compose -f docker-compose.simple.yml up -d
        ;;
    2)
        echo "Running debug build..."
        ./debug_build.sh
        ;;
    3)
        echo "Building lightweight Alpine version..."
        docker build -f Dockerfile.lightweight -t ao1-scanner:lightweight .
        docker run -d -p ${PORT:-8000}:8000 --env-file .env \
            -v $(pwd)/credentials:/app/credentials \
            -v $(pwd)/data:/app/data \
            --name ao1-scanner ao1-scanner:lightweight
        ;;
    4)
        echo "Running without Docker..."
        ./run_without_docker.sh
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Waiting for service to start..."
sleep 10

echo "Testing endpoints..."
if curl -f http://localhost:${PORT:-8000}/health >/dev/null 2>&1; then
    echo "✅ AO1 Scanner is running!"
    echo "• Health: http://localhost:${PORT:-8000}/health"
    echo "• Dashboard: http://localhost:${PORT:-8000}/dashboard/stats"
else
    echo "❌ Service not responding. Check logs:"
    if [ "$choice" = "4" ]; then
        echo "Check terminal output above"
    else
        echo "docker logs ao1-scanner"
    fi
fi
EOF

chmod +x fix_build_issues.sh quick_fix_and_run.sh

echo "✅ Docker build fixes created!"
echo ""
echo "🛠️ To fix your build issues and run:"
echo ""
echo "Option 1 - Quick fix and choose deployment:"
echo "  ./quick_fix_and_run.sh"
echo ""
echo "Option 2 - Manual steps:"
echo "  1. ./fix_build_issues.sh"
echo "  2. docker build -f Dockerfile.fixed -t ao1-scanner:latest ."
echo "  3. docker-compose -f docker-compose.simple.yml up -d"
echo ""
echo "Option 3 - Skip Docker entirely:"
echo "  ./run_without_docker.sh"
echo ""
echo "This fixes:"
echo "• Python package dependency conflicts"
echo "• License classifier warnings"
echo "• Subprocess build errors"
echo "• Docker layer caching issues"