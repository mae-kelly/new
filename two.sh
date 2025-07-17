#!/bin/bash

cat > fix_subprocess_issues.sh << 'EOF'
#!/bin/bash

echo "🔧 Fixing Subprocess and Build Issues"
echo "====================================="

echo "Step 1: Creating Dockerfile with proper system dependencies..."
cat > Dockerfile << 'DOCKERFILE'
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install ALL system dependencies needed for subprocess builds
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    curl \
    wget \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade build tools to avoid subprocess errors
RUN pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    build

# Copy requirements and install with explicit build isolation
COPY requirements.txt .

# Install packages one by one to isolate subprocess issues
RUN pip install --no-cache-dir --no-build-isolation fastapi==0.104.1
RUN pip install --no-cache-dir --no-build-isolation uvicorn[standard]==0.24.0
RUN pip install --no-cache-dir --no-build-isolation google-cloud-bigquery==3.12.0
RUN pip install --no-cache-dir --no-build-isolation google-cloud-resource-manager==1.10.4
RUN pip install --no-cache-dir --no-build-isolation google-auth==2.23.4
RUN pip install --no-cache-dir --no-build-isolation PyJWT==2.8.0
RUN pip install --no-cache-dir --no-build-isolation bcrypt==4.1.1
RUN pip install --no-cache-dir --no-build-isolation duckdb==0.9.2
RUN pip install --no-cache-dir --no-build-isolation "pydantic[email]==2.5.0"
RUN pip install --no-cache-dir --no-build-isolation python-dotenv==1.0.0
RUN pip install --no-cache-dir --no-build-isolation structlog==23.2.0
RUN pip install --no-cache-dir --no-build-isolation httpx==0.25.2
RUN pip install --no-cache-dir --no-build-isolation "pandas==2.1.4"
RUN pip install --no-cache-dir --no-build-isolation "numpy==1.25.2"
RUN pip install --no-cache-dir --no-build-isolation redis==5.0.1

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data outputs logs credentials results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "run_server.py"]
DOCKERFILE

echo "Step 2: Creating minimal requirements.txt (for reference)..."
cat > requirements.txt << 'REQS'
# Core dependencies - installed individually in Dockerfile to avoid subprocess conflicts
fastapi==0.104.1
uvicorn[standard]==0.24.0
google-cloud-bigquery==3.12.0
google-cloud-resource-manager==1.10.4
google-auth==2.23.4
PyJWT==2.8.0
bcrypt==4.1.1
duckdb==0.9.2
pydantic[email]==2.5.0
python-dotenv==1.0.0
structlog==23.2.0
httpx==0.25.2
pandas==2.1.4
numpy==1.25.2
redis==5.0.1
REQS

echo "Step 3: Creating alternative Alpine Dockerfile (if Debian still fails)..."
cat > Dockerfile.alpine << 'DOCKERFILE'
FROM python:3.11-alpine

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Alpine system dependencies for subprocess builds
RUN apk add --no-cache \
    build-base \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    libffi-dev \
    openssl-dev \
    postgresql-dev \
    curl \
    git

# Install Python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core packages without subprocess issues
RUN pip install --no-cache-dir fastapi==0.104.1 uvicorn==0.24.0
RUN pip install --no-cache-dir google-cloud-bigquery==3.12.0 google-auth==2.23.4
RUN pip install --no-cache-dir PyJWT==2.8.0 python-dotenv==1.0.0
RUN pip install --no-cache-dir duckdb==0.9.2 structlog==23.2.0
RUN pip install --no-cache-dir httpx==0.25.2 redis==5.0.1

# Skip problematic packages or use alternatives
# RUN pip install --no-cache-dir bcrypt==4.1.1  # Can cause subprocess issues
# RUN pip install --no-cache-dir "pydantic[email]==2.5.0"  # Use basic pydantic instead
RUN pip install --no-cache-dir pydantic==2.5.0

# Copy and run
COPY . .
RUN mkdir -p data outputs logs credentials results

EXPOSE 8000
CMD ["python", "run_server.py"]
DOCKERFILE

echo "Step 4: Creating prebuilt wheel Dockerfile (fastest option)..."
cat > Dockerfile.prebuilt << 'DOCKERFILE'
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Use only prebuilt wheels (no compilation, no subprocess issues)
RUN pip install --no-cache-dir --only-binary=all \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    google-cloud-bigquery==3.12.0 \
    google-cloud-resource-manager==1.10.4 \
    google-auth==2.23.4 \
    PyJWT==2.8.0 \
    python-dotenv==1.0.0 \
    structlog==23.2.0 \
    httpx==0.25.2 \
    redis==5.0.1

# Skip packages that don't have prebuilt wheels or cause issues
# We'll handle these differently in the app if needed

COPY . .
RUN mkdir -p data outputs logs credentials results

EXPOSE 8000
CMD ["python", "run_server.py"]
DOCKERFILE

echo "Step 5: Creating docker-compose.yml with build options..."
cat > docker-compose.yml << 'YAML'
version: '3.8'

services:
  ao1-scanner:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "${PORT:-8000}:8000"
    environment:
      - DATABASE_PATH=/app/data/scanner.duckdb
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/bigquery-service-account.json
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./logs:/app/logs
      - ./credentials:/app/credentials
      - ./results:/app/results
    depends_on:
      redis:
        condition: service_healthy
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

echo "Step 6: Creating build troubleshooting script..."
cat > debug_subprocess.sh << 'SCRIPT'
#!/bin/bash

echo "🔍 Debugging Subprocess Build Issues"
echo "==================================="

echo "Testing different build approaches..."

echo ""
echo "Option 1: Full system dependencies (recommended)"
echo "Building with Dockerfile..."
docker build -t ao1-scanner:full . --no-cache --progress=plain 2>&1 | tee build-full.log

if [ $? -eq 0 ]; then
    echo "✅ Full build successful"
else
    echo "❌ Full build failed, trying Alpine..."
    
    echo ""
    echo "Option 2: Alpine Linux"
    echo "Building with Dockerfile.alpine..."
    docker build -f Dockerfile.alpine -t ao1-scanner:alpine . --no-cache --progress=plain 2>&1 | tee build-alpine.log
    
    if [ $? -eq 0 ]; then
        echo "✅ Alpine build successful"
    else
        echo "❌ Alpine build failed, trying prebuilt wheels..."
        
        echo ""
        echo "Option 3: Prebuilt wheels only"
        echo "Building with Dockerfile.prebuilt..."
        docker build -f Dockerfile.prebuilt -t ao1-scanner:prebuilt . --no-cache --progress=plain 2>&1 | tee build-prebuilt.log
        
        if [ $? -eq 0 ]; then
            echo "✅ Prebuilt wheels successful"
        else
            echo "❌ All builds failed"
            echo ""
            echo "Subprocess error analysis:"
            echo "Check these log files:"
            echo "• build-full.log"
            echo "• build-alpine.log" 
            echo "• build-prebuilt.log"
            echo ""
            echo "Common subprocess issues:"
            echo "• Missing build-essential"
            echo "• Insufficient memory"
            echo "• Architecture mismatch"
            echo "• Corrupted package cache"
        fi
    fi
fi

echo ""
echo "System info for debugging:"
echo "Docker version: $(docker --version)"
echo "Platform: $(uname -a)"
echo "Available space: $(df -h . | tail -1)"
echo "Memory: $(free -h 2>/dev/null || echo 'N/A')"
SCRIPT

chmod +x debug_subprocess.sh

echo "✅ Subprocess fix scripts created"
EOF

cat > quick_subprocess_fix.sh << 'EOF'
#!/bin/bash

echo "⚡ Quick Subprocess Fix and Deploy"
echo "================================="

# Apply fixes
./fix_subprocess_issues.sh

if [ ! -f ".env" ]; then
    echo "❌ Need .env file with BigQuery credentials"
    exit 1
fi

source .env

if [ "$BIGQUERY_PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Edit .env with real BigQuery credentials first"
    exit 1
fi

echo ""
echo "Choose build approach:"
echo "1. Full system deps (recommended)"
echo "2. Alpine Linux (lightweight)"  
echo "3. Prebuilt wheels only (fastest)"
echo "4. Debug all approaches"
echo ""
read -p "Enter choice (1-4): " choice

# Create credentials first
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

mkdir -p data outputs logs results

case $choice in
    1)
        echo "Building with full system dependencies..."
        docker-compose up --build -d
        ;;
    2)
        echo "Building with Alpine..."
        sed -i.bak 's/dockerfile: Dockerfile/dockerfile: Dockerfile.alpine/' docker-compose.yml
        docker-compose up --build -d
        ;;
    3)
        echo "Building with prebuilt wheels..."
        sed -i.bak 's/dockerfile: Dockerfile/dockerfile: Dockerfile.prebuilt/' docker-compose.yml
        docker-compose up --build -d
        ;;
    4)
        echo "Running debug build..."
        ./debug_subprocess.sh
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Waiting for services..."
sleep 15

if curl -f http://localhost:${PORT:-8000}/health >/dev/null 2>&1; then
    echo "✅ Success! AO1 Scanner running at http://localhost:${PORT:-8000}"
else
    echo "❌ Build may have failed. Check logs:"
    echo "docker-compose logs"
fi
EOF

chmod +x fix_subprocess_issues.sh quick_subprocess_fix.sh

echo "✅ Subprocess fix created!"
echo ""
echo "🔧 To fix subprocess build issues:"
echo ""
echo "Quick fix:"
echo "  ./quick_subprocess_fix.sh"
echo ""
echo "Debug approach:"
echo "  ./fix_subprocess_issues.sh"
echo "  ./debug_subprocess.sh"
echo ""
echo "This addresses:"
echo "• Missing build-essential packages"
echo "• Subprocess compilation errors"
echo "• Wheel building failures"
echo "• Architecture-specific issues"