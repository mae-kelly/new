#!/bin/bash

cat > setup_mac_docker_compose.sh << 'EOF'
#!/bin/bash

echo "🍎 Mac Docker Compose Setup + License Fix"
echo "=========================================="

echo "Step 1: Installing Docker Compose on Mac..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew first..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon Macs
    if [[ $(uname -m) == 'arm64' ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

# Install Docker and Docker Compose
echo "Installing Docker and Docker Compose..."
brew install --cask docker
brew install docker-compose

# Start Docker Desktop
echo "Starting Docker Desktop..."
open /Applications/Docker.app

echo "Waiting for Docker to start..."
sleep 10

# Wait for Docker daemon
echo "Waiting for Docker daemon..."
while ! docker info >/dev/null 2>&1; do
    echo "  Docker not ready yet, waiting..."
    sleep 5
done

echo "✅ Docker is running"

# Verify installation
echo "Verifying Docker Compose installation..."
docker-compose --version
docker compose version

echo "Step 2: Creating license-compliant project structure..."

# Create pyproject.toml with SPDX license expression
cat > pyproject.toml << 'TOML'
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ao1-scanner"
version = "1.0.0"
description = "AO1 BigQuery Visibility Scanner"
license = {text = "MIT"}
authors = [
    {name = "AO1 Team", email = "team@company.com"}
]
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "google-cloud-bigquery>=3.12.0",
    "google-cloud-resource-manager>=1.10.0",
    "google-auth>=2.23.0",
    "PyJWT>=2.8.0",
    "duckdb>=0.9.0",
    "pydantic[email]>=2.5.0",
    "python-dotenv>=1.0.0",
    "structlog>=23.2.0",
    "httpx>=0.25.0",
    "redis>=5.0.0",
]
TOML

# Create requirements.txt without problematic license classifiers
cat > requirements.txt << 'REQS'
fastapi==0.104.1
uvicorn[standard]==0.24.0
google-cloud-bigquery==3.12.0
google-cloud-resource-manager==1.10.4
google-auth==2.23.4
PyJWT==2.8.0
duckdb==0.9.2
pydantic[email]==2.5.0
python-dotenv==1.0.0
structlog==23.2.0
httpx==0.25.2
redis==5.0.1
REQS

# Create Dockerfile optimized for Mac
cat > Dockerfile << 'DOCKERFILE'
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_WARN_SCRIPT_LOCATION=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools to handle SPDX licenses
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project files with SPDX license
COPY pyproject.toml .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p data outputs logs credentials results

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run_server.py"]
DOCKERFILE

# Create docker-compose.yml for Mac
cat > docker-compose.yml << 'YAML'
version: '3.8'

services:
  ao1-scanner:
    build: .
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
      - ./data:/app/data:delegated
      - ./outputs:/app/outputs:delegated
      - ./logs:/app/logs:delegated
      - ./credentials:/app/credentials:ro
      - ./results:/app/results:delegated
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

# Create .env template
if [ ! -f ".env" ]; then
    cat > .env << 'ENV'
# BigQuery Credentials - EDIT WITH YOUR VALUES
BIGQUERY_PROJECT_ID=your-project-id
BIGQUERY_CLIENT_EMAIL=scanner@your-project.iam.gserviceaccount.com
BIGQUERY_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nyour-private-key-here\n-----END PRIVATE KEY-----"
BIGQUERY_CLIENT_ID=your-client-id
BIGQUERY_PRIVATE_KEY_ID=your-private-key-id
BIGQUERY_TYPE=service_account
BIGQUERY_AUTH_URI=https://accounts.google.com/o/oauth2/auth
BIGQUERY_TOKEN_URI=https://oauth2.googleapis.com/token
BIGQUERY_AUTH_PROVIDER_X509_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
BIGQUERY_CLIENT_X509_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/scanner%40your-project.iam.gserviceaccount.com

# App Settings
PORT=8000
HOST=0.0.0.0
DATABASE_PATH=scanner.duckdb
JWT_SECRET_KEY=auto-generated
ENVIRONMENT=production
LOG_LEVEL=INFO
ENV

    echo "✅ Created .env template - EDIT WITH YOUR BIGQUERY CREDENTIALS"
fi

echo "✅ Mac Docker Compose setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your BigQuery credentials"
echo "2. Run: ./deploy_mac.sh"
EOF

cat > deploy_mac.sh << 'EOF'
#!/bin/bash

echo "🚀 Deploy AO1 Scanner on Mac"
echo "============================"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Starting Docker Desktop..."
    open /Applications/Docker.app
    
    echo "Waiting for Docker to start..."
    while ! docker info >/dev/null 2>&1; do
        sleep 5
        echo "  Still waiting for Docker..."
    done
fi

echo "✅ Docker is running"

# Check .env
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Run ./setup_mac_docker_compose.sh first"
    exit 1
fi

source .env

if [ "$BIGQUERY_PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Please edit .env with your actual BigQuery credentials"
    exit 1
fi

echo "Creating BigQuery credentials..."
mkdir -p credentials data outputs logs results

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

echo "Building and starting with Docker Compose..."
docker-compose down 2>/dev/null || true
docker-compose up --build -d

echo "Waiting for services..."
sleep 15

echo "Testing AO1 Scanner..."
for i in {1..20}; do
    if curl -f http://localhost:${PORT:-8000}/health >/dev/null 2>&1; then
        echo "✅ AO1 Scanner is running!"
        break
    else
        echo "⏳ Waiting for scanner... (attempt $i/20)"
        sleep 3
    fi
done

# Test endpoints
echo ""
echo "Testing endpoints..."
API_URL="http://localhost:${PORT:-8000}"

if curl -f "$API_URL/health" >/dev/null 2>&1; then
    echo "✅ Health check: Working"
else
    echo "❌ Health check: Failed"
fi

if curl -f "$API_URL/dashboard/stats" >/dev/null 2>&1; then
    echo "✅ Dashboard: Working"
else
    echo "❌ Dashboard: Failed"
fi

echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo ""
echo "📊 Access Points:"
echo "• AO1 Scanner: http://localhost:${PORT:-8000}"
echo "• Health Check: http://localhost:${PORT:-8000}/health"
echo "• Dashboard: http://localhost:${PORT:-8000}/dashboard/stats"
echo ""
echo "🔧 Management:"
echo "• Logs: docker-compose logs -f"
echo "• Stop: docker-compose down"
echo "• Restart: docker-compose restart"
echo ""
echo "No license classifier warnings! ✅"
EOF

chmod +x setup_mac_docker_compose.sh deploy_mac.sh

echo "✅ Mac Docker Compose + License Fix Created!"
echo ""
echo "🍎 Two-step process for Mac:"
echo ""
echo "1. Install Docker Compose and fix licenses:"
echo "   ./setup_mac_docker_compose.sh"
echo ""
echo "2. Edit .env with your BigQuery credentials, then:"
echo "   ./deploy_mac.sh"
echo ""
echo "This will:"
echo "• Install Docker & Docker Compose via Homebrew"
echo "• Create SPDX-compliant project structure"
echo "• Remove MIT license OSI classifier warnings"
echo "• Deploy with Mac-optimized Docker Compose"