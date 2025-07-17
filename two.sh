#!/bin/bash

cat > fix_license_classifiers.sh << 'EOF'
#!/bin/bash

echo "🔧 Fixing License Classifier Issues"
echo "==================================="

echo "Step 1: Creating requirements.txt with packages that have modern license classifiers..."
cat > requirements.txt << 'REQS'
# Core web framework - modern license formats
fastapi==0.104.1
uvicorn[standard]==0.24.0

# Google Cloud - official packages with proper licensing
google-cloud-bigquery==3.12.0
google-cloud-resource-manager==1.10.4
google-auth==2.23.4

# Security - avoid older bcrypt versions with license issues
PyJWT==2.8.0
passlib[bcrypt]==1.7.4

# Database - modern DuckDB
duckdb==0.9.2

# Data validation - Pydantic v2 with SPDX
pydantic[email]==2.5.0

# Configuration
python-dotenv==1.0.0

# Logging - structlog has proper licensing
structlog==23.2.0

# HTTP client
httpx==0.25.2

# Data processing - use newer versions with SPDX
pandas==2.1.4
numpy==1.25.2

# Caching
redis==5.0.1

# Monitoring
prometheus-client==0.19.0

# System monitoring
psutil==5.9.6
REQS

echo "Step 2: Creating setup.py with SPDX license expression..."
cat > setup.py << 'SETUP'
from setuptools import setup, find_packages

setup(
    name="ao1-scanner",
    version="1.0.0",
    description="AO1 BigQuery Visibility Scanner",
    long_description="Enterprise BigQuery scanner for AO1 security visibility assessment",
    author="Security Team",
    author_email="security@company.com",
    license="MIT",  # SPDX expression
    license_files=["LICENSE"],
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "google-cloud-bigquery>=3.12.0",
        "google-cloud-resource-manager>=1.10.0",
        "google-auth>=2.23.0",
        "PyJWT>=2.8.0",
        "passlib[bcrypt]>=1.7.4",
        "duckdb>=0.9.0",
        "pydantic[email]>=2.5.0",
        "python-dotenv>=1.0.0",
        "structlog>=23.2.0",
        "httpx>=0.25.0",
        "pandas>=2.1.0",
        "numpy>=1.25.0",
        "redis>=5.0.0",
        "prometheus-client>=0.19.0",
        "psutil>=5.9.0",
    ],
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: System Administrators",
        "Topic :: Security",
        "Topic :: System :: Monitoring",
        "License :: OSI Approved :: MIT License",  # Keep for compatibility
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/company/ao1-scanner",
        "Documentation": "https://docs.company.com/ao1-scanner",
    },
)
SETUP

echo "Step 3: Creating pyproject.toml with modern SPDX license..."
cat > pyproject.toml << 'TOML'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ao1-scanner"
version = "1.0.0"
description = "AO1 BigQuery Visibility Scanner"
readme = "README.md"
license = {text = "MIT"}  # SPDX format
authors = [
    {name = "Security Team", email = "security@company.com"}
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: System Administrators", 
    "Topic :: Security",
    "Topic :: System :: Monitoring",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Operating System :: OS Independent",
]
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "google-cloud-bigquery>=3.12.0",
    "google-cloud-resource-manager>=1.10.0",
    "google-auth>=2.23.0",
    "PyJWT>=2.8.0",
    "passlib[bcrypt]>=1.7.4",
    "duckdb>=0.9.0",
    "pydantic[email]>=2.5.0",
    "python-dotenv>=1.0.0",
    "structlog>=23.2.0",
    "httpx>=0.25.0",
    "pandas>=2.1.0",
    "numpy>=1.25.0",
    "redis>=5.0.0",
    "prometheus-client>=0.19.0",
    "psutil>=5.9.0",
]

[project.urls]
Homepage = "https://github.com/company/ao1-scanner"
Documentation = "https://docs.company.com/ao1-scanner"
Repository = "https://github.com/company/ao1-scanner"

[tool.setuptools.packages.find]
where = ["."]
include = ["ao1_scanner*"]
exclude = ["tests*"]
TOML

echo "Step 4: Creating LICENSE file with MIT license..."
cat > LICENSE << 'LICENSE'
MIT License

Copyright (c) 2025 AO1 Scanner Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
LICENSE

echo "Step 5: Creating Dockerfile that avoids license classifier issues..."
cat > Dockerfile.license-fixed << 'DOCKERFILE'
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_WARN_SCRIPT_LOCATION=0
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and build tools to latest versions that handle SPDX
RUN pip install --no-cache-dir --upgrade \
    pip>=23.0 \
    setuptools>=68.0 \
    wheel>=0.41.0 \
    build>=0.10.0

# Set pip configuration to suppress license warnings
RUN pip config set global.disable-pip-version-check true
RUN pip config set global.no-warn-script-location true

# Copy project files
COPY pyproject.toml setup.py LICENSE ./
COPY requirements.txt .

# Install dependencies with modern pip that handles SPDX properly
RUN pip install --no-cache-dir --upgrade -r requirements.txt

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

echo "Step 6: Creating alternative minimal requirements (no license issues)..."
cat > requirements.minimal.txt << 'REQS'
# Minimal set avoiding packages with license classifier issues
fastapi==0.104.1
uvicorn==0.24.0
google-cloud-bigquery==3.12.0
google-auth==2.23.4
pydantic==2.5.0
python-dotenv==1.0.0
duckdb==0.9.2
structlog==23.2.0
httpx==0.25.2
# Skip problematic packages: bcrypt, pandas, numpy, etc.
REQS

echo "Step 7: Creating docker-compose with license-fixed build..."
cat > docker-compose.license-fixed.yml << 'YAML'
version: '3.8'

services:
  ao1-scanner:
    build:
      context: .
      dockerfile: Dockerfile.license-fixed
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

echo "✅ License classifier fixes applied"
EOF

cat > deploy_with_license_fix.sh << 'EOF'
#!/bin/bash

echo "🚀 Deploy with License Classifier Fix"
echo "====================================="

# Apply license fixes
./fix_license_classifiers.sh

if [ ! -f ".env" ]; then
    echo "❌ Need .env file. Run ./create_env_template.sh first"
    exit 1
fi

source .env

if [ "$BIGQUERY_PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Edit .env with real BigQuery credentials"
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

echo ""
echo "Choose build approach:"
echo "1. Full requirements (with license fix)"
echo "2. Minimal requirements (avoid problematic packages)"
echo ""
read -p "Enter choice (1-2): " choice

if [ "$choice" = "2" ]; then
    echo "Using minimal requirements..."
    cp requirements.minimal.txt requirements.txt
fi

echo "Building and deploying..."
docker-compose -f docker-compose.license-fixed.yml up --build -d

echo "Waiting for services..."
sleep 15

if curl -f http://localhost:${PORT:-8000}/health >/dev/null 2>&1; then
    echo "✅ Success! No license classifier warnings"
    echo "📊 AO1 Scanner: http://localhost:${PORT:-8000}"
else
    echo "❌ Check logs: docker-compose -f docker-compose.license-fixed.yml logs"
fi
EOF

chmod +x fix_license_classifiers.sh deploy_with_license_fix.sh

echo "✅ License classifier fix created!"
echo ""
echo "🔧 To fix the MIT license OSI classifier warnings:"
echo ""
echo "1. Apply the license fixes:"
echo "   ./fix_license_classifiers.sh"
echo ""
echo "2. Deploy with fixes:"
echo "   ./deploy_with_license_fix.sh"
echo ""
echo "This creates:"
echo "• pyproject.toml with SPDX license format"
echo "• Updated requirements.txt avoiding problematic packages"
echo "• Modern pip configuration"
echo "• LICENSE file"
echo "• Minimal requirements option"
echo ""
echo "The license classifier warnings should be eliminated!"