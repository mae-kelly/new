#!/bin/bash

cat > use_crt_key_no_sudo.sh << 'EOF'
#!/bin/bash

echo "🔐 Using Your CRT/KEY Files (No Sudo Required)"
echo "=============================================="

# Find your certificate files
CRT_FILE=$(find ssl/ -name "*.crt" | head -1)
KEY_FILE=$(find ssl/ -name "*.key" | head -1)

if [ ! -f "$CRT_FILE" ]; then
    echo "❌ No .crt file found in ssl/ folder"
    ls -la ssl/
    exit 1
fi

if [ ! -f "$KEY_FILE" ]; then
    echo "❌ No .key file found in ssl/ folder"
    ls -la ssl/
    exit 1
fi

echo "✅ Found certificate: $CRT_FILE"
echo "✅ Found key: $KEY_FILE"

echo "Step 1: Adding certificate to user keychain (no sudo needed)..."

# Add to user keychain only (no admin required)
security add-trusted-cert -d -r trustRoot -k ~/Library/Keychains/login.keychain "$CRT_FILE"
echo "✅ Added certificate to user keychain"

echo "Step 2: Configuring Docker Desktop settings directly..."

# Configure Docker Desktop to bypass certificate verification
DOCKER_CONFIG_DIR="$HOME/.docker"
mkdir -p "$DOCKER_CONFIG_DIR"

# Create daemon.json with insecure registries (bypasses cert verification)
cat > "$DOCKER_CONFIG_DIR/daemon.json" << 'JSON'
{
  "insecure-registries": [
    "registry-1.docker.io",
    "docker.io",
    "index.docker.io",
    "registry.docker.io"
  ],
  "registry-mirrors": [],
  "dns": ["8.8.8.8", "1.1.1.1"],
  "experimental": false,
  "features": {
    "buildkit": true
  }
}
JSON

echo "✅ Created Docker daemon configuration"

# Add certificates to Docker-specific directories
REGISTRIES=("registry-1.docker.io" "docker.io" "index.docker.io")

for registry in "${REGISTRIES[@]}"; do
    mkdir -p "$DOCKER_CONFIG_DIR/certs.d/$registry"
    cp "$CRT_FILE" "$DOCKER_CONFIG_DIR/certs.d/$registry/ca.crt"
    cp "$KEY_FILE" "$DOCKER_CONFIG_DIR/certs.d/$registry/client.key"
    cp "$CRT_FILE" "$DOCKER_CONFIG_DIR/certs.d/$registry/client.cert"
    echo "✅ Added certificates for $registry"
done

echo "Step 3: Restarting Docker Desktop..."

# Stop Docker Desktop
osascript -e 'quit app "Docker"' 2>/dev/null || true
sleep 5

# Start Docker Desktop
open /Applications/Docker.app
sleep 15

# Wait for Docker daemon
echo "Waiting for Docker daemon..."
for i in {1..30}; do
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker daemon ready"
        break
    else
        echo "  Waiting... ($i/30)"
        sleep 2
    fi
done

echo "Step 4: Testing Docker connectivity with your certificates..."

# Test Docker Hub access
if timeout 45 docker pull hello-world >/dev/null 2>&1; then
    echo "✅ Docker Hub working with your certificates!"
    docker rmi hello-world >/dev/null 2>&1
    
    # Test Python image
    if timeout 60 docker pull python:3.11-slim >/dev/null 2>&1; then
        echo "✅ Python image available"
        echo "🎉 Certificate configuration successful!"
    else
        echo "⚠️ Python image still having issues"
    fi
else
    echo "❌ Docker Hub still blocked"
    echo "Your corporate network may be doing deep packet inspection"
fi

echo ""
echo "Certificate configuration complete!"
echo "Files configured:"
echo "• User keychain: ~/Library/Keychains/login.keychain"
echo "• Docker daemon: $DOCKER_CONFIG_DIR/daemon.json"
echo "• Docker certs: $DOCKER_CONFIG_DIR/certs.d/"
EOF

cat > build_with_your_certs.sh << 'EOF'
#!/bin/bash

echo "🚀 Build AO1 Scanner with Your Certificates"
echo "==========================================="

# Configure certificates
./use_crt_key_no_sudo.sh

if [ ! -f ".env" ]; then
    echo "❌ .env file missing"
    exit 1
fi

source .env

if [ "$BIGQUERY_PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Edit .env with BigQuery credentials"
    exit 1
fi

echo "Creating project structure..."
mkdir -p credentials data outputs logs results

# Create BigQuery credentials
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

# Create requirements.txt
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

echo "Testing Docker image pull..."
if timeout 60 docker pull python:3.11-slim >/dev/null 2>&1; then
    echo "✅ Using Python base image"
    
    # Create Dockerfile with your certificates
    cat > Dockerfile << 'DOCKERFILE'
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy YOUR SSL certificates into the container
COPY ssl/*.crt /usr/local/share/ca-certificates/
COPY ssl/*.key /usr/local/share/ca-certificates/

# Update certificate store with YOUR certificates
RUN update-ca-certificates

# Set environment variables to use YOUR certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Install requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p data outputs logs credentials results

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run_server.py"]
DOCKERFILE

elif timeout 30 docker pull alpine:latest >/dev/null 2>&1; then
    echo "✅ Using Alpine base image"
    
    cat > Dockerfile << 'DOCKERFILE'
FROM alpine:latest

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install Python and dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    curl \
    ca-certificates \
    gcc \
    musl-dev \
    python3-dev

# Copy YOUR certificates
COPY ssl/*.crt /usr/local/share/ca-certificates/
COPY ssl/*.key /usr/local/share/ca-certificates/
RUN update-ca-certificates

# Create python symlinks
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Install requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p data outputs logs credentials results

EXPOSE 8000
CMD ["python", "run_server.py"]
DOCKERFILE

else
    echo "❌ Cannot pull any base images"
    echo "Network completely blocking Docker registries"
    exit 1
fi

echo "Building AO1 Scanner with your certificates..."
docker build -t ao1-scanner:latest .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Create docker-compose.yml
    cat > docker-compose.yml << 'YAML'
version: '3.8'

services:
  ao1-scanner:
    image: ao1-scanner:latest
    ports:
      - "${PORT:-8000}:8000"
    environment:
      - DATABASE_PATH=/app/data/scanner.duckdb
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/bigquery-service-account.json
      - SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
      - REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
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
      - ./ssl:/app/ssl:ro
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

    echo "Starting services..."
    docker-compose up -d
    
    echo "Waiting for services..."
    sleep 20
    
    PORT=${PORT:-8000}
    if curl -f http://localhost:${PORT}/health >/dev/null 2>&1; then
        echo "✅ AO1 Scanner running at http://localhost:${PORT}"
        echo ""
        echo "🎉 SUCCESS with your CRT/KEY files!"
        echo "================================="
        echo "• API: http://localhost:${PORT}"
        echo "• Health: http://localhost:${PORT}/health"
        echo "• Uses YOUR SSL certificates from ssl/ folder"
    else
        echo "❌ Service not responding"
        docker-compose logs --tail=10 ao1-scanner
    fi
else
    echo "❌ Build failed"
fi
EOF

chmod +x use_crt_key_no_sudo.sh build_with_your_certs.sh

echo "✅ No-Sudo Certificate Solution Created!"
echo ""
echo "🔐 Using your existing CRT and KEY files (no sudo required):"
echo ""
echo "1. Configure Docker with your certificates:"
echo "   ./use_crt_key_no_sudo.sh"
echo ""
echo "2. Build AO1 Scanner with your certificates:"
echo "   ./build_with_your_certs.sh"
echo ""
echo "This will:"
echo "• Add your CRT to user keychain (no admin required)"
echo "• Configure Docker to use your certificates"
echo "• Copy your CRT/KEY into the Docker container"
echo "• Set up proper SSL environment variables"
echo "• Build and deploy AO1 Scanner"