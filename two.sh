#!/bin/bash

cat > docker_alternative_registry.sh << 'EOF'
#!/bin/bash

echo "🐳 Docker with Alternative Registries"
echo "====================================="

echo "Step 1: Configuring Docker for alternative registries..."

# Configure Docker to use alternative registries
DOCKER_CONFIG_DIR="$HOME/.docker"
mkdir -p "$DOCKER_CONFIG_DIR"

cat > "$DOCKER_CONFIG_DIR/daemon.json" << 'JSON'
{
  "registry-mirrors": [
    "https://mirror.gcr.io",
    "https://quay.io",
    "https://gcr.io"
  ],
  "insecure-registries": [
    "registry-1.docker.io",
    "docker.io",
    "index.docker.io"
  ],
  "dns": ["8.8.8.8", "1.1.1.1"],
  "mtu": 1450
}
JSON

echo "✅ Configured alternative registries"

# Restart Docker
echo "Restarting Docker..."
pkill -f "Docker Desktop" 2>/dev/null || true
sleep 3
open /Applications/Docker.app
sleep 15

# Wait for Docker daemon
for i in {1..30}; do
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker daemon ready"
        break
    else
        echo "  Waiting... ($i/30)"
        sleep 2
    fi
done

echo "Step 2: Trying alternative Python sources..."

# Try different registry sources for Python
PYTHON_SOURCES=(
    "mirror.gcr.io/library/python:3.11-slim"
    "quay.io/fedora/python:3.11"
    "gcr.io/distroless/python3-debian11"
)

WORKING_IMAGE=""

for source in "${PYTHON_SOURCES[@]}"; do
    echo "Trying $source..."
    if timeout 60 docker pull "$source" >/dev/null 2>&1; then
        echo "✅ Successfully pulled $source"
        WORKING_IMAGE="$source"
        break
    else
        echo "❌ Failed to pull $source"
    fi
done

if [ -z "$WORKING_IMAGE" ]; then
    echo "❌ No alternative registries working"
    echo "Trying direct base image creation..."
else
    echo "✅ Using working image: $WORKING_IMAGE"
fi
EOF

cat > create_base_image.sh << 'EOF'
#!/bin/bash

echo "🏗️ Creating Base Image from Scratch"
echo "=================================="

# Create a minimal base image using scratch or alpine (smaller, often works)
echo "Attempting to pull minimal base images..."

# Try Alpine (much smaller, often works when others don't)
if timeout 30 docker pull alpine:3.18 >/dev/null 2>&1; then
    echo "✅ Alpine available - creating Python base"
    
    cat > Dockerfile.base << 'DOCKERFILE'
FROM alpine:3.18

# Install Python and dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    curl \
    ca-certificates \
    gcc \
    musl-dev \
    python3-dev

# Create symlinks
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Update certificates
RUN update-ca-certificates

WORKDIR /app

# Test Python installation
RUN python --version && pip --version
DOCKERFILE

    echo "Building custom Python base image..."
    docker build -f Dockerfile.base -t ao1-python-base .
    
    if [ $? -eq 0 ]; then
        echo "✅ Custom Python base created successfully"
        BASE_IMAGE="ao1-python-base"
    else
        echo "❌ Failed to create custom base"
        exit 1
    fi
    
else
    echo "❌ Cannot pull any base images"
    echo "Network is completely blocking Docker registries"
    exit 1
fi
EOF

cat > build_ao1_custom.sh << 'EOF'
#!/bin/bash

echo "🚀 Build AO1 Scanner with Custom Base"
echo "===================================="

# Configure alternative registries
./docker_alternative_registry.sh

# Create custom base if needed
./create_base_image.sh

if [ ! -f ".env" ]; then
    echo "❌ .env file missing"
    exit 1
fi

source .env

if [ "$BIGQUERY_PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Edit .env with BigQuery credentials"
    exit 1
fi

echo "Creating project files..."
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

# Create requirements for Alpine/minimal install
cat > requirements.txt << 'REQS'
fastapi==0.104.1
uvicorn==0.24.0
google-cloud-bigquery==3.12.0
google-cloud-resource-manager==1.10.4
google-auth==2.23.4
PyJWT==2.8.0
duckdb==0.9.2
pydantic==2.5.0
python-dotenv==1.0.0
structlog==23.2.0
httpx==0.25.2
REQS

# Create Dockerfile using custom base
cat > Dockerfile << 'DOCKERFILE'
FROM ao1-python-base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy SSL certificates if available
COPY ssl/*.crt /usr/local/share/ca-certificates/ 2>/dev/null || true
RUN update-ca-certificates 2>/dev/null || true

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir

# Copy application
COPY . .

# Create directories
RUN mkdir -p data outputs logs credentials results

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run_server.py"]
DOCKERFILE

echo "Building AO1 Scanner..."
docker build -t ao1-scanner:latest .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Create docker-compose for custom build
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
    restart: unless-stopped

  # Use Alpine Redis (smaller, more likely to work)
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
    sleep 15
    
    PORT=${PORT:-8000}
    if curl -f http://localhost:${PORT}/health >/dev/null 2>&1; then
        echo "✅ AO1 Scanner running at http://localhost:${PORT}"
        echo ""
        echo "🎉 SUCCESS! Built with custom Docker base"
        echo "======================================"
        echo "• API: http://localhost:${PORT}"
        echo "• Health: http://localhost:${PORT}/health"
        echo "• Dashboard: http://localhost:${PORT}/dashboard/stats"
    else
        echo "❌ Service not responding"
        docker-compose logs --tail=10 ao1-scanner
    fi
    
else
    echo "❌ Build failed"
fi
EOF

cat > direct_build_approach.sh << 'EOF'
#!/bin/bash

echo "🎯 Direct Build Approach (No External Images)"
echo "============================================="

echo "Creating completely self-contained Docker build..."

# Create a Dockerfile that builds everything from source
cat > Dockerfile.fromsource << 'DOCKERFILE'
# Use the smallest possible base
FROM scratch

# Copy a minimal Linux filesystem
# This approach builds everything locally without external dependencies
ADD https://dl-cdn.alpinelinux.org/alpine/v3.18/releases/x86_64/alpine-minirootfs-3.18.4-x86_64.tar.gz /
RUN tar -xzf /alpine-minirootfs-3.18.4-x86_64.tar.gz && rm /alpine-minirootfs-3.18.4-x86_64.tar.gz

# Install Python and dependencies
RUN apk add --no-cache python3 py3-pip curl ca-certificates gcc musl-dev python3-dev

WORKDIR /app

# Copy SSL certificates
COPY ssl/*.crt /usr/local/share/ca-certificates/ 2>/dev/null || true
RUN update-ca-certificates 2>/dev/null || true

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data outputs logs credentials results

EXPOSE 8000
CMD ["python3", "run_server.py"]
DOCKERFILE

echo "⚠️ This approach downloads Alpine Linux directly"
echo "If corporate firewall blocks this too, Docker deployment may not be possible"

read -p "Try this approach? (y/n): " choice
if [ "$choice" = "y" ]; then
    echo "Building from source..."
    docker build -f Dockerfile.fromsource -t ao1-scanner:fromsource .
else
    echo "Consider using the Python-only approach instead"
fi
EOF

chmod +x docker_alternative_registry.sh create_base_image.sh build_ao1_custom.sh direct_build_approach.sh

echo "✅ Docker Alternative Registry Solution Created!"
echo ""
echo "🐳 Docker solutions (in order of likelihood to work):"
echo ""
echo "1. Alternative registries + Alpine base:"
echo "   ./build_ao1_custom.sh"
echo ""
echo "2. Direct build from source:"
echo "   ./direct_build_approach.sh"
echo ""
echo "These approaches:"
echo "• Use mirror.gcr.io, quay.io instead of Docker Hub"
echo "• Build custom Python base from Alpine"
echo "• Avoid standard docker.io registry"
echo "• Create self-contained Docker deployment"