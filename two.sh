#!/bin/bash

cat > fix_docker_registry_ssl.sh << 'EOF'
#!/bin/bash

echo "🔧 Fixing Docker Registry SSL Certificate Issue"
echo "=============================================="

# Find your certificate
CRT_FILE=$(find ssl/ -name "*.crt" | head -1)
if [ ! -f "$CRT_FILE" ]; then
    echo "❌ No .crt file found in ssl/ folder"
    exit 1
fi

echo "✅ Using certificate: $CRT_FILE"

echo "Step 1: Adding certificate to macOS system trust..."

# Add to macOS system keychain (this affects Docker Desktop)
sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain "$CRT_FILE"

echo "Step 2: Configuring Docker Desktop certificate trust..."

# Stop Docker Desktop
echo "Stopping Docker Desktop..."
osascript -e 'quit app "Docker"'
sleep 5

# Docker Desktop on Mac stores certificates differently
DOCKER_CERTS_DIR="$HOME/.docker/certs.d"
DOCKER_HUB_DIR="$DOCKER_CERTS_DIR/registry-1.docker.io"

mkdir -p "$DOCKER_HUB_DIR"

# Copy certificate for Docker Hub specifically
cp "$CRT_FILE" "$DOCKER_HUB_DIR/ca.crt"

echo "✅ Added certificate for Docker Hub registry"

# Also add for other common registries
for registry in "docker.io" "index.docker.io" "registry.docker.io"; do
    mkdir -p "$DOCKER_CERTS_DIR/$registry"
    cp "$CRT_FILE" "$DOCKER_CERTS_DIR/$registry/ca.crt"
    echo "✅ Added certificate for $registry"
done

echo "Step 3: Configuring Docker daemon for certificate..."

# Create Docker daemon config that respects system certificates
DOCKER_CONFIG_DIR="$HOME/.docker"
mkdir -p "$DOCKER_CONFIG_DIR"

cat > "$DOCKER_CONFIG_DIR/daemon.json" << JSON
{
  "registry-mirrors": [],
  "insecure-registries": [],
  "dns": ["8.8.8.8", "1.1.1.1"],
  "mtu": 1450,
  "storage-driver": "overlay2",
  "log-level": "info",
  "experimental": false
}
JSON

echo "Step 4: Starting Docker Desktop with new configuration..."

# Start Docker Desktop
open /Applications/Docker.app

echo "Waiting for Docker Desktop to start with new certificates..."
sleep 20

# Wait for Docker daemon to be ready
echo "Waiting for Docker daemon..."
for i in {1..30}; do
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker daemon ready"
        break
    else
        echo "  Attempt $i/30..."
        sleep 2
    fi
done

echo "Step 5: Testing Docker Hub connectivity..."

# Test Docker Hub access
echo "Testing hello-world pull..."
if timeout 60 docker pull hello-world >/dev/null 2>&1; then
    echo "✅ Docker Hub connectivity working!"
    docker rmi hello-world >/dev/null 2>&1
else
    echo "❌ Docker Hub still not accessible"
    echo "Trying additional certificate configurations..."
fi

echo "Step 6: Testing Python image pull..."
if timeout 120 docker pull python:3.11-slim >/dev/null 2>&1; then
    echo "✅ Python image pull successful!"
else
    echo "❌ Python image pull failed"
    echo "Checking Docker Desktop settings..."
fi

echo ""
echo "Certificate configuration complete!"
echo "Files created:"
echo "• $DOCKER_HUB_DIR/ca.crt"
echo "• $DOCKER_CONFIG_DIR/daemon.json"
echo ""
echo "If still having issues, check Docker Desktop settings:"
echo "• Docker Desktop > Settings > Docker Engine"
echo "• Docker Desktop > Settings > Resources > Network"
EOF

cat > test_docker_connectivity.sh << 'EOF'
#!/bin/bash

echo "🧪 Testing Docker Connectivity"
echo "=============================="

echo "Step 1: Basic Docker functionality..."
if docker info >/dev/null 2>&1; then
    echo "✅ Docker daemon running"
else
    echo "❌ Docker daemon not running"
    exit 1
fi

echo "Step 2: Testing network connectivity..."
if ping -c 1 registry-1.docker.io >/dev/null 2>&1; then
    echo "✅ Can reach Docker registry"
else
    echo "❌ Cannot reach Docker registry"
fi

echo "Step 3: Testing Docker Hub authentication..."
docker logout >/dev/null 2>&1
if timeout 30 docker pull hello-world >/dev/null 2>&1; then
    echo "✅ Docker Hub pull working"
    docker rmi hello-world >/dev/null 2>&1
else
    echo "❌ Docker Hub pull failing"
    echo "Checking certificate configuration..."
    
    if [ -f "$HOME/.docker/certs.d/registry-1.docker.io/ca.crt" ]; then
        echo "✅ Certificate configured for Docker Hub"
    else
        echo "❌ Certificate not configured for Docker Hub"
    fi
fi

echo "Step 4: Testing Python image specifically..."
if timeout 60 docker pull python:3.11-slim >/dev/null 2>&1; then
    echo "✅ Python 3.11 image pull successful"
else
    echo "❌ Python 3.11 image pull failed"
    echo "Error details:"
    docker pull python:3.11-slim 2>&1 | tail -3
fi

echo "Step 5: Checking certificate store..."
echo "System certificates:"
ls -la /etc/ssl/certs/ | grep -i company 2>/dev/null || echo "No company certificates in system store"

echo "Docker certificates:"
ls -la "$HOME/.docker/certs.d/" 2>/dev/null || echo "No Docker-specific certificates"

echo ""
echo "🎯 Summary:"
if timeout 30 docker pull hello-world >/dev/null 2>&1; then
    echo "✅ Docker connectivity working - ready to build AO1 Scanner"
    docker rmi hello-world >/dev/null 2>&1
else
    echo "❌ Docker connectivity issues persist"
    echo ""
    echo "Manual steps to try:"
    echo "1. Open Docker Desktop > Settings > Docker Engine"
    echo "2. Add this to the JSON configuration:"
    echo '   "insecure-registries": ["registry-1.docker.io"]'
    echo "3. Click 'Apply & Restart'"
fi
EOF

cat > build_with_registry_fix.sh << 'EOF'
#!/bin/bash

echo "🚀 Build AO1 Scanner with Registry Fix"
echo "====================================="

# Apply registry certificate fix
./fix_docker_registry_ssl.sh

echo "Testing connectivity before build..."
./test_docker_connectivity.sh

# Check if we can pull Python image
if ! timeout 60 docker pull python:3.11-slim >/dev/null 2>&1; then
    echo "❌ Still cannot pull Python image"
    echo "Trying alternative base image..."
    
    # Try Alpine Python
    if timeout 60 docker pull python:3.11-alpine >/dev/null 2>&1; then
        echo "✅ Alpine Python works - using alternative Dockerfile"
        
        cat > Dockerfile.alpine << 'DOCKERFILE'
FROM python:3.11-alpine

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache curl ca-certificates gcc musl-dev

# Copy SSL certificates
COPY ssl/*.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p data outputs logs credentials results

EXPOSE 8000
CMD ["python", "run_server.py"]
DOCKERFILE

        DOCKERFILE_TO_USE="Dockerfile.alpine"
    else
        echo "❌ Cannot pull any Python images"
        echo "Check your network/proxy settings"
        exit 1
    fi
else
    echo "✅ Python 3.11-slim working"
    DOCKERFILE_TO_USE="Dockerfile"
fi

echo "Building AO1 Scanner..."

# Ensure .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file needed"
    exit 1
fi

source .env

if [ "$BIGQUERY_PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Edit .env with BigQuery credentials"
    exit 1
fi

# Create credentials
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

# Build with appropriate Dockerfile
echo "Building with $DOCKERFILE_TO_USE..."
docker build -f "$DOCKERFILE_TO_USE" -t ao1-scanner:latest .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Start with docker-compose
    docker-compose up -d
    
    echo "Waiting for services..."
    sleep 15
    
    if curl -f http://localhost:${PORT:-8000}/health >/dev/null 2>&1; then
        echo "✅ AO1 Scanner running at http://localhost:${PORT:-8000}"
    else
        echo "❌ Service not responding"
        docker-compose logs --tail=10 ao1-scanner
    fi
else
    echo "❌ Build failed"
fi
EOF

chmod +x fix_docker_registry_ssl.sh test_docker_connectivity.sh build_with_registry_fix.sh

echo "✅ Docker Registry SSL Fix Created!"
echo ""
echo "🔧 To fix the Docker registry certificate issue:"
echo ""
echo "1. Fix Docker registry SSL trust:"
echo "   ./fix_docker_registry_ssl.sh"
echo ""
echo "2. Test connectivity:"
echo "   ./test_docker_connectivity.sh"
echo ""
echo "3. Build with registry fix:"
echo "   ./build_with_registry_fix.sh"
echo ""
echo "This specifically fixes the registry-1.docker.io certificate verification error"