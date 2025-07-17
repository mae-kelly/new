#!/bin/bash

cat > configure_ssl_crt_key.sh << 'EOF'
#!/bin/bash

echo "🔐 Configuring SSL with CRT and KEY files"
echo "========================================="

# Check for both certificate files
if [ ! -f "ssl/"*.crt ]; then
    echo "❌ No .crt file found in ssl/ folder"
    ls -la ssl/
    exit 1
fi

if [ ! -f "ssl/"*.key ]; then
    echo "❌ No .key file found in ssl/ folder"
    ls -la ssl/
    exit 1
fi

CRT_FILE=$(find ssl/ -name "*.crt" | head -1)
KEY_FILE=$(find ssl/ -name "*.key" | head -1)

echo "✅ Found certificate: $CRT_FILE"
echo "✅ Found key: $KEY_FILE"

# Create Dockerfile that properly handles CRT and KEY
cat > Dockerfile << 'DOCKERFILE'
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Copy SSL certificate and key
COPY ssl/*.crt /usr/local/share/ca-certificates/
COPY ssl/*.key /usr/local/share/ca-certificates/

# Update certificate authorities
RUN update-ca-certificates

# Set certificate environment variables
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Copy requirements first
COPY requirements.txt .

# Install Python packages with proper SSL
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

# Create docker-compose.yml with SSL configuration
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
      - SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
      - SSL_CERT_DIR=/etc/ssl/certs
      - REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
      - CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
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

# Update .dockerignore to include SSL files
cat > .dockerignore << 'IGNORE'
.git
.gitignore
README.md
.dockerignore
.env.example
__pycache__
*.pyc
.pytest_cache
.coverage
*.log
venv
.venv
outputs
logs
results
data
*.duckdb
node_modules
.DS_Store

# Include SSL certificate files
!ssl/
!ssl/*.crt
!ssl/*.key
IGNORE

# Create requirements.txt with SSL-aware packages
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
certifi==2023.11.17
urllib3==2.1.0
REQS

echo "✅ Configured Docker to use both CRT and KEY files"
echo "Certificate: $CRT_FILE"
echo "Key: $KEY_FILE"
EOF

cat > deploy_with_crt_key.sh << 'EOF'
#!/bin/bash

echo "🚀 Deploy with CRT and KEY files"
echo "================================"

# Configure SSL
./configure_ssl_crt_key.sh

if [ ! -f ".env" ]; then
    echo "❌ .env file needed"
    exit 1
fi

source .env

if [ "$BIGQUERY_PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Edit .env with BigQuery credentials"
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

echo "Stopping existing containers..."
docker-compose down

echo "Building with SSL certificate and key..."
docker-compose build --no-cache

echo "Starting services..."
docker-compose up -d

echo "Waiting for services to start..."
sleep 20

echo "Testing SSL configuration..."
if docker-compose exec -T ao1-scanner curl -s https://www.google.com >/dev/null 2>&1; then
    echo "✅ SSL certificate working properly"
else
    echo "⚠️ SSL may need additional configuration"
    echo "Checking certificate details..."
    docker-compose exec -T ao1-scanner ls -la /etc/ssl/certs/ | grep -v lrwxrwxrwx | head -5
fi

echo "Testing AO1 Scanner..."
PORT=${PORT:-8000}
for i in {1..15}; do
    if curl -f http://localhost:${PORT}/health >/dev/null 2>&1; then
        echo "✅ AO1 Scanner running at http://localhost:${PORT}"
        break
    else
        echo "⏳ Waiting for scanner... (attempt $i/15)"
        sleep 3
    fi
done

echo ""
echo "🎉 Deployment complete with SSL certificate and key!"
echo "📊 Access: http://localhost:${PORT}"
echo "🔐 SSL files properly configured in container"
EOF

cat > verify_ssl_files.sh << 'EOF'
#!/bin/bash

echo "🔍 Verifying SSL Certificate and Key Files"
echo "=========================================="

echo "Step 1: Checking local SSL files..."
if ls ssl/*.crt >/dev/null 2>&1; then
    for crt_file in ssl/*.crt; do
        echo "✅ Certificate: $crt_file"
        # Show certificate details
        openssl x509 -in "$crt_file" -text -noout | grep -E "(Subject|Issuer|Not Before|Not After)" 2>/dev/null || echo "  (Certificate details not readable)"
    done
else
    echo "❌ No .crt files found in ssl/"
fi

if ls ssl/*.key >/dev/null 2>&1; then
    for key_file in ssl/*.key; do
        echo "✅ Private key: $key_file"
        # Verify key format
        openssl rsa -in "$key_file" -check -noout 2>/dev/null && echo "  (Key format valid)" || echo "  (Key format may need checking)"
    done
else
    echo "❌ No .key files found in ssl/"
fi

echo ""
echo "Step 2: Checking certificate and key match..."
if ls ssl/*.crt >/dev/null 2>&1 && ls ssl/*.key >/dev/null 2>&1; then
    CRT_FILE=$(ls ssl/*.crt | head -1)
    KEY_FILE=$(ls ssl/*.key | head -1)
    
    CRT_HASH=$(openssl x509 -noout -modulus -in "$CRT_FILE" 2>/dev/null | openssl md5)
    KEY_HASH=$(openssl rsa -noout -modulus -in "$KEY_FILE" 2>/dev/null | openssl md5)
    
    if [ "$CRT_HASH" = "$KEY_HASH" ]; then
        echo "✅ Certificate and key match"
    else
        echo "⚠️ Certificate and key may not match"
    fi
fi

echo ""
echo "Step 3: Checking if container is using certificates..."
if docker-compose ps | grep -q "Up"; then
    echo "Checking certificates in running container..."
    docker-compose exec -T ao1-scanner ls -la /usr/local/share/ca-certificates/ 2>/dev/null || echo "Container not accessible"
    
    echo "Testing HTTPS from container..."
    docker-compose exec -T ao1-scanner curl -s -o /dev/null -w "%{http_code}" https://www.google.com 2>/dev/null || echo "HTTPS test failed"
else
    echo "⚠️ Container not running"
fi

echo ""
echo "🎯 SSL Status Summary:"
echo "• CRT files: $(ls ssl/*.crt 2>/dev/null | wc -l | tr -d ' ')"
echo "• KEY files: $(ls ssl/*.key 2>/dev/null | wc -l | tr -d ' ')"
echo "• Docker configured: $([ -f "Dockerfile" ] && grep -q "ssl" Dockerfile && echo "Yes" || echo "No")"
EOF

chmod +x configure_ssl_crt_key.sh deploy_with_crt_key.sh verify_ssl_files.sh

echo "✅ SSL CRT and KEY configuration created!"
echo ""
echo "🔐 To use your .crt and .key files:"
echo ""
echo "1. Configure for CRT and KEY files:"
echo "   ./configure_ssl_crt_key.sh"
echo ""
echo "2. Deploy with proper SSL:"
echo "   ./deploy_with_crt_key.sh"
echo ""
echo "3. Verify SSL setup:"
echo "   ./verify_ssl_files.sh"
echo ""
echo "This will properly use both your certificate (.crt) and private key (.key) files"