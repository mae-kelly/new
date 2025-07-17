#!/bin/bash

cat > use_local_ssl_cert.sh << 'EOF'
#!/bin/bash

echo "🔐 Using Local SSL Certificate"
echo "=============================="

# Check if ssl.cert exists
if [ ! -f "ssl/ssl.cert" ]; then
    echo "❌ SSL certificate not found at ssl/ssl.cert"
    echo "Please ensure your certificate is at: ssl/ssl.cert"
    exit 1
fi

echo "✅ Found SSL certificate at ssl/ssl.cert"

echo "Step 1: Creating Dockerfile with local SSL certificate..."
cat > Dockerfile << 'DOCKERFILE'
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy local SSL certificate
COPY ssl/ssl.cert /usr/local/share/ca-certificates/ssl.crt

# Update certificate store with local certificate
RUN update-ca-certificates

# Set certificate environment variables
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Copy and install requirements
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

echo "Step 2: Creating docker-compose.yml with SSL certificate..."
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

echo "Step 3: Creating .dockerignore to include SSL certificate..."
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

# Keep SSL certificate
!ssl/
!ssl/ssl.cert
IGNORE

echo "Step 4: Creating requirements.txt with SSL-compatible packages..."
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
REQS

echo "✅ Docker configuration updated to use local SSL certificate"
EOF

cat > deploy_with_ssl_cert.sh << 'EOF'
#!/bin/bash

echo "🚀 Deploy with Local SSL Certificate"
echo "===================================="

# Check SSL certificate
if [ ! -f "ssl/ssl.cert" ]; then
    echo "❌ SSL certificate not found at ssl/ssl.cert"
    echo "Please place your SSL certificate at: ssl/ssl.cert"
    exit 1
fi

echo "✅ Using SSL certificate: ssl/ssl.cert"

# Apply SSL configuration
./use_local_ssl_cert.sh

# Check .env
if [ ! -f ".env" ]; then
    echo "❌ .env file not found"
    exit 1
fi

source .env

if [ "$BIGQUERY_PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Edit .env with your BigQuery credentials"
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

echo "Building with local SSL certificate..."
docker-compose down 2>/dev/null || true
docker-compose up --build -d

echo "Waiting for services..."
sleep 15

echo "Testing SSL certificate configuration..."
if docker-compose exec -T ao1-scanner curl -s https://www.google.com >/dev/null 2>&1; then
    echo "✅ SSL certificate working in container"
else
    echo "⚠️ SSL certificate may need additional configuration"
fi

echo "Testing AO1 Scanner..."
for i in {1..15}; do
    if curl -f http://localhost:${PORT:-8000}/health >/dev/null 2>&1; then
        echo "✅ AO1 Scanner is running with SSL certificate!"
        break
    else
        echo "⏳ Waiting for scanner... (attempt $i/15)"
        sleep 3
    fi
done

echo ""
echo "🎉 Deployment Complete with Local SSL Certificate!"
echo "================================================="
echo ""
echo "📊 Access Points:"
echo "• AO1 Scanner: http://localhost:${PORT:-8000}"
echo "• Health Check: http://localhost:${PORT:-8000}/health"
echo ""
echo "🔐 SSL Certificate:"
echo "• Loaded from: ssl/ssl.cert"
echo "• Added to container certificate store"
echo ""
echo "🔧 Management:"
echo "• Logs: docker-compose logs -f"
echo "• Stop: docker-compose down"
EOF

cat > verify_ssl_setup.sh << 'EOF'
#!/bin/bash

echo "🔍 Verifying SSL Certificate Setup"
echo "=================================="

# Check local certificate
if [ -f "ssl/ssl.cert" ]; then
    echo "✅ SSL certificate found: ssl/ssl.cert"
    
    # Show certificate info
    echo "Certificate details:"
    openssl x509 -in ssl/ssl.cert -text -noout 2>/dev/null | grep -E "(Subject|Issuer|Not Before|Not After)" || echo "  (Certificate details not readable)"
else
    echo "❌ SSL certificate not found at ssl/ssl.cert"
fi

# Check Docker configuration
if [ -f "Dockerfile" ]; then
    if grep -q "ssl.cert" Dockerfile; then
        echo "✅ Dockerfile configured for SSL certificate"
    else
        echo "❌ Dockerfile not configured for SSL certificate"
    fi
fi

# Check docker-compose configuration
if [ -f "docker-compose.yml" ]; then
    if grep -q "ssl:" docker-compose.yml; then
        echo "✅ docker-compose.yml configured for SSL"
    else
        echo "❌ docker-compose.yml not configured for SSL"
    fi
fi

# Test if container can access external HTTPS
if docker-compose ps | grep -q "Up"; then
    echo "Testing HTTPS connectivity from container..."
    if docker-compose exec -T ao1-scanner curl -s --max-time 10 https://httpbin.org/get >/dev/null 2>&1; then
        echo "✅ Container can access HTTPS with SSL certificate"
    else
        echo "⚠️ Container HTTPS access may have issues"
    fi
else
    echo "⚠️ Containers not running - start with docker-compose up -d"
fi

echo ""
echo "SSL Setup Status:"
if [ -f "ssl/ssl.cert" ] && [ -f "Dockerfile" ] && grep -q "ssl.cert" Dockerfile; then
    echo "✅ SSL certificate properly configured"
else
    echo "❌ SSL certificate setup incomplete"
    echo "Run: ./use_local_ssl_cert.sh"
fi
EOF

chmod +x use_local_ssl_cert.sh deploy_with_ssl_cert.sh verify_ssl_setup.sh

echo "✅ Local SSL Certificate Setup Created!"
echo ""
echo "🔐 To use your SSL certificate at ssl/ssl.cert:"
echo ""
echo "1. Configure Docker for your SSL certificate:"
echo "   ./use_local_ssl_cert.sh"
echo ""
echo "2. Deploy with SSL certificate:"
echo "   ./deploy_with_ssl_cert.sh"
echo ""
echo "3. Verify SSL setup:"
echo "   ./verify_ssl_setup.sh"
echo ""
echo "This will:"
echo "• Copy ssl/ssl.cert into the Docker container"
echo "• Add it to the container's certificate store"
echo "• Configure all Python requests to use it"
echo "• Mount ssl/ folder as read-only volume"