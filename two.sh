#!/bin/bash

cat > deploy.sh << 'EOF'
#!/bin/bash

set -e

DEPLOYMENT_TYPE=${1:-docker}
ENVIRONMENT=${2:-production}

echo "Deploying AO1 Scanner - Type: $DEPLOYMENT_TYPE, Environment: $ENVIRONMENT"

if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
    echo "Deploying to Kubernetes..."
    
    if ! command -v kubectl &> /dev/null; then
        echo "kubectl not found. Please install kubectl first."
        exit 1
    fi
    
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    
    if [ -f "k8s/secret-configured.yaml" ]; then
        kubectl apply -f k8s/secret-configured.yaml
    else
        echo "⚠️  Secrets not configured. Run ./configure_secrets.sh first"
        kubectl apply -f k8s/secret.yaml
    fi
    
    kubectl apply -f k8s/pvc.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/ingress.yaml
    
    echo "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/ao1-scanner -n ao1-scanner
    
    echo "Deployment complete!"
    kubectl get pods -n ao1-scanner
    
elif [ "$DEPLOYMENT_TYPE" = "docker" ]; then
    echo "Deploying with Docker Compose..."
    
    if [ ! -f ".env" ]; then
        echo "Creating .env from template..."
        cp .env.example .env
    fi
    
    if [ ! -f "docker-compose.prod.yml" ]; then
        echo "docker-compose.prod.yml not found. Creating basic version..."
        cat > docker-compose.prod.yml << 'YAML'
version: '3.8'
services:
  ao1-scanner:
    image: ao1-scanner:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_PATH=/app/data/scanner.duckdb
      - ENVIRONMENT=production
      - LOG_LEVEL=info
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./logs:/app/logs
      - ./credentials:/app/credentials
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
YAML
    fi
    
    echo "Creating required directories..."
    mkdir -p data outputs logs credentials results
    
    echo "Starting services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    echo "Waiting for services to be ready..."
    sleep 15
    
    echo "Checking service health..."
    for i in {1..10}; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            echo "✅ Service is healthy!"
            break
        else
            echo "Waiting for service... (attempt $i/10)"
            sleep 5
        fi
    done
    
    echo "Deployment complete!"
    docker-compose -f docker-compose.prod.yml ps
    
else
    echo "Unknown deployment type: $DEPLOYMENT_TYPE"
    echo "Usage: $0 {docker|kubernetes} [environment]"
    exit 1
fi

echo ""
echo "🎉 AO1 Scanner deployed successfully!"
echo "• API: http://localhost:8000"
echo "• Health: http://localhost:8000/health"
echo "• Dashboard: http://localhost:8000/dashboard/stats"
echo "• Metrics: http://localhost:8000/metrics"
EOF

cat > verify_deployment.sh << 'EOF'
#!/bin/bash

echo "🔍 Verifying AO1 Scanner deployment..."

API_URL=${1:-http://localhost:8000}
MAX_RETRIES=30
RETRY_COUNT=0

echo "Testing basic connectivity..."
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f "$API_URL/health" > /dev/null 2>&1; then
        echo "✅ Health check passed"
        break
    else
        echo "⏳ Waiting for service... (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)"
        sleep 2
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ Service failed to start after $MAX_RETRIES attempts"
    echo "Check logs with: docker-compose logs ao1-scanner"
    exit 1
fi

echo "Testing API endpoints..."

echo "• Basic health:"
if curl -f "$API_URL/health" > /dev/null 2>&1; then
    echo "  ✅ Basic health check"
else
    echo "  ❌ Basic health check failed"
fi

echo "• Detailed health:"
HEALTH_RESPONSE=$(curl -s "$API_URL/health/detailed" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "  ✅ Detailed health endpoint"
    OVERALL_STATUS=$(echo "$HEALTH_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('overall_status', 'unknown'))" 2>/dev/null || echo "unknown")
    echo "  📊 Status: $OVERALL_STATUS"
else
    echo "  ❌ Detailed health endpoint failed"
fi

echo "Testing database connectivity..."
if echo "$HEALTH_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('components', {}).get('database', {}).get('status', 'unknown'))" 2>/dev/null | grep -q "healthy"; then
    echo "  ✅ Database connection"
else
    echo "  ❌ Database connection issues"
fi

echo "Testing BigQuery connectivity..."
if echo "$HEALTH_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('components', {}).get('bigquery', {}).get('status', 'unknown'))" 2>/dev/null | grep -q "healthy"; then
    echo "  ✅ BigQuery connection"
else
    echo "  ⚠️  BigQuery connection issues"
    echo "     Check credentials and permissions"
fi

echo ""
echo "🎯 Verification complete!"
echo "✅ AO1 Scanner is running"
EOF

cat > configure_secrets.sh << 'EOF'
#!/bin/bash

echo "🔐 Configuring AO1 Scanner secrets..."

create_jwt_secret() {
    if command -v openssl &> /dev/null; then
        openssl rand -hex 32
    elif command -v python3 &> /dev/null; then
        python3 -c "import secrets; print(secrets.token_hex(32))"
    else
        echo "jwt_secret_$(date +%s)_$(whoami)" | tr -d '\n' | base64 | head -c 64
    fi
}

mkdir -p credentials

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from template"
fi

JWT_SECRET=$(create_jwt_secret)

if ! grep -q "JWT_SECRET_KEY=" .env; then
    echo "JWT_SECRET_KEY=$JWT_SECRET" >> .env
    echo "✅ Added JWT secret to .env"
else
    if [ "$(uname)" = "Darwin" ]; then
        sed -i '' "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET/" .env
    else
        sed -i "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET/" .env
    fi
    echo "✅ Updated JWT secret in .env"
fi

if [ -f "credentials/bigquery-service-account.json" ]; then
    echo "✅ BigQuery service account found"
    
    if command -v base64 &> /dev/null; then
        if [ "$(uname)" = "Darwin" ]; then
            GCP_CREDS_B64=$(base64 credentials/bigquery-service-account.json)
        else
            GCP_CREDS_B64=$(base64 -w 0 credentials/bigquery-service-account.json)
        fi
        
        if [ "$(uname)" = "Darwin" ]; then
            JWT_B64=$(echo -n "$JWT_SECRET" | base64)
        else
            JWT_B64=$(echo -n "$JWT_SECRET" | base64 -w 0)
        fi
        
        cat > k8s/secret-configured.yaml << YAML
apiVersion: v1
kind: Secret
metadata:
  name: ao1-scanner-secrets
  namespace: ao1-scanner
type: Opaque
data:
  jwt-secret: $JWT_B64
  gcp-credentials: $GCP_CREDS_B64
YAML
        echo "✅ Kubernetes secret configured"
    fi
    
    if ! grep -q "GOOGLE_APPLICATION_CREDENTIALS=" .env; then
        echo "GOOGLE_APPLICATION_CREDENTIALS=credentials/bigquery-service-account.json" >> .env
        echo "✅ Added BigQuery credentials path to .env"
    fi
    
else
    echo "⚠️  BigQuery service account not found"
    echo "Please place your service account JSON at: credentials/bigquery-service-account.json"
    echo ""
    echo "To get the service account:"
    echo "1. Go to Google Cloud Console > IAM & Admin > Service Accounts"
    echo "2. Create a service account with these roles:"
    echo "   - roles/bigquery.dataViewer"
    echo "   - roles/bigquery.jobUser"
    echo "   - roles/resourcemanager.projectViewer"
    echo "3. Download the JSON key file"
    echo "4. Save it as credentials/bigquery-service-account.json"
fi

echo ""
echo "🎯 Secret configuration complete!"
EOF

cat > quick_start.sh << 'EOF'
#!/bin/bash

echo "🚀 AO1 Scanner Quick Start"
echo "========================="

if [ ! -f "credentials/bigquery-service-account.json" ]; then
    echo "⚠️  Missing BigQuery service account credentials"
    echo ""
    echo "Quick setup:"
    echo "1. Download your BigQuery service account JSON key"
    echo "2. Save it as: credentials/bigquery-service-account.json"
    echo "3. Run this script again"
    echo ""
    exit 1
fi

echo "Step 1: Configuring secrets..."
./configure_secrets.sh

echo ""
echo "Step 2: Deploying with Docker..."
./deploy.sh docker

echo ""
echo "Step 3: Verifying deployment..."
./verify_deployment.sh

echo ""
echo "🎉 Quick start complete!"
echo ""
echo "Access your AO1 Scanner:"
echo "• Web API: http://localhost:8000"
echo "• Health Check: http://localhost:8000/health/detailed"
echo "• Dashboard: http://localhost:8000/dashboard/stats"
EOF

chmod +x deploy.sh verify_deployment.sh configure_secrets.sh quick_start.sh

echo "✅ Fixed deployment scripts created!"
echo ""
echo "Now run:"
echo "./quick_start.sh"