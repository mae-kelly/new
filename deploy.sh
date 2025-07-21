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
