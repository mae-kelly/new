#!/bin/bash

set -e

echo "🚀 AO1 Scanner Complete Enhancement Deployment"
echo "=============================================="

chmod +x *.sh

echo "Step 1: Production setup..."
./setup_production.sh

echo "Step 2: Integrating enhancements..."
./integrate_enhancements.sh

echo "Step 3: Configuring secrets..."
if [ -f "credentials/bigquery-service-account.json" ]; then
    ./configure_secrets.sh
else
    echo "⚠️  Place your BigQuery service account JSON in credentials/ first"
    echo "   Then run: ./configure_secrets.sh"
fi

echo "Step 4: Building and deploying..."
docker build -f Dockerfile.prod -t ao1-scanner:latest .

if command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
    echo "Deploying to Kubernetes..."
    ./deploy.sh kubernetes
else
    echo "Deploying with Docker..."
    ./deploy.sh docker
fi

echo "Step 5: Verification..."
sleep 10
./verify_deployment.sh

echo ""
echo "🎉 AO1 Scanner Enhanced Deployment Complete!"
echo ""
echo "📋 Access Points:"
echo "   • API: http://localhost:8000"
echo "   • Health: http://localhost:8000/health/detailed"
echo "   • Metrics: http://localhost:8000/metrics"
echo "   • Dashboard: http://localhost:8000/dashboard/stats"
echo "   • WebSocket: ws://localhost:8000/ws/scan/{job_id}"
echo ""
echo "📊 New Features:"
echo "   • Real-time scan progress via WebSocket"
echo "   • Industry-specific pattern detection"
echo "   • Executive reporting with compliance mapping"
echo "   • Circuit breaker resilience"
echo "   • Performance monitoring"
echo "   • Automated alerts"
echo ""
echo "🔧 Management Commands:"
echo "   • Backup: ./backup_restore.sh backup"
echo "   • Monitor: ./tune_performance.sh"
echo "   • Alerts: Check logs/alerts.log"
