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
