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
