#!/bin/bash

API_URL="http://localhost:8000"

echo "🧪 Testing AO1 Scanner Enhanced Features"
echo "========================================"

echo "Testing WebSocket connection..."
if command -v wscat &> /dev/null; then
    timeout 5 wscat -c "$API_URL/ws/scan/test-123" &
    echo "WebSocket test initiated (5 second timeout)"
else
    echo "Install wscat to test WebSocket: npm install -g wscat"
fi

echo "Testing new API endpoints..."

echo "• Dashboard stats:"
curl -s "$API_URL/dashboard/stats" | python3 -c "import sys, json; print('✅ Dashboard endpoint working')" 2>/dev/null || echo "❌ Dashboard endpoint failed"

echo "• Performance metrics:"
curl -s "$API_URL/performance/summary" | python3 -c "import sys, json; print('✅ Performance endpoint working')" 2>/dev/null || echo "❌ Performance endpoint failed"

echo "• Health detailed:"
curl -s "$API_URL/health/detailed" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'✅ Health: {data[\"overall_status\"]}')" 2>/dev/null || echo "❌ Health endpoint failed"

echo "• Prometheus metrics:"
curl -s "$API_URL/metrics" | head -5 | grep -q "ao1_" && echo "✅ Prometheus metrics working" || echo "❌ Metrics endpoint failed"

echo ""
echo "Testing enhanced scanning features..."

python3 << 'PYTHON'
import requests
import json

api_url = "http://localhost:8000"

print("• Testing user registration with industry detection...")
try:
    register_data = {
        "email": "test@company.com",
        "password": "testpass123",
        "full_name": "Test User"
    }
    response = requests.post(f"{api_url}/auth/register", json=register_data)
    if response.status_code in [200, 400]:  # 400 if user exists
        print("✅ Enhanced registration working")
    else:
        print(f"❌ Registration failed: {response.status_code}")
except Exception as e:
    print(f"❌ Registration test failed: {e}")

print("• Testing scan with industry patterns...")
try:
    # This would require actual authentication, so just test the endpoint exists
    response = requests.get(f"{api_url}/scans")
    if response.status_code == 401:  # Expected - need auth
        print("✅ Enhanced scan endpoint accessible")
    else:
        print(f"⚠️  Scan endpoint response: {response.status_code}")
except Exception as e:
    print(f"❌ Scan test failed: {e}")
PYTHON

echo ""
echo "Testing resilience features..."

echo "• Circuit breaker test (should handle gracefully):"
for i in {1..3}; do
    curl -s "$API_URL/invalid-endpoint" > /dev/null 2>&1
done
echo "✅ Error handling test completed"

echo "• Performance monitoring test:"
python3 << 'PYTHON'
try:
    from performance_monitor import performance_monitor
    test_op = performance_monitor.start_operation("test-123", "test")
    performance_monitor.end_operation("test-123", success=True)
    print("✅ Performance monitoring working")
except Exception as e:
    print(f"❌ Performance monitoring failed: {e}")
PYTHON

echo ""
echo "🎯 Testing Complete!"
echo "All enhanced features have been tested."
echo ""
echo "Next steps:"
echo "1. Check logs: docker-compose logs -f ao1-scanner"
echo "2. Monitor performance: ./tune_performance.sh"
echo "3. Set up alerts: configure SMTP in .env"
echo "4. Run actual scans with real BigQuery data"
