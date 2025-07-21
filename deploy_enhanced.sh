#!/bin/bash

set -e

echo "🚀 Deploying Enhanced AO1 Scanner with Full Integration Stack"
echo "============================================================"

DEPLOYMENT_TYPE=${1:-docker}

if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
    echo "Step 1: Creating monitoring directories..."
    mkdir -p monitoring data outputs logs credentials results
    
    echo "Step 2: Setting up enhanced Docker Compose..."
    ./enhanced_docker_compose.sh
    
    echo "Step 3: Validating credentials..."
    if [ -f "validate_credentials.py" ]; then
        python3 validate_credentials.py || {
            echo "⚠️  Credential validation failed, but continuing deployment..."
        }
    fi
    
    echo "Step 4: Starting enhanced stack..."
    docker-compose -f docker-compose.enhanced.yml up -d
    
    echo "Step 5: Waiting for services..."
    sleep 20
    
    echo "Step 6: Health checks..."
    echo "• AO1 Scanner:"
    for i in {1..10}; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            echo "  ✅ AO1 Scanner healthy"
            break
        else
            echo "  ⏳ Waiting... (attempt $i/10)"
            sleep 3
        fi
    done
    
    echo "• Redis:"
    if docker-compose -f docker-compose.enhanced.yml exec -T redis redis-cli ping | grep -q PONG; then
        echo "  ✅ Redis healthy"
    else
        echo "  ⚠️  Redis connection issues"
    fi
    
    echo "• Prometheus:"
    if curl -f http://localhost:9090/-/healthy >/dev/null 2>&1; then
        echo "  ✅ Prometheus healthy"
    else
        echo "  ⚠️  Prometheus connection issues"
    fi
    
    echo "• Grafana:"
    if curl -f http://localhost:3000/api/health >/dev/null 2>&1; then
        echo "  ✅ Grafana healthy"
    else
        echo "  ⚠️  Grafana connection issues"
    fi
    
    echo ""
    echo "🎉 Enhanced AO1 Scanner Stack Deployed!"
    echo ""
    echo "📊 Access Points:"
    echo "• AO1 Scanner API: http://localhost:8000"
    echo "• AO1 Dashboard: http://localhost:8000/dashboard/stats"
    echo "• Prometheus: http://localhost:9090"
    echo "• Grafana: http://localhost:3000 (admin/admin)"
    echo "• Redis: localhost:6379"
    echo ""
    echo "🔧 Management:"
    echo "• Logs: docker-compose -f docker-compose.enhanced.yml logs -f"
    echo "• Stop: docker-compose -f docker-compose.enhanced.yml down"
    echo "• Restart: docker-compose -f docker-compose.enhanced.yml restart"
    
else
    echo "Kubernetes deployment with full integration stack..."
    # Kubernetes deployment would go here
    echo "Kubernetes deployment not implemented yet. Use: $0 docker"
fi
