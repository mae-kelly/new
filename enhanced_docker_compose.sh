#!/bin/bash

echo "Creating enhanced Docker Compose with all integrations..."

cat > docker-compose.enhanced.yml << 'YAML'
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
      - ENABLE_SIEM_INTEGRATION=true
      - ENABLE_GRAPHQL_QUERIES=true
      - ENABLE_CHRONICLE_EXPORT=true
      - ENABLE_REAL_TIME_STREAMING=true
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./logs:/app/logs
      - ./credentials:/app/credentials
      - ./results:/app/results
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/detailed"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    depends_on:
      - redis
    networks:
      - ao1-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 5s
      retries: 3
    restart: unless-stopped
    networks:
      - ao1-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus-config.yaml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=15d'
    networks:
      - ao1-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana-dashboard.json:/etc/grafana/provisioning/dashboards/ao1-dashboard.json
    networks:
      - ao1-network

volumes:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  ao1-network:
    driver: bridge
YAML

echo "✅ Enhanced Docker Compose created: docker-compose.enhanced.yml"

cat > monitoring/prometheus-config.yaml << 'YAML'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ao1-scanner'
    static_configs:
      - targets: ['ao1-scanner:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: []
YAML

cat > monitoring/alert_rules.yml << 'YAML'
groups:
  - name: ao1_scanner_alerts
    rules:
      - alert: HighScanFailureRate
        expr: rate(ao1_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High scan failure rate detected"
          description: "AO1 Scanner error rate is {{ $value }} errors per second"

      - alert: SlowScanPerformance
        expr: rate(ao1_scan_duration_seconds_sum[5m]) / rate(ao1_scan_duration_seconds_count[5m]) > 300
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow scan performance"
          description: "Average scan duration is {{ $value }} seconds"

      - alert: BigQueryConnectionDown
        expr: up{job="ao1-scanner"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "AO1 Scanner is down"
          description: "AO1 Scanner has been down for more than 1 minute"
YAML

echo "✅ Monitoring configuration created"
