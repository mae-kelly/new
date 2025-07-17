#!/bin/bash

set -e

echo "Setting up AO1 Scanner Production Environment..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env file from template"
fi

mkdir -p credentials outputs data results logs k8s monitoring

cat > k8s/namespace.yaml << 'YAML'
apiVersion: v1
kind: Namespace
metadata:
  name: ao1-scanner
  labels:
    name: ao1-scanner
YAML

cat > k8s/configmap.yaml << 'YAML'
apiVersion: v1
kind: ConfigMap
metadata:
  name: ao1-scanner-config
  namespace: ao1-scanner
data:
  MAX_PROJECTS_PER_SCAN: "50"
  MAX_TABLES_PER_PROJECT: "200"
  SCAN_TIMEOUT_MINUTES: "60"
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
YAML

cat > k8s/secret.yaml << 'YAML'
apiVersion: v1
kind: Secret
metadata:
  name: ao1-scanner-secrets
  namespace: ao1-scanner
type: Opaque
data:
  jwt-secret: ""
  gcp-credentials: ""
YAML

cat > k8s/deployment.yaml << 'YAML'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ao1-scanner
  namespace: ao1-scanner
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ao1-scanner
  template:
    metadata:
      labels:
        app: ao1-scanner
    spec:
      containers:
      - name: ao1-scanner
        image: ao1-scanner:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: PORT
          value: "8000"
        - name: HOST
          value: "0.0.0.0"
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: ao1-scanner-secrets
              key: jwt-secret
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: "/app/credentials/service-account.json"
        envFrom:
        - configMapRef:
            name: ao1-scanner-config
        volumeMounts:
        - name: gcp-credentials
          mountPath: /app/credentials
          readOnly: true
        - name: data-volume
          mountPath: /app/data
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: gcp-credentials
        secret:
          secretName: ao1-scanner-secrets
          items:
          - key: gcp-credentials
            path: service-account.json
      - name: data-volume
        persistentVolumeClaim:
          claimName: ao1-scanner-data
YAML

cat > k8s/service.yaml << 'YAML'
apiVersion: v1
kind: Service
metadata:
  name: ao1-scanner-service
  namespace: ao1-scanner
spec:
  selector:
    app: ao1-scanner
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
YAML

cat > k8s/pvc.yaml << 'YAML'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ao1-scanner-data
  namespace: ao1-scanner
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
YAML

cat > k8s/ingress.yaml << 'YAML'
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ao1-scanner-ingress
  namespace: ao1-scanner
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: ao1-scanner.company.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ao1-scanner-service
            port:
              number: 80
YAML

cat > monitoring/prometheus-config.yaml << 'YAML'
global:
  scrape_interval: 15s
scrape_configs:
- job_name: 'ao1-scanner'
  static_configs:
  - targets: ['ao1-scanner-service:80']
  metrics_path: /metrics
  scrape_interval: 30s
YAML

cat > monitoring/grafana-dashboard.json << 'JSON'
{
  "dashboard": {
    "title": "AO1 Scanner Performance",
    "panels": [
      {
        "title": "Active Scans",
        "type": "stat",
        "targets": [{"expr": "ao1_active_scans"}]
      },
      {
        "title": "Scan Duration",
        "type": "graph",
        "targets": [{"expr": "rate(ao1_scan_duration_seconds_sum[5m])"}]
      },
      {
        "title": "Detection Rate",
        "type": "graph", 
        "targets": [{"expr": "rate(ao1_detections_total[5m])"}]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [{"expr": "rate(ao1_errors_total[5m])"}]
      }
    ]
  }
}
JSON

cat > docker-compose.prod.yml << 'YAML'
version: '3.8'
services:
  ao1-scanner:
    build: .
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
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus-config.yaml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
YAML

cat > Dockerfile.prod << 'DOCKERFILE'
FROM python:3.11-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN groupadd -r scanner && useradd -r -g scanner scanner
RUN chown -R scanner:scanner /app
USER scanner

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run_server.py"]
DOCKERFILE

echo "Created production deployment files"
echo "Next steps:"
echo "1. Update k8s/secret.yaml with base64 encoded secrets"
echo "2. Build image: docker build -f Dockerfile.prod -t ao1-scanner:latest ."
echo "3. Deploy: kubectl apply -f k8s/"
echo "4. Or use Docker Compose: docker-compose -f docker-compose.prod.yml up -d"
