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
