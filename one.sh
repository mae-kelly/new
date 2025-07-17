#!/bin/bash
set -e

echo "Testing BigQuery connection..."

source .env

cat > test_bigquery.py << 'EOF'
import os
from google.cloud import bigquery
from google.cloud import resourcemanager_v3

def test_connection():
    try:
        bq_client = bigquery.Client()
        resource_client = resourcemanager_v3.ProjectsClient()
        
        request = resourcemanager_v3.ListProjectsRequest()
        projects = list(resource_client.list_projects(request=request))
        active_projects = [p for p in projects if p.state == resourcemanager_v3.Project.State.ACTIVE]
        
        print(f"✅ Found {len(active_projects)} active projects:")
        for i, project in enumerate(active_projects[:5]):
            print(f"   {i+1}. {project.project_id}")
        
        if len(active_projects) > 5:
            print(f"   ... and {len(active_projects)-5} more")
        
        if len(active_projects) > 0:
            test_project = active_projects[0]
            datasets = list(bq_client.list_datasets(test_project.project_id))
            print(f"✅ Found {len(datasets)} datasets in {test_project.project_id}")
        
        print("✅ BigQuery connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
EOF

python test_bigquery.py

if [ $? -eq 0 ]; then
    echo "Ready to scan BigQuery projects"
else
    echo "Fix connection before proceeding"
    exit 1
fi

#!/bin/bash
set -e

echo "Starting BigQuery multi-project scanner..."

source .env

echo "This will scan ALL accessible BigQuery projects for visibility data"
echo "Including hosts, networks, security systems, platforms, locations, etc."
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo "🚀 Starting scan across all projects..."
echo "This may take several minutes..."
echo ""

python bigquery_existing_ml.py

echo ""
echo "🎉 Scan completed!"
echo "📁 Results saved to: outputs/bigquery_existing_ml_scan.json"

if [ -f "outputs/bigquery_existing_ml_scan.json" ]; then
    echo ""
    echo "📊 Quick summary:"
    python -c "
import json
with open('outputs/bigquery_existing_ml_scan.json', 'r') as f:
    data = json.load(f)
meta = data.get('scan_metadata', {})
print(f'Projects scanned: {meta.get(\"projects_scanned\", 0)}')
print(f'Tables scanned: {meta.get(\"tables_scanned\", 0)}') 
print(f'Total mappings found: {meta.get(\"total_mappings\", 0)}')
print(f'High confidence: {meta.get(\"high_confidence_mappings\", 0)}')
"
fi