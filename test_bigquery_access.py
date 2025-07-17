#!/usr/bin/env python3
import os
import sys
from google.cloud import bigquery
from google.cloud import resourcemanager_v3

def test_bigquery_access():
    try:
        # Test BigQuery access
        print("🔍 Testing BigQuery access...")
        bq_client = bigquery.Client()
        
        # List projects
        print("📋 Testing project access...")
        resource_client = resourcemanager_v3.ProjectsClient()
        request = resourcemanager_v3.ListProjectsRequest()
        projects = list(resource_client.list_projects(request=request))
        
        active_projects = [p for p in projects if p.state == resourcemanager_v3.Project.State.ACTIVE]
        print(f"✅ Found {len(active_projects)} active projects")
        
        if len(active_projects) > 0:
            # Test accessing first project
            test_project = active_projects[0]
            print(f"🧪 Testing dataset access on project: {test_project.project_id}")
            
            try:
                datasets = list(bq_client.list_datasets(test_project.project_id))
                print(f"✅ Can access {len(datasets)} datasets in {test_project.project_id}")
                
                if len(datasets) > 0:
                    # Test table access
                    test_dataset = datasets[0]
                    tables = list(bq_client.list_tables(test_dataset.reference))
                    print(f"✅ Can access {len(tables)} tables in {test_dataset.dataset_id}")
                    
            except Exception as e:
                print(f"⚠️  Limited access to {test_project.project_id}: {e}")
        
        print("✅ BigQuery access test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ BigQuery access test failed: {e}")
        print("   Make sure GOOGLE_APPLICATION_CREDENTIALS is set correctly")
        return False

if __name__ == "__main__":
    if test_bigquery_access():
        print("\n🚀 Ready to run multi-project BigQuery scanning!")
        print("   Run: python bigquery_existing_ml.py")
    else:
        print("\n❌ Fix BigQuery credentials before proceeding")
        sys.exit(1)
