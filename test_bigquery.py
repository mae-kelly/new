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
