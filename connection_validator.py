import os
import sys
import json
from google.cloud import bigquery
from google.cloud import resourcemanager_v3
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigQueryConnectionValidator:
    def __init__(self):
        self.bq_client = None
        self.resource_client = None
        self.credentials = None
        self.project_id = None
    
    def validate_complete_setup(self) -> bool:
        print("🔍 Validating BigQuery Connection Setup...")
        
        steps = [
            ("Checking credentials", self._validate_credentials),
            ("Testing BigQuery client", self._validate_bigquery_client),
            ("Testing Resource Manager", self._validate_resource_manager),
            ("Discovering projects", self._validate_project_access),
            ("Testing dataset access", self._validate_dataset_access),
            ("Testing table scanning", self._validate_table_scanning),
            ("Verifying permissions", self._validate_required_permissions)
        ]
        
        for step_name, step_func in steps:
            print(f"  {step_name}...", end=" ")
            try:
                result = step_func()
                if result:
                    print("✅")
                else:
                    print("❌")
                    return False
            except Exception as e:
                print(f"❌ {str(e)}")
                return False
        
        print("\n🎉 All BigQuery connections validated successfully!")
        return True
    
    def _validate_credentials(self) -> bool:
        try:
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if not credentials_path:
                raise Exception("GOOGLE_APPLICATION_CREDENTIALS not set")
            
            if not os.path.exists(credentials_path):
                raise Exception(f"Credentials file not found: {credentials_path}")
            
            self.credentials, self.project_id = default()
            return True
        except DefaultCredentialsError as e:
            raise Exception(f"Invalid credentials: {e}")
        except Exception as e:
            raise Exception(f"Credential validation failed: {e}")
    
    def _validate_bigquery_client(self) -> bool:
        try:
            self.bq_client = bigquery.Client(credentials=self.credentials)
            query = "SELECT 1 as test"
            query_job = self.bq_client.query(query)
            list(query_job.result())
            return True
        except Exception as e:
            raise Exception(f"BigQuery client failed: {e}")
    
    def _validate_resource_manager(self) -> bool:
        try:
            self.resource_client = resourcemanager_v3.ProjectsClient(credentials=self.credentials)
            request = resourcemanager_v3.ListProjectsRequest()
            projects = list(self.resource_client.list_projects(request=request))
            if not projects:
                raise Exception("No projects accessible")
            return True
        except Exception as e:
            raise Exception(f"Resource Manager failed: {e}")
    
    def _validate_project_access(self) -> bool:
        try:
            request = resourcemanager_v3.ListProjectsRequest()
            projects = list(self.resource_client.list_projects(request=request))
            active_projects = [p for p in projects if p.state == resourcemanager_v3.Project.State.ACTIVE]
            
            if len(active_projects) == 0:
                raise Exception("No active projects found")
            
            print(f"\n    📋 Found {len(active_projects)} active projects:")
            for i, project in enumerate(active_projects[:5]):
                print(f"      {i+1}. {project.project_id}")
            if len(active_projects) > 5:
                print(f"      ... and {len(active_projects)-5} more")
            
            return True
        except Exception as e:
            raise Exception(f"Project discovery failed: {e}")
    
    def _validate_dataset_access(self) -> bool:
        try:
            request = resourcemanager_v3.ListProjectsRequest()
            projects = list(self.resource_client.list_projects(request=request))
            active_projects = [p for p in projects if p.state == resourcemanager_v3.Project.State.ACTIVE]
            
            datasets_found = 0
            for project in active_projects[:3]:
                try:
                    datasets = list(self.bq_client.list_datasets(project.project_id))
                    datasets_found += len(datasets)
                except Exception:
                    continue
            
            if datasets_found == 0:
                raise Exception("No datasets accessible in any project")
            
            print(f"\n    📊 Found {datasets_found} total datasets")
            return True
        except Exception as e:
            raise Exception(f"Dataset access failed: {e}")
    
    def _validate_table_scanning(self) -> bool:
        try:
            request = resourcemanager_v3.ListProjectsRequest()
            projects = list(self.resource_client.list_projects(request=request))
            active_projects = [p for p in projects if p.state == resourcemanager_v3.Project.State.ACTIVE]
            
            scannable_tables = 0
            
            for project in active_projects[:2]:
                try:
                    datasets = list(self.bq_client.list_datasets(project.project_id))
                    for dataset in datasets[:3]:
                        try:
                            tables = list(self.bq_client.list_tables(dataset.reference))
                            
                            for table in tables[:1]:
                                if table.table_type == 'TABLE':
                                    try:
                                        query = f"SELECT * FROM `{project.project_id}.{dataset.dataset_id}.{table.table_id}` LIMIT 1"
                                        job_config = bigquery.QueryJobConfig()
                                        job_config.maximum_bytes_billed = 1024 * 1024
                                        job_config.job_timeout_ms = 30000
                                        
                                        query_job = self.bq_client.query(query, job_config=job_config)
                                        list(query_job.result())
                                        scannable_tables += 1
                                    except Exception:
                                        continue
                        except Exception:
                            continue
                except Exception:
                    continue
            
            if scannable_tables == 0:
                raise Exception("No tables can be scanned")
            
            print(f"\n    🗂️ Found {scannable_tables} scannable tables")
            return True
        except Exception as e:
            raise Exception(f"Table scanning failed: {e}")
    
    def _validate_required_permissions(self) -> bool:
        try:
            request = resourcemanager_v3.ListProjectsRequest()
            projects = list(self.resource_client.list_projects(request=request))
            test_project = projects[0].project_id if projects else None
            
            if not test_project:
                raise Exception("No test project available")
            
            print(f"\n    🔐 Testing permissions on {test_project}")
            
            datasets = list(self.bq_client.list_datasets(test_project))
            
            return True
        except Exception as e:
            raise Exception(f"Permission validation failed: {e}")

def main():
    validator = BigQueryConnectionValidator()
    success = validator.validate_complete_setup()
    
    if success:
        print("\n🚀 Ready to run AO1 BigQuery scanner!")
        print("   All connections and permissions verified")
        return 0
    else:
        print("\n❌ Fix connection issues before proceeding")
        print("\nTroubleshooting:")
        print("1. Ensure service account has these roles:")
        print("   - roles/bigquery.dataViewer")
        print("   - roles/bigquery.jobUser") 
        print("   - roles/resourcemanager.projectViewer")
        print("2. Download JSON key and set GOOGLE_APPLICATION_CREDENTIALS")
        print("3. Verify project access in Google Cloud Console")
        return 1

if __name__ == "__main__":
    sys.exit(main())
