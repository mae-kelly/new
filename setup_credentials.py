import os
import sys
import json
from pathlib import Path

def setup_bigquery_credentials():
    print("🔧 BigQuery Credentials Setup Assistant")
    print("="*50)
    
    creds_dir = Path("credentials")
    creds_dir.mkdir(exist_ok=True)
    
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        print(f"✅ GOOGLE_APPLICATION_CREDENTIALS already set: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
        
        if Path(os.getenv('GOOGLE_APPLICATION_CREDENTIALS')).exists():
            print("✅ Credentials file exists")
            return True
        else:
            print("❌ Credentials file not found at specified path")
    
    print("\n📋 Required Service Account Roles:")
    print("   roles/bigquery.dataViewer")
    print("   roles/bigquery.jobUser")
    print("   roles/resourcemanager.projectViewer")
    
    print("\n📝 Setup Steps:")
    print("1. Go to Google Cloud Console > IAM & Admin > Service Accounts")
    print("2. Create service account with above roles")
    print("3. Download JSON key file")
    print("4. Save as credentials/bigquery-service-account.json")
    
    expected_path = creds_dir / "bigquery-service-account.json"
    
    if expected_path.exists():
        print(f"\n✅ Found credentials at: {expected_path}")
        
        try:
            with open(expected_path) as f:
                creds = json.load(f)
            
            if all(key in creds for key in ['type', 'project_id', 'private_key', 'client_email']):
                print("✅ Credentials file format valid")
                
                with open('.env', 'r') as f:
                    env_content = f.read()
                
                if 'GOOGLE_APPLICATION_CREDENTIALS' not in env_content:
                    with open('.env', 'a') as f:
                        f.write(f"\nGOOGLE_APPLICATION_CREDENTIALS={expected_path}\n")
                    print("✅ Updated .env file")
                
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(expected_path)
                print("✅ Credentials configured successfully")
                return True
            else:
                print("❌ Invalid credentials file format")
                
        except json.JSONDecodeError:
            print("❌ Credentials file is not valid JSON")
    else:
        print(f"\n❌ Credentials not found at: {expected_path}")
        print("   Download service account JSON key and place it there")
    
    return False

def verify_project_access():
    print("\n🔍 Verifying Project Access...")
    
    try:
        from google.cloud import resourcemanager_v3
        from google.auth import default
        
        credentials, project = default()
        client = resourcemanager_v3.ProjectsClient(credentials=credentials)
        
        request = resourcemanager_v3.ListProjectsRequest()
        projects = list(client.list_projects(request=request))
        active_projects = [p for p in projects if p.state == resourcemanager_v3.Project.State.ACTIVE]
        
        if active_projects:
            print(f"✅ Access to {len(active_projects)} projects:")
            for i, project in enumerate(active_projects[:5]):
                print(f"   {i+1}. {project.project_id}")
            if len(active_projects) > 5:
                print(f"   ... and {len(active_projects)-5} more")
            return True
        else:
            print("❌ No accessible projects found")
            return False
            
    except Exception as e:
        print(f"❌ Project access verification failed: {e}")
        return False

def main():
    if setup_bigquery_credentials():
        if verify_project_access():
            print(f"\n🎉 Setup complete! Ready to run AO1 scanner")
            print("   Run: python run_ao1_scan.py")
            return 0
        else:
            print("\n❌ Project access verification failed")
            print("   Check service account permissions")
            return 1
    else:
        print("\n❌ Credentials setup incomplete")
        return 1

if __name__ == "__main__":
    sys.exit(main())
