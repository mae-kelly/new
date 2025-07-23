#!/usr/bin/env python3

import os
import sys
import socket
import ssl
import requests
import subprocess
import json
from pathlib import Path
import time

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def print_section(title):
    print(f"\n📋 {title}")
    print('-'*40)

def test_basic_connectivity():
    print_header("BASIC NETWORK CONNECTIVITY")
    
    # Test basic network
    print_section("DNS Resolution")
    domains_to_test = [
        'google.com',
        'huggingface.co', 
        'pypi.org',
        'nexia.1dc.com',
        'lzf1pvap1560.1dc.com'
    ]
    
    for domain in domains_to_test:
        try:
            ip = socket.gethostbyname(domain)
            print(f"   ✅ {domain} -> {ip}")
        except Exception as e:
            print(f"   ❌ {domain} -> {e}")
    
    # Test basic TCP connectivity
    print_section("TCP Connectivity")
    tcp_tests = [
        ('google.com', 80),
        ('google.com', 443),
        ('10.184.3.109', 8080),  # Your proxy
        ('nexia.1dc.com', 6379)  # Your Redis
    ]
    
    for host, port in tcp_tests:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"   ✅ {host}:{port} - OPEN")
            else:
                print(f"   ❌ {host}:{port} - CLOSED/FILTERED")
        except Exception as e:
            print(f"   ❌ {host}:{port} - {e}")

def test_proxy_configuration():
    print_header("PROXY CONFIGURATION")
    
    # Check environment variables
    print_section("Environment Variables")
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'NO_PROXY', 'no_proxy']
    
    for var in proxy_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {value}")
        else:
            print(f"   ⚠️  {var}: NOT SET")
    
    # Test proxy connectivity
    print_section("Proxy Connectivity Test")
    proxy = os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY')
    
    if proxy:
        try:
            # Test proxy with a simple request
            session = requests.Session()
            session.proxies = {
                'http': proxy,
                'https': proxy
            }
            session.timeout = 10
            
            response = session.get('http://httpbin.org/ip', timeout=10)
            if response.status_code == 200:
                ip_info = response.json()
                print(f"   ✅ Proxy working - External IP: {ip_info.get('origin', 'unknown')}")
            else:
                print(f"   ❌ Proxy returned status: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Proxy test failed: {e}")
            
            # Try direct connection for comparison
            try:
                direct_response = requests.get('http://httpbin.org/ip', timeout=10)
                if direct_response.status_code == 200:
                    print(f"   ℹ️  Direct connection works (proxy may be blocking)")
                else:
                    print(f"   ℹ️  Direct connection also fails")
            except:
                print(f"   ℹ️  Direct connection also blocked")
    else:
        print("   ⚠️  No proxy configured")

def test_ssl_certificates():
    print_header("SSL CERTIFICATE CONFIGURATION")
    
    # Find SSL certificates
    print_section("Certificate Location")
    current_file = Path(__file__)
    
    # Try different possible locations
    possible_ssl_dirs = [
        current_file.parent / 'ssl',
        current_file.parent / '../ssl',
        current_file.parent / '../../ssl', 
        Path.cwd() / 'ssl',
        Path.cwd() / '../ssl'
    ]
    
    ssl_dir = None
    for ssl_path in possible_ssl_dirs:
        if ssl_path.exists():
            ssl_dir = ssl_path
            print(f"   ✅ Found SSL directory: {ssl_dir}")
            break
    
    if not ssl_dir:
        print(f"   ❌ SSL directory not found in any of these locations:")
        for path in possible_ssl_dirs:
            print(f"      - {path}")
        return
    
    # Check certificate files
    cert_file = ssl_dir / 'nexia.1dc.com.crt'
    key_file = ssl_dir / 'nexia.1dc.com.key'
    
    print_section("Certificate Files")
    
    if cert_file.exists():
        print(f"   ✅ Certificate found: {cert_file}")
        
        # Check certificate validity
        try:
            result = subprocess.run(['openssl', 'x509', '-in', str(cert_file), '-text', '-noout'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"   ✅ Certificate is valid")
                
                # Extract certificate info
                cert_info = result.stdout
                if 'nexia.1dc.com' in cert_info:
                    print(f"   ✅ Certificate matches nexia.1dc.com")
                else:
                    print(f"   ⚠️  Certificate may not match nexia.1dc.com")
                    
            else:
                print(f"   ❌ Certificate validation failed: {result.stderr}")
        except FileNotFoundError:
            print(f"   ⚠️  OpenSSL not found - cannot validate certificate")
        except Exception as e:
            print(f"   ⚠️  Certificate validation error: {e}")
    else:
        print(f"   ❌ Certificate not found: {cert_file}")
    
    if key_file.exists():
        print(f"   ✅ Private key found: {key_file}")
        
        # Check key file permissions
        stat_info = key_file.stat()
        if stat_info.st_mode & 0o077:
            print(f"   ⚠️  Key file permissions too open: {oct(stat_info.st_mode)}")
        else:
            print(f"   ✅ Key file permissions OK: {oct(stat_info.st_mode)}")
    else:
        print(f"   ❌ Private key not found: {key_file}")

def test_environment_variables():
    print_header("ENVIRONMENT VARIABLES")
    
    required_vars = {
        'Authentication': ['CLIENT_ID', 'CLIENT_SECRET', 'AUTHORITY'],
        'Chronicle': ['CHRONICLE_API_KEY', 'CHRONICLE_ENDPOINT'],
        'Network': ['HTTP_PROXY', 'HTTPS_PROXY'],
        'Optional': ['FLASK_SECRET_KEY', 'REDIRECT_URI', 'SCOPE']
    }
    
    for category, vars_list in required_vars.items():
        print_section(f"{category} Variables")
        
        for var in vars_list:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                if any(secret in var for secret in ['SECRET', 'KEY', 'PASSWORD']):
                    masked_value = f"{'*' * 8}...({len(value)} chars)"
                else:
                    masked_value = value[:50] + '...' if len(value) > 50 else value
                print(f"   ✅ {var}: {masked_value}")
            else:
                print(f"   ❌ {var}: NOT SET")

def test_bigquery_connectivity():
    print_header("BIGQUERY CONNECTIVITY")
    
    # Check service account file
    print_section("Service Account File")
    service_account_files = [
        'gcp_prod_key.json',
        './gcp_prod_key.json',
        '../gcp_prod_key.json'
    ]
    
    sa_file = None
    for sa_path in service_account_files:
        if os.path.exists(sa_path):
            sa_file = sa_path
            print(f"   ✅ Service account found: {sa_file}")
            break
    
    if not sa_file:
        print(f"   ❌ Service account file not found in:")
        for path in service_account_files:
            print(f"      - {path}")
        return
    
    # Try to load and validate service account
    try:
        with open(sa_file, 'r') as f:
            sa_data = json.load(f)
        
        required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if field not in sa_data]
        
        if missing_fields:
            print(f"   ❌ Service account missing fields: {missing_fields}")
        else:
            print(f"   ✅ Service account format valid")
            print(f"   ✅ Project ID: {sa_data.get('project_id', 'unknown')}")
            print(f"   ✅ Client email: {sa_data.get('client_email', 'unknown')}")
            
    except Exception as e:
        print(f"   ❌ Service account file error: {e}")
        return
    
    # Test BigQuery connection
    print_section("BigQuery Connection Test")
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
        
        credentials = service_account.Credentials.from_service_account_file(sa_file)
        project_id = "prj-fisv-p-gcss-sas-dl9dd0f1df"
        
        # Configure client with proxy if available
        client = bigquery.Client(project=project_id, credentials=credentials)
        
        # Test connection
        datasets = list(client.list_datasets())
        print(f"   ✅ BigQuery connection successful!")
        print(f"   ✅ Found {len(datasets)} datasets")
        
    except ImportError as e:
        print(f"   ❌ Google Cloud libraries not installed: {e}")
    except Exception as e:
        print(f"   ❌ BigQuery connection failed: {e}")

def test_python_imports():
    print_header("PYTHON DEPENDENCIES")
    
    required_modules = [
        'requests', 'numpy', 'torch', 'sklearn', 'google.cloud.bigquery',
        'sentence_transformers', 'transformers'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")

def test_model_download():
    print_header("MODEL DOWNLOAD TEST")
    
    print_section("Testing Model Download URLs")
    
    # Create test session with proxy
    session = requests.Session()
    proxy = os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY')
    
    if proxy:
        session.proxies = {'http': proxy, 'https': proxy}
        print(f"   🔧 Using proxy: {proxy}")
    
    # Configure SSL if certificates exist
    current_file = Path(__file__)
    ssl_dirs = [current_file.parent / 'ssl', current_file.parent / '../ssl']
    
    for ssl_dir in ssl_dirs:
        cert_file = ssl_dir / 'nexia.1dc.com.crt'
        key_file = ssl_dir / 'nexia.1dc.com.key'
        
        if cert_file.exists() and key_file.exists():
            session.cert = (str(cert_file), str(key_file))
            session.verify = str(cert_file)
            print(f"   🔒 Using SSL certificates: {cert_file}")
            break
    
    # Test model download URLs
    test_urls = [
        'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
        'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json',
        'https://files.pythonhosted.org/packages/',
        'https://pypi.org/simple/'
    ]
    
    for url in test_urls:
        try:
            response = session.head(url, timeout=15)
            if response.status_code < 400:
                print(f"   ✅ {url}: {response.status_code}")
            else:
                print(f"   ⚠️  {url}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ {url}: {e}")

def generate_troubleshooting_report():
    print_header("TROUBLESHOOTING RECOMMENDATIONS")
    
    # Analyze results and provide recommendations
    proxy = os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY')
    
    if not proxy:
        print("🔧 SET PROXY CONFIGURATION:")
        print("   export HTTP_PROXY=http://10.184.3.109:8080")
        print("   export HTTPS_PROXY=http://10.184.3.109:8080")
    
    if not os.getenv('CLIENT_ID'):
        print("\n🔧 SET AUTHENTICATION VARIABLES:")
        print("   export CLIENT_ID=your_client_id")
        print("   export CLIENT_SECRET=your_client_secret")
        print("   export CHRONICLE_API_KEY=your_api_key")
    
    print("\n🔧 COMMON SOLUTIONS:")
    print("   1. Verify proxy allows HTTPS CONNECT method")
    print("   2. Check firewall allows access to huggingface.co")
    print("   3. Ensure SSL certificates are valid and readable")
    print("   4. Verify service account has BigQuery permissions")
    print("   5. Test network connectivity from this server")
    
    print("\n🔧 MANUAL TESTS:")
    print("   # Test proxy:")
    print("   curl -x http://10.184.3.109:8080 -v https://huggingface.co")
    print("   ")
    print("   # Test SSL certificates:")
    print("   curl --cert ssl/nexia.1dc.com.crt --key ssl/nexia.1dc.com.key -v https://huggingface.co")
    print("   ")
    print("   # Test BigQuery:")
    print("   gcloud auth activate-service-account --key-file=gcp_prod_key.json")
    print("   gcloud projects list")

def main():
    print("🏢 AO1 Scanner - Corporate Network Diagnostic Tool")
    print("=" * 60)
    print("This tool will diagnose connectivity issues in your corporate environment")
    
    test_basic_connectivity()
    test_proxy_configuration()
    test_ssl_certificates()
    test_environment_variables()
    test_python_imports()
    test_bigquery_connectivity()
    test_model_download()
    generate_troubleshooting_report()
    
    print(f"\n🎯 DIAGNOSTIC COMPLETE")
    print("Review the results above to identify connectivity issues.")

if __name__ == "__main__":
    main()