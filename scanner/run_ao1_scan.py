#!/usr/bin/env python3

import os
import sys
import logging
import signal
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account

# Add current directory to path
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_path)

def check_corporate_environment():
    """Check and validate corporate environment setup"""
    print("🏢 Checking corporate environment configuration...")
    
    issues = []
    
    # Check proxy settings
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    
    if http_proxy:
        print(f"   ✅ HTTP_PROXY: {http_proxy}")
    else:
        print("   ⚠️  HTTP_PROXY not set")
        issues.append("Set HTTP_PROXY environment variable")
    
    if https_proxy:
        print(f"   ✅ HTTPS_PROXY: {https_proxy}")
    else:
        print("   ⚠️  HTTPS_PROXY not set")
        issues.append("Set HTTPS_PROXY environment variable")
    
    # Check CA bundle
    ca_bundle = os.environ.get('REQUESTS_CA_BUNDLE')
    if ca_bundle and os.path.exists(ca_bundle):
        print(f"   ✅ CA Bundle: {ca_bundle}")
    else:
        # Try to find system CA bundle
        ca_locations = [
            '/etc/ssl/certs/ca-certificates.crt',
            '/etc/ssl/certs/ca-bundle.crt',
            '/etc/pki/tls/certs/ca-bundle.crt',
            '/usr/local/share/certs/ca-root-nss.crt',
            '/etc/ssl/cert.pem'
        ]
        
        found_ca = None
        for ca_path in ca_locations:
            if os.path.exists(ca_path):
                found_ca = ca_path
                os.environ['REQUESTS_CA_BUNDLE'] = ca_path
                print(f"   ✅ Auto-detected CA Bundle: {ca_path}")
                break
        
        if not found_ca:
            print("   ⚠️  No CA bundle found")
            issues.append("Set REQUESTS_CA_BUNDLE to corporate CA certificate file")
    
    # Test basic connectivity
    print("\n🌐 Testing network connectivity...")
    test_urls = [
        "https://huggingface.co",
        "https://pypi.org"
    ]
    
    import requests
    session = requests.Session()
    session.timeout = 10
    
    # Configure proxy for test
    if http_proxy or https_proxy:
        session.proxies = {
            'http': http_proxy or https_proxy,
            'https': https_proxy or http_proxy
        }
    
    # Configure CA bundle for test
    ca_bundle = os.environ.get('REQUESTS_CA_BUNDLE')
    if ca_bundle:
        session.verify = ca_bundle
    
    connectivity_ok = True
    for url in test_urls:
        try:
            response = session.head(url, timeout=10)
            if response.status_code < 400:
                print(f"   ✅ {url}: OK")
            else:
                print(f"   ⚠️  {url}: HTTP {response.status_code}")
                connectivity_ok = False
        except Exception as e:
            print(f"   ❌ {url}: {str(e)[:50]}...")
            connectivity_ok = False
    
    if issues:
        print(f"\n⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   • {issue}")
        
        print(f"\n🔧 To fix these issues:")
        print(f"   1. Run the corporate setup script:")
        print(f"      bash setup_corporate_network.sh")
        print(f"   2. Or manually set environment variables:")
        print(f"      export HTTP_PROXY=http://proxy.company.com:8080")
        print(f"      export HTTPS_PROXY=http://proxy.company.com:8080")
        print(f"      export REQUESTS_CA_BUNDLE=/path/to/corporate-ca.crt")
        
        return False
    
    if not connectivity_ok:
        print(f"\n🚨 Network connectivity issues detected!")
        print(f"   This will prevent model downloads.")
        print(f"   Contact IT to whitelist these domains:")
        print(f"   • huggingface.co")
        print(f"   • cdn-lfs.huggingface.co") 
        print(f"   • pypi.org")
        print(f"   • files.pythonhosted.org")
        return False
    
    print(f"\n✅ Corporate environment looks good!")
    return True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Scan interrupted by user")
    sys.exit(0)

def main():
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ao1_scanner.log')
        ]
    )
    
    # Reduce noise from network modules
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    
    logger = logging.getLogger(__name__)
    
    print("🚀 AO1 BigQuery Semantic Scanner v2.0 - Corporate Edition")
    print("=" * 65)
    
    # Check corporate environment first
    if not check_corporate_environment():
        print("\n❌ Corporate environment not properly configured")
        print("Please fix the configuration issues above and try again.")
        sys.exit(1)
    
    # Check service account
    SERVICE_ACCOUNT_FILE = os.path.join(file_path, "gcp_prod_key.json")
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        logger.error(f"❌ Service account file not found: {SERVICE_ACCOUNT_FILE}")
        sys.exit(1)
    
    try:
        # Test BigQuery connection
        print("\n🔗 Testing BigQuery connection...")
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        project_id = "prj-fisv-p-gcss-sas-dl9dd0f1df"
        client = bigquery.Client(project=project_id, credentials=credentials)
        
        datasets = list(client.list_datasets())
        print(f"   ✅ Connected to project: {project_id}")
        print(f"   ✅ Found {len(datasets)} datasets")
        
        # Initialize scanner with explicit corporate support
        print("\n🤖 Initializing AO1 semantic analyzer...")
        print("   📦 Downloading AI models (this may take 2-3 minutes)...")
        print("   🌐 Using corporate proxy and certificates...")
        
        from scanner import AO1Scanner
        
        try:
            scanner = AO1Scanner(service_account_path=SERVICE_ACCOUNT_FILE)
            print("   ✅ Scanner initialized successfully with AI models!")
            
        except Exception as e:
            logger.error(f"❌ Scanner initialization failed: {e}")
            print(f"\n🚨 Model download failed!")
            print(f"   Error: {str(e)[:100]}...")
            print(f"\n🔧 Possible solutions:")
            print(f"   1. Check that huggingface.co is accessible from your network")
            print(f"   2. Verify proxy settings are correct")
            print(f"   3. Contact IT to whitelist required domains")
            print(f"   4. Check firewall logs for blocked requests")
            
            # Show exact curl command for testing
            proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
            if proxy:
                print(f"\n🧪 Test connectivity manually:")
                print(f"   curl --proxy {proxy} -v https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
            
            sys.exit(1)
        
        # Run the scan
        print("\n🔍 Starting comprehensive dataset scan...")
        print("📊 Analyzing BigQuery data for AO1 semantic patterns...")
        print("⏱️  Estimated time: 3-7 minutes...")
        
        results = scanner.scan_all_datasets(max_workers=2, quick_scan=False)
        
        # Display results
        print("\n" + "="*65)
        print("🎯 AO1 SEMANTIC SCAN COMPLETE")
        print("="*65)
        
        summary = results['summary']
        metadata = results['scan_metadata']
        
        print(f"📊 Scan Results:")
        print(f"   Project: {project_id}")
        print(f"   Datasets: {metadata['total_datasets']}")
        print(f"   Tables analyzed: {metadata['analyzed_tables']}")
        print(f"   Execution time: {metadata['execution_time_seconds']}s")
        
        discovery = summary['discovery_metrics']
        print(f"\n🔍 Semantic Discovery:")
        print(f"   📋 Fields analyzed: {discovery['total_fields_analyzed']:,}")
        print(f"   ⭐ High-confidence AO1 fields: {discovery['high_confidence_fields']}")
        print(f"   🎯 AO1-relevant tables: {discovery['ao1_relevant_tables']}")
        
        # Show AO1 findings
        ao1_types = summary['ao1_table_types']
        if any(ao1_types.values()):
            print(f"\n🛡️  AO1 Data Sources Found:")
            if ao1_types['asset_identity_tables']:
                print(f"   🖥️  Asset identity: {ao1_types['asset_identity_tables']} tables")
            if ao1_types['security_tool_tables']:
                print(f"   🔒 Security tools: {ao1_types['security_tool_tables']} tables")
            if ao1_types['log_source_tables']:
                print(f"   📝 Log sources: {ao1_types['log_source_tables']} tables")
        
        # Show categories
        categories = summary['category_distribution']
        if categories:
            print(f"\n📂 Field Categories:")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"   • {category.replace('_', ' ').title()}: {count}")
        
        # Show best queries
        performance = summary['query_performance']
        if performance['total_queries_generated'] > 0:
            print(f"\n🔧 Generated Queries:")
            print(f"   📊 Total: {performance['total_queries_generated']}")
            print(f"   ✅ Success rate: {performance['query_success_rate']:.1f}%")
            
            best_queries = scanner.get_best_queries(min_confidence=0.8)
            if best_queries:
                print(f"\n⭐ Top Queries ({len(best_queries)}):")
                for i, query in enumerate(best_queries[:3], 1):
                    stars = "★" * min(int(query.confidence_score * 5), 5)
                    print(f"   {i}. {query.purpose}")
                    print(f"      Confidence: {query.confidence_score:.2f} {stars}")
        
        # Recommendations
        if summary['recommendations']:
            print(f"\n💡 Key Recommendations:")
            for i, rec in enumerate(summary['recommendations'][:3], 1):
                print(f"   {i}. {rec}")
        
        # Output
        print(f"\n📁 Results: {scanner.output_dir}")
        print("📄 Files:")
        print("   • ao1_scan_results_*.json - Complete analysis")
        print("   • ao1_queries_*.sql - SQL queries") 
        print("   • ao1_summary_*.txt - Executive summary")
        
        print("\n✅ AO1 Semantic Scan completed successfully!")
        print("🎯 Review the SQL queries to analyze your AO1 visibility coverage")
        
    except KeyboardInterrupt:
        print("\n🛑 Scan interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Scan failed: {e}")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()