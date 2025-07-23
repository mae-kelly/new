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

def check_environment_variables():
    """Check all required environment variables are set"""
    print("🔍 Checking environment variables...")
    
    required_vars = {
        'Authentication': [
            'FLASK_SECRET_KEY',
            'AUTHORITY', 
            'CLIENT_ID',
            'CLIENT_SECRET',
            'REDIRECT_URI',
            'SCOPE',
            'ENDPOINT'
        ],
        'Chronicle': [
            'CHRONICLE_API_KEY',
            'CHRONICLE_SECRET_KEY', 
            'CHRONICLE_FEED_ID',
            'CHRONICLE_ENDPOINT'
        ],
        'Network': [
            'HTTP_PROXY',
            'HTTPS_PROXY'
        ]
    }
    
    optional_vars = {
        'SSL/TLS': [
            'REQUESTS_CA_BUNDLE',
            'CURL_CA_BUNDLE',
            'SSL_CERT_FILE'
        ],
        'Redis': [
            'REDIS_HOST',
            'REDIS_PORT',
            'REDIS_DB',
            'REDIS_PASSWORD'
        ]
    }
    
    issues = []
    warnings = []
    
    # Check required variables
    for category, vars_list in required_vars.items():
        print(f"\n📋 {category} Configuration:")
        for var in vars_list:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                if any(secret in var.upper() for secret in ['SECRET', 'KEY', 'PASSWORD']):
                    display_value = f"{'*' * min(len(value), 8)}...({len(value)} chars)"
                else:
                    display_value = value[:50] + '...' if len(value) > 50 else value
                print(f"   ✅ {var}: {display_value}")
            else:
                print(f"   ❌ {var}: NOT SET")
                issues.append(f"{var} is required for {category}")
    
    # Check optional variables
    for category, vars_list in optional_vars.items():
        print(f"\n📋 {category} Configuration (Optional):")
        for var in vars_list:
            value = os.getenv(var)
            if value:
                if any(secret in var.upper() for secret in ['SECRET', 'KEY', 'PASSWORD']):
                    display_value = f"{'*' * min(len(value), 8)}...({len(value)} chars)"
                else:
                    display_value = value[:50] + '...' if len(value) > 50 else value
                print(f"   ✅ {var}: {display_value}")
            else:
                print(f"   ⚠️  {var}: NOT SET")
                if category == 'Network' and var in ['HTTP_PROXY', 'HTTPS_PROXY']:
                    warnings.append(f"{var} not set - may be required for corporate network")
    
    # Report status
    if issues:
        print(f"\n❌ Configuration Issues Found:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    
    if warnings:
        print(f"\n⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"   • {warning}")
    
    print(f"\n✅ Environment configuration looks good!")
    return True

def test_corporate_connectivity():
    """Test corporate network connectivity"""
    print("\n🌐 Testing corporate network connectivity...")
    
    import requests
    
    # Create session with environment proxy settings
    session = requests.Session()
    
    http_proxy = os.getenv('HTTP_PROXY')
    https_proxy = os.getenv('HTTPS_PROXY')
    
    if http_proxy or https_proxy:
        session.proxies = {
            'http': http_proxy,
            'https': https_proxy
        }
        print(f"   🔧 Using proxy: {https_proxy or http_proxy}")
    
    # Test Chronicle endpoint if configured
    chronicle_endpoint = os.getenv('CHRONICLE_ENDPOINT')
    if chronicle_endpoint:
        try:
            response = session.head(chronicle_endpoint, timeout=10)
            if response.status_code < 500:
                print(f"   ✅ Chronicle endpoint accessible: {chronicle_endpoint}")
            else:
                print(f"   ⚠️  Chronicle endpoint returned {response.status_code}: {chronicle_endpoint}")
        except Exception as e:
            print(f"   ❌ Chronicle endpoint failed: {str(e)[:60]}...")
    
    # Test model download domains
    test_domains = [
        'https://huggingface.co',
        'https://files.pythonhosted.org'
    ]
    
    for domain in test_domains:
        try:
            response = session.head(domain, timeout=10)
            if response.status_code < 400:
                print(f"   ✅ {domain}: Accessible")
            else:
                print(f"   ⚠️  {domain}: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ {domain}: {str(e)[:60]}...")

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
    
    # Reduce noise from certain modules
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    
    logger = logging.getLogger(__name__)
    
    print("🚀 AO1 BigQuery Semantic Scanner v2.0 - Environment Based")
    print("=" * 70)
    
    # Check environment configuration
    if not check_environment_variables():
        print("\n❌ Environment configuration incomplete")
        print("Please set the required environment variables and try again.")
        print("\n💡 Example setup:")
        print("export HTTP_PROXY=http://10.184.3.109:8080")
        print("export HTTPS_PROXY=http://10.184.3.109:8080") 
        print("export CHRONICLE_API_KEY=your_api_key")
        print("export CLIENT_ID=your_client_id")
        print("# ... (set other required variables)")
        sys.exit(1)
    
    # Test network connectivity
    test_corporate_connectivity()
    
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
        
        # Initialize scanner with environment-based configuration
        print("\n🤖 Initializing AO1 semantic analyzer...")
        print("   📦 Loading AI models with corporate authentication...")
        print("   🔐 Using environment-based configuration...")
        
        from scanner import AO1Scanner
        
        try:
            scanner = AO1Scanner(service_account_path=SERVICE_ACCOUNT_FILE)
            
            # Get configuration summary
            if hasattr(scanner.semantic_analyzer, 'embedding_manager') and \
               hasattr(scanner.semantic_analyzer.embedding_manager, 'foundation_models'):
                config_summary = scanner.semantic_analyzer.embedding_manager.foundation_models.get_configuration_summary()
                
                print("   ✅ Scanner initialized successfully!")
                print(f"   📊 Configuration Summary:")
                print(f"      • Authentication: {'✅' if config_summary['authentication_configured'] else '❌'}")
                print(f"      • Chronicle: {'✅' if config_summary['chronicle_configured'] else '❌'}")
                print(f"      • Proxy: {'✅' if config_summary['proxy_configured'] else '❌'}")
                print(f"      • AI Models: {'✅' if config_summary['models_loaded'] else '❌'}")
            else:
                print("   ✅ Scanner initialized!")
            
        except Exception as e:
            logger.error(f"❌ Scanner initialization failed: {e}")
            print(f"\n🚨 Initialization failed!")
            print(f"   Error: {str(e)[:100]}...")
            print(f"\n🔧 Troubleshooting steps:")
            print(f"   1. Verify all environment variables are set correctly")
            print(f"   2. Test network connectivity to required domains")
            print(f"   3. Check corporate firewall allows AI model downloads")
            print(f"   4. Verify authentication credentials are valid")
            print(f"   5. Check proxy configuration and certificates")
            
            sys.exit(1)
        
        # Run the scan
        print("\n🔍 Starting comprehensive dataset scan...")
        print("📊 Analyzing BigQuery data with corporate authentication...")
        print("⏱️  Estimated time: 3-7 minutes...")
        
        results = scanner.scan_all_datasets(max_workers=2, quick_scan=False)
        
        # Display results
        print("\n" + "="*70)
        print("🎯 AO1 SEMANTIC SCAN COMPLETE")
        print("="*70)
        
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