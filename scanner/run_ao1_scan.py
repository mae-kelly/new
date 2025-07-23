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

def check_corporate_proxy():
    """Check and display corporate proxy configuration"""
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    ca_vars = ['REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'SSL_CERT_FILE']
    
    print("🔍 Checking corporate environment configuration...")
    
    # Check proxy settings
    proxy_found = False
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"   ✅ {var}: {value}")
            proxy_found = True
    
    if not proxy_found:
        print("   ⚠️  No proxy environment variables found")
        print("      If you're behind a corporate proxy, you may need to set:")
        print("      export HTTP_PROXY=http://proxy.company.com:8080")
        print("      export HTTPS_PROXY=http://proxy.company.com:8080")
    
    # Check CA bundle settings
    ca_found = False
    for var in ca_vars:
        value = os.environ.get(var)
        if value:
            if os.path.exists(value):
                print(f"   ✅ {var}: {value}")
                ca_found = True
            else:
                print(f"   ⚠️  {var}: {value} (file not found)")
    
    if not ca_found:
        # Look for common CA bundle locations
        ca_paths = [
            '/etc/ssl/certs/ca-certificates.crt',  # Ubuntu/Debian
            '/etc/ssl/certs/ca-bundle.crt',        # CentOS/RHEL
            '/etc/pki/tls/certs/ca-bundle.crt',    # Fedora/CentOS
            '/usr/local/share/certs/ca-root-nss.crt',  # FreeBSD
            '/etc/ssl/cert.pem'                     # macOS
        ]
        
        found_ca = None
        for path in ca_paths:
            if os.path.exists(path):
                found_ca = path
                print(f"   ✅ Found system CA bundle: {path}")
                break
        
        if not found_ca:
            print("   ⚠️  No CA bundle found. SSL verification may fail.")
            print("      If you have corporate certificates, set:")
            print("      export REQUESTS_CA_BUNDLE=/path/to/corporate-ca-bundle.crt")
    
    return proxy_found or ca_found

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Scan interrupted by user")
    sys.exit(0)

def main():
    # Setup signal handlers for graceful shutdown
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
    
    # Reduce noise from SSL/proxy related modules
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    
    print("🚀 AO1 BigQuery Semantic Scanner v2.0 - Corporate Edition")
    print("=" * 60)
    
    # Check corporate environment
    corporate_env = check_corporate_proxy()
    
    # Check for service account file
    SERVICE_ACCOUNT_FILE = os.path.join(file_path, "gcp_prod_key.json")
    
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        logger.error(f"❌ Service account file not found: {SERVICE_ACCOUNT_FILE}")
        logger.error("Please ensure 'gcp_prod_key.json' is in the same directory as this script")
        sys.exit(1)
    
    try:
        # Test BigQuery connection first
        print("\n🔗 Testing BigQuery connection...")
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        project_id = "prj-fisv-p-gcss-sas-dl9dd0f1df"
        client = bigquery.Client(project=project_id, credentials=credentials)
        
        # Quick connection test
        datasets = list(client.list_datasets())
        logger.info(f"✅ Connected to BigQuery project: {project_id}")
        logger.info(f"✅ Found {len(datasets)} datasets available")
        print(f"   Project: {project_id}")
        print(f"   Datasets: {len(datasets)}")
        
        # Import and initialize scanner
        print("\n🤖 Initializing AO1 semantic analyzer...")
        print("   📦 Loading NLP models (this may take 1-2 minutes)...")
        
        if corporate_env:
            print("   🏢 Using corporate proxy/certificate settings")
        
        from scanner import AO1Scanner
        
        scanner = AO1Scanner(service_account_path=SERVICE_ACCOUNT_FILE)
        logger.info("✅ Scanner initialized successfully")
        
        # Check if we're in fallback mode
        try:
            if (hasattr(scanner.semantic_analyzer, 'embedding_manager') and 
                hasattr(scanner.semantic_analyzer.embedding_manager, 'foundation_models') and 
                scanner.semantic_analyzer.embedding_manager.foundation_models.is_fallback_mode()):
                print("   ⚠️  Running in fallback mode (simple embeddings)")
                print("      This is OK - analysis will still work effectively")
            else:
                print("   ✅ Full AI models loaded successfully")
        except:
            print("   ✅ Semantic analyzer ready")
        
        # Run scan
        print("\n🔍 Starting comprehensive dataset scan...")
        print("📊 Analyzing table schemas and data for AO1 semantic patterns")
        print("⏱️  Estimated time: 2-5 minutes...")
        print("\n" + "-" * 60)
        
        results = scanner.scan_all_datasets(
            max_workers=2,  # Conservative for corporate environments
            quick_scan=False
        )
        
        # Display results
        print("\n" + "="*60)
        print("🎯 AO1 SEMANTIC SCAN COMPLETE")
        print("="*60)
        
        summary = results['summary']
        metadata = results['scan_metadata']
        
        print(f"📊 Scan Summary:")
        print(f"   Project: {project_id}")
        print(f"   Datasets scanned: {metadata['total_datasets']}")
        print(f"   Tables with semantic data: {metadata['analyzed_tables']}")
        print(f"   Total execution time: {metadata['execution_time_seconds']}s")
        
        # Discovery metrics
        discovery = summary['discovery_metrics']
        print(f"\n🔍 Discovery Results:")
        print(f"   📋 Fields analyzed: {discovery['total_fields_analyzed']:,}")
        print(f"   ⭐ High-confidence AO1 fields: {discovery['high_confidence_fields']}")
        print(f"   🎯 AO1-relevant tables: {discovery['ao1_relevant_tables']}")
        
        # Check if we found any data
        if discovery['total_fields_analyzed'] == 0:
            print("\n⚠️  Analysis Status:")
            print("   No fields were successfully analyzed.")
            print("   Possible causes:")
            print("   • Tables contain no sample data")
            print("   • Permission restrictions on data access")
            print("   • All tables are views or external references")
            print("   • Network connectivity issues")
            print("\n💡 Recommendations:")
            print("   • Check BigQuery permissions for data access")
            print("   • Verify tables contain actual data (not just schema)")
            print("   • Try running with --quick-scan flag")
            
        else:
            # AO1 data sources found
            ao1_types = summary['ao1_table_types']
            if any(ao1_types.values()):
                print(f"\n🛡️  AO1 Data Sources Discovered:")
                if ao1_types['asset_identity_tables'] > 0:
                    print(f"   🖥️  Asset identity tables: {ao1_types['asset_identity_tables']}")
                if ao1_types['security_tool_tables'] > 0:
                    print(f"   🔒 Security tool tables: {ao1_types['security_tool_tables']}")
                if ao1_types['log_source_tables'] > 0:
                    print(f"   📝 Log source tables: {ao1_types['log_source_tables']}")
            
            # Category distribution
            categories = summary['category_distribution']
            if categories:
                print(f"\n📂 Field Categories Found:")
                sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
                for category, count in sorted_categories:
                    if count > 0:
                        category_name = category.replace('_', ' ').title()
                        print(f"   • {category_name}: {count} fields")
            
            # Query performance
            performance = summary['query_performance']
            if performance['total_queries_generated'] > 0:
                print(f"\n🔧 Generated Queries:")
                print(f"   📊 Total queries: {performance['total_queries_generated']}")
                print(f"   ✅ Validation success: {performance['query_success_rate']:.1f}%")
                print(f"   ⚡ Avg execution time: {performance['average_query_time_ms']:.0f}ms")
                
                # Best queries
                best_queries = scanner.get_best_queries(min_confidence=0.8)
                if best_queries:
                    print(f"\n⭐ High-Confidence Queries ({len(best_queries)}):")
                    for i, query in enumerate(best_queries[:3], 1):
                        confidence_stars = "★" * min(int(query.confidence_score * 5), 5)
                        print(f"   {i}. {query.purpose}")
                        print(f"      Confidence: {query.confidence_score:.2f} {confidence_stars}")
        
        # Recommendations
        if summary['recommendations']:
            print(f"\n💡 Key Recommendations:")
            for i, rec in enumerate(summary['recommendations'][:5], 1):
                print(f"   {i}. {rec}")
        
        # Output files
        print(f"\n📁 Results saved to: {scanner.output_dir}")
        print("📄 Generated files:")
        print("   • ao1_scan_results_*.json - Complete analysis data")
        print("   • ao1_queries_*.sql - Ready-to-run SQL queries") 
        print("   • ao1_summary_*.txt - Executive summary")
        print("   • ao1_scanner.log - Detailed execution log")
        
        print("\n" + "="*60)
        print("✅ AO1 Semantic Scan completed successfully!")
        print("🎯 Use the generated SQL queries to analyze your AO1 visibility gaps")
        
    except KeyboardInterrupt:
        print("\n🛑 Scan interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Scan failed: {e}")
        print(f"\n❌ Error occurred: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Check network connectivity")
        print("   2. Verify BigQuery permissions")
        print("   3. Check corporate proxy settings")
        print("   4. Review the log file: ao1_scanner.log")
        
        import traceback
        logger.debug("Full traceback:")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()