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

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Scan interrupted by user")
    sys.exit(0)

def main():
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup logging with better formatting
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ao1_scanner.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Check for service account file
    SERVICE_ACCOUNT_FILE = os.path.join(file_path, "gcp_prod_key.json")
    
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        logger.error(f"❌ Service account file not found: {SERVICE_ACCOUNT_FILE}")
        logger.error("Please ensure 'gcp_prod_key.json' is in the same directory as this script")
        sys.exit(1)
    
    try:
        print("🚀 Starting AO1 BigQuery Semantic Scanner v2.0")
        print("=" * 60)
        
        # Test BigQuery connection first
        logger.info("Testing BigQuery connection...")
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        project_id = "prj-fisv-p-gcss-sas-dl9dd0f1df"
        client = bigquery.Client(project=project_id, credentials=credentials)
        
        # Quick connection test
        datasets = list(client.list_datasets())
        logger.info(f"✅ Connected to BigQuery project: {project_id}")
        logger.info(f"✅ Found {len(datasets)} datasets available")
        
        # Import and initialize scanner
        logger.info("Initializing AO1 semantic analyzer...")
        from scanner import AO1Scanner
        
        scanner = AO1Scanner(service_account_path=SERVICE_ACCOUNT_FILE)
        logger.info("✅ Scanner initialized successfully")
        
        # Run scan with conservative settings
        print("\n🔍 Starting comprehensive dataset scan...")
        print("This may take several minutes depending on data size...")
        
        results = scanner.scan_all_datasets(
            max_workers=2,  # Conservative worker count
            quick_scan=False
        )
        
        # Display results
        print("\n" + "="*60)
        print("🎯 AO1 SEMANTIC SCAN COMPLETE")
        print("="*60)
        
        summary = results['summary']
        metadata = results['scan_metadata']
        
        print(f"📊 Project: {project_id}")
        print(f"📊 Datasets analyzed: {metadata['total_datasets']}")
        print(f"📊 Tables analyzed: {metadata['analyzed_tables']}")
        print(f"📊 Execution time: {metadata['execution_time_seconds']}s")
        
        # Discovery metrics
        discovery = summary['discovery_metrics']
        print(f"\n🔍 Semantic Discovery:")
        print(f"   📋 Total fields analyzed: {discovery['total_fields_analyzed']:,}")
        print(f"   ⭐ High-confidence AO1 fields: {discovery['high_confidence_fields']}")
        print(f"   🎯 AO1-relevant tables found: {discovery['ao1_relevant_tables']}")
        
        # AO1 data sources
        ao1_types = summary['ao1_table_types']
        print(f"\n🛡️  AO1 Data Sources Found:")
        print(f"   🖥️  Asset identity tables: {ao1_types['asset_identity_tables']}")
        print(f"   🔒 Security tool tables: {ao1_types['security_tool_tables']}")
        print(f"   📝 Log source tables: {ao1_types['log_source_tables']}")
        
        # Category distribution
        categories = summary['category_distribution']
        if categories:
            print(f"\n📂 Field Category Distribution:")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    category_name = category.replace('_', ' ').title()
                    print(f"   • {category_name}: {count} fields")
        
        # Query performance
        performance = summary['query_performance']
        print(f"\n🔧 Generated Queries:")
        print(f"   📊 Total queries: {performance['total_queries_generated']}")
        if performance['total_queries_generated'] > 0:
            print(f"   ✅ Validation success rate: {performance['query_success_rate']:.1f}%")
            print(f"   ⚡ Average execution time: {performance['average_query_time_ms']:.0f}ms")
        
        # Best queries
        best_queries = scanner.get_best_queries(min_confidence=0.8)
        print(f"\n⭐ High-Confidence Queries (≥0.8): {len(best_queries)}")
        
        if best_queries:
            print(f"\n🎯 Top AO1 Queries for Log Visibility:")
            for i, query in enumerate(best_queries[:5], 1):
                confidence_stars = "★" * min(int(query.confidence_score * 5), 5)
                print(f"   {i}. {query.purpose}")
                print(f"      Confidence: {query.confidence_score:.2f} {confidence_stars}")
                print(f"      Tables: {', '.join(query.source_tables)}")
        
        # Recommendations
        if summary['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # Validation summary
        validation_summary = scanner.get_validation_summary()
        if validation_summary and validation_summary.get('success_rate', 0) > 0:
            print(f"\n📈 Data Quality Assessment:")
            print(f"   ✅ Query success rate: {validation_summary['success_rate']:.1f}%")
            print(f"   📊 Average data quality score: {validation_summary['average_data_quality_score']:.2f}")
        
        # Output files
        print(f"\n📁 Detailed results saved to: {scanner.output_dir}")
        print("📄 Files generated:")
        print("   • ao1_scan_results_*.json - Complete semantic analysis")
        print("   • ao1_queries_*.sql - Generated SQL queries")
        print("   • ao1_summary_*.txt - Human-readable summary")
        
        print("="*60)
        print("✅ Scan completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Scan interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Scan failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()