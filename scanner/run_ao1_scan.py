#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account

file_path = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_FILE = os.path.join(file_path, "gcp_prod_key.json")

sys.path.insert(0, os.path.dirname(__file__))

from scanner import AO1Scanner

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        logger.error(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
        logger.error("Please ensure 'gcp_prod_key.json' is in the same directory as this script")
        sys.exit(1)
    
    try:
        logger.info("Authenticating with BigQuery using exact Flask app pattern...")
        logger.info(f"Using service account file: {SERVICE_ACCOUNT_FILE}")
        
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        project_id = "prj-fisv-p-gcss-sas-dl9dd0f1df"
        client = bigquery.Client(project=project_id, credentials=credentials)
        
        logger.info(f"Connected to BigQuery project: {project_id}")
        
        logger.info("Starting AO1 BigQuery semantic scan...")
        
        scanner = AO1Scanner(service_account_path=SERVICE_ACCOUNT_FILE)
        
        results = scanner.scan_all_datasets(max_workers=4, quick_scan=False)
        
        print("\n" + "="*60)
        print("AO1 SEMANTIC SCAN COMPLETE")
        print("="*60)
        
        summary = results['summary']
        metadata = results['scan_metadata']
        
        print(f"Project: prj-fisv-p-gcss-sas-dl9dd0f1df")
        print(f"Datasets analyzed: {metadata['total_datasets']}")
        print(f"Tables analyzed: {metadata['analyzed_tables']}")
        print(f"Execution time: {metadata['execution_time_seconds']}s")
        
        discovery = summary['discovery_metrics']
        print(f"\nSemantic Discovery:")
        print(f"  Total fields analyzed: {discovery['total_fields_analyzed']:,}")
        print(f"  High-confidence AO1 fields: {discovery['high_confidence_fields']}")
        print(f"  AO1-relevant tables found: {discovery['ao1_relevant_tables']}")
        
        ao1_types = summary['ao1_table_types']
        print(f"\nAO1 Data Sources Found:")
        print(f"  Asset identity tables: {ao1_types['asset_identity_tables']}")
        print(f"  Security tool tables: {ao1_types['security_tool_tables']}")
        print(f"  Log source tables: {ao1_types['log_source_tables']}")
        
        categories = summary['category_distribution']
        if categories:
            print(f"\nField Category Distribution:")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"  {category}: {count} fields")
        
        performance = summary['query_performance']
        print(f"\nGenerated Queries:")
        print(f"  Total queries: {performance['total_queries_generated']}")
        print(f"  Validation success rate: {performance['query_success_rate']:.1f}%")
        print(f"  Average execution time: {performance['average_query_time_ms']:.0f}ms")
        
        best_queries = scanner.get_best_queries(min_confidence=0.8)
        print(f"\nHigh-Confidence Queries (≥0.8): {len(best_queries)}")
        
        if best_queries:
            print(f"\nTop AO1 Queries for Log Visibility:")
            for i, query in enumerate(best_queries[:5], 1):
                confidence_stars = "★" * int(query.confidence_score * 5)
                print(f"  {i}. {query.purpose}")
                print(f"     Confidence: {query.confidence_score:.2f} {confidence_stars}")
                print(f"     Tables: {', '.join(query.source_tables)}")
        
        if summary['recommendations']:
            print(f"\nRecommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        validation_summary = scanner.get_validation_summary()
        if validation_summary and validation_summary.get('success_rate', 0) > 0:
            print(f"\nData Quality Assessment:")
            print(f"  Query success rate: {validation_summary['success_rate']:.1f}%")
            print(f"  Average data quality score: {validation_summary['average_data_quality_score']:.2f}")
        
        print(f"\nDetailed results saved to: {scanner.output_dir}")
        print("Files generated:")
        print("  • ao1_scan_results_*.json - Complete semantic analysis")
        print("  • ao1_queries_*.sql - Generated SQL queries")
        print("  • ao1_summary_*.txt - Human-readable summary")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()