#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

from .scanner import AO1Scanner

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ao1_scanner.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="AO1 BigQuery Semantic Scanner")
    parser.add_argument('--service-account', '-s', type=str, help="Path to service account JSON file")
    parser.add_argument('--output-dir', '-o', type=str, help="Output directory for results")
    parser.add_argument('--quick-scan', '-q', action='store_true', help="Quick scan mode (fewer samples)")
    parser.add_argument('--max-workers', '-w', type=int, default=4, help="Maximum concurrent workers")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose logging")
    parser.add_argument('--min-confidence', '-c', type=float, default=0.8, help="Minimum confidence for best queries")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing AO1 Scanner")
        scanner = AO1Scanner(
            service_account_path=args.service_account,
            output_dir=args.output_dir
        )
        
        logger.info("Starting comprehensive dataset scan")
        results = scanner.scan_all_datasets(
            max_workers=args.max_workers,
            quick_scan=args.quick_scan
        )
        
        print("\n" + "="*60)
        print("AO1 SEMANTIC SCAN COMPLETE")
        print("="*60)
        
        summary = results['summary']
        print(f"Datasets analyzed: {results['scan_metadata']['total_datasets']}")
        print(f"Tables analyzed: {results['scan_metadata']['analyzed_tables']}")
        print(f"Execution time: {results['scan_metadata']['execution_time_seconds']}s")
        
        print(f"\nAO1-Relevant discoveries:")
        print(f"  Asset identity tables: {summary['ao1_table_types']['asset_identity_tables']}")
        print(f"  Security tool tables: {summary['ao1_table_types']['security_tool_tables']}")
        print(f"  Log source tables: {summary['ao1_table_types']['log_source_tables']}")
        
        print(f"\nQuery generation:")
        print(f"  Total queries: {summary['query_performance']['total_queries_generated']}")
        print(f"  Success rate: {summary['query_performance']['query_success_rate']:.1f}%")
        
        best_queries = scanner.get_best_queries(args.min_confidence)
        print(f"\nHigh-confidence queries (≥{args.min_confidence}): {len(best_queries)}")
        
        if best_queries:
            print("\nTop AO1 queries:")
            for i, query in enumerate(best_queries[:5], 1):
                print(f"  {i}. {query.purpose} (confidence: {query.confidence_score:.2f})")
        
        if summary['recommendations']:
            print(f"\nRecommendations:")
            for rec in summary['recommendations'][:3]:
                print(f"  • {rec}")
        
        validation_summary = scanner.get_validation_summary()
        if validation_summary:
            print(f"\nQuery validation:")
            print(f"  Success rate: {validation_summary['success_rate']:.1f}%")
            print(f"  Avg execution time: {validation_summary['average_execution_time_ms']:.0f}ms")
            print(f"  Data quality score: {validation_summary['average_data_quality_score']:.2f}")
        
        print(f"\nResults saved to: {scanner.output_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()