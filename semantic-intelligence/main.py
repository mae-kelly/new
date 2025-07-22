#!/usr/bin/env python3

import argparse
import logging
import sys
import os
from datetime import datetime

from engine import IntelligentSemanticEngine

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Semantic Database Analysis Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -d database.sqlite --save-results
  python main.py -d data.duckdb -t 0.9 --output-dir ./results
  python main.py -d database.sqlite --verbose --summary-only
        """
    )
    
    parser.add_argument('-d', '--database', required=True,
                       help='Path to database file (.sqlite or .duckdb)')
    parser.add_argument('-t', '--intelligence-threshold', type=float, default=0.8,
                       help='Intelligence threshold for field selection (default: 0.8)')
    parser.add_argument('--save-results', action='store_true',
                       help='Save analysis results to files')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for saved results (default: current directory)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Show only analysis summary without detailed results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.database):
        print(f"Error: Database file '{args.database}' not found")
        sys.exit(1)
        
    if not (0.0 <= args.intelligence_threshold <= 1.0):
        print("Error: Intelligence threshold must be between 0.0 and 1.0")
        sys.exit(1)
        
    if args.save_results and not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except Exception as e:
            print(f"Error: Could not create output directory '{args.output_dir}': {e}")
            sys.exit(1)
            
    # Setup logging
    setup_logging(args.verbose)
    
    # Welcome message
    print("🧠 Intelligent Semantic Database Analysis Engine")
    print("=" * 50)
    print(f"Database: {args.database}")
    print(f"Intelligence Threshold: {args.intelligence_threshold}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize and run analysis
        engine = IntelligentSemanticEngine(args.database, args.intelligence_threshold)
        
        print("🚀 Starting analysis...")
        results = engine.run_analysis()
        
        if not results.success:
            print(f"❌ Analysis failed: {results.error_message}")
            sys.exit(1)
            
        # Display results
        print("✅ Analysis completed successfully!")
        print()
        
        # Summary
        print("📊 ANALYSIS SUMMARY")
        print("-" * 30)
        print(f"Processing Time: {results.processing_time_seconds:.2f} seconds")
        print(f"Total Fields: {results.total_fields}")
        print(f"Analyzed Fields: {results.analyzed_fields}")
        print(f"High Intelligence Fields: {results.high_intelligence_fields}")
        print(f"Relationships Found: {results.relationships_found}")
        print(f"Semantic Clusters: {results.clusters_created}")
        print(f"Queries Generated: {results.queries_generated}")
        print()
        
        if not args.summary_only:
            # Top fields
            top_fields = engine.get_top_fields(10)
            if top_fields:
                print("🏆 TOP INTELLIGENT FIELDS")
                print("-" * 30)
                for i, field in enumerate(top_fields, 1):
                    domain = field.business_context.get('domain_classification', 'unknown') if field.business_context else 'unknown'
                    print(f"{i:2d}. {field.get_key()}")
                    print(f"    Intelligence: {field.intelligence_score:.3f} | Domain: {domain}")
                    print(f"    Confidence: {field.confidence_level:.3f}")
                print()
                
            # Top queries
            if engine.generated_queries:
                print("⚡ TOP GENERATED QUERIES")
                print("-" * 30)
                for i, query in enumerate(engine.generated_queries[:5], 1):
                    print(f"{i}. {query.name}")
                    print(f"   Description: {query.description}")
                    print(f"   Intelligence: {query.intelligence_score:.3f} | Complexity: {query.complexity_rating}/10")
                    print(f"   Performance: {query.estimated_performance}")
                    print()
                    
            # Domain distribution
            summary = engine.get_analysis_summary()
            if 'top_domains' in summary and summary['top_domains']:
                print("🏷️  DOMAIN DISTRIBUTION")
                print("-" * 30)
                for domain, count in list(summary['top_domains'].items())[:8]:
                    print(f"{domain}: {count} fields")
                print()
                
        # Save results if requested
        if args.save_results:
            print("💾 Saving results...")
            output_files = engine.save_results(args.output_dir)
            
            print("📁 Output Files:")
            for file_path in output_files:
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"   📄 {os.path.basename(file_path)} ({file_size:.1f} KB)")
                
        # Final statistics
        efficiency_score = (results.high_intelligence_fields / max(results.analyzed_fields, 1)) * 100
        
        print()
        print("🎯 FINAL METRICS")
        print("-" * 20)
        print(f"Analysis Efficiency: {efficiency_score:.1f}%")
        print(f"Fields per Second: {results.analyzed_fields / max(results.processing_time_seconds, 0.1):.1f}")
        
        if results.queries_generated > 0:
            avg_query_intelligence = sum(q.intelligence_score for q in engine.generated_queries) / len(engine.generated_queries)
            print(f"Avg Query Intelligence: {avg_query_intelligence:.3f}")
            
        print()
        print("🎉 Analysis complete! Thank you for using the Intelligent Semantic Engine.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()