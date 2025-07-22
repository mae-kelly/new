#!/usr/bin/env python3

import argparse
import logging
import sys
import os
from datetime import datetime

from engine import IntelligentSemanticEngine
from ao1_main_engine import AO1Engine

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Semantic Database Analysis Engine - AO1 Log Visibility Focus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -d database.sqlite --ao1-dashboard
  python main.py -d data.duckdb -t 0.9 --ao1-only --output-dir ./results
  python main.py -d database.sqlite --verbose --ao1-dashboard --save-results
        """
    )
    
    parser.add_argument('-d', '--database', required=True,
                       help='Path to database file (.sqlite or .duckdb)')
    parser.add_argument('-t', '--intelligence-threshold', type=float, default=0.8,
                       help='Intelligence threshold for field selection (default: 0.8)')
    parser.add_argument('--ao1-dashboard', action='store_true',
                       help='Generate AO1 Log Visibility Dashboard (recommended)')
    parser.add_argument('--ao1-only', action='store_true',
                       help='Only run AO1 analysis, skip general semantic analysis')
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
    print("🎯 AO1 Log Visibility Measurement Focus")
    print("=" * 60)
    print(f"Database: {args.database}")
    print(f"Intelligence Threshold: {args.intelligence_threshold}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.ao1_dashboard or args.ao1_only:
        print("🎯 AO1 Dashboard Mode: Enabled")
    print()
    
    try:
        # Initialize semantic engine
        engine = IntelligentSemanticEngine(args.database, args.intelligence_threshold)
        
        print("🚀 Starting semantic field analysis...")
        results = engine.run_analysis()
        
        if not results.success:
            print(f"❌ Semantic analysis failed: {results.error_message}")
            sys.exit(1)
            
        print("✅ Semantic analysis completed!")
        print()
        
        # Display semantic analysis summary
        print("📊 SEMANTIC ANALYSIS SUMMARY")
        print("-" * 40)
        print(f"Processing Time: {results.processing_time_seconds:.2f} seconds")
        print(f"Total Fields: {results.total_fields}")
        print(f"Analyzed Fields: {results.analyzed_fields}")
        print(f"High Intelligence Fields: {results.high_intelligence_fields}")
        print(f"Relationships Found: {results.relationships_found}")
        print(f"Semantic Clusters: {results.clusters_created}")
        print()
        
        # Get high intelligence fields for AO1 analysis
        high_intelligence_fields = engine.get_top_fields(50)  # Get top 50 for AO1
        
        # AO1 Dashboard Generation
        if args.ao1_dashboard or args.ao1_only:
            print("🎯 GENERATING AO1 LOG VISIBILITY DASHBOARD")
            print("=" * 50)
            
            # Initialize AO1 engine
            ao1_engine = AO1Engine(engine.database_connector)
            
            # Generate AO1 dashboard
            print("📊 Building AO1 visibility metrics...")
            dashboard = ao1_engine.generate_ao1_dashboard(high_intelligence_fields)
            
            if not dashboard:
                print("❌ Could not generate AO1 dashboard - insufficient field intelligence")
            else:
                print(f"✅ Generated {len(dashboard)} AO1 metrics")
                print()
                
                # Display AO1 results
                print("🎯 AO1 VISIBILITY METRICS")
                print("-" * 30)
                
                for metric_name, metric in dashboard.items():
                    print(f"📊 {metric.name}")
                    print(f"   Assessment: {metric.validation.business_assessment}")
                    print(f"   Confidence: {metric.validation.confidence_score:.2f}")
                    print(f"   Business Value: {metric.business_value}")
                    
                    if metric.results:
                        # Show first result row for key metrics
                        if metric_name == 'global_visibility':
                            row = metric.results[0]
                            visibility_pct = next((v for v in row if isinstance(v, (int, float)) and 0 <= v <= 100), 0)
                            total_assets = next((v for v in row if isinstance(v, int) and v > 100), 0)
                            print(f"   🎯 VISIBILITY: {visibility_pct:.1f}% of {total_assets:,} assets")
                            
                        elif metric_name == 'silent_assets':
                            silent_count = len(metric.results)
                            print(f"   🔇 SILENT ASSETS: {silent_count:,} assets with no logging")
                            
                        elif metric_name == 'platform_coverage':
                            platform_count = len(metric.results)
                            print(f"   🔧 PLATFORMS: Coverage across {platform_count} logging systems")
                    
                    print()
                
                # Generate executive report
                print("📋 EXECUTIVE SUMMARY")
                print("-" * 20)
                executive_report = ao1_engine.generate_executive_report(dashboard)
                
                print(f"Overall Status: {executive_report['executive_summary']['overall_status']}")
                print(f"Data Confidence: {executive_report['executive_summary']['confidence_level']:.1%}")
                print(f"Recommendation: {executive_report['overall_recommendation']}")
                print()
                
                # Key findings
                if executive_report['key_findings']:
                    print("Key Findings:")
                    for finding in executive_report['key_findings']:
                        print(f"  • {finding}")
                    print()
                
                # Critical actions
                if executive_report['recommended_actions']:
                    print("⚠️  Action Items:")
                    for action in executive_report['recommended_actions'][:5]:  # Top 5
                        print(f"  • {action}")
                    print()
                
                # Save AO1 SQL queries
                if args.save_results:
                    sql_file = ao1_engine.save_sql_queries(dashboard, 
                                                          f"{args.output_dir}/ao1_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql")
                    print(f"💾 Saved working AO1 queries to {sql_file}")
        
        # Regular semantic analysis results (if not AO1-only)
        if not args.ao1_only and not args.summary_only:
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
        
        # Save general results if requested
        if args.save_results and not args.ao1_only:
            print("💾 Saving semantic analysis results...")
            output_files = engine.save_results(args.output_dir)
            
            print("📁 Output Files:")
            for file_path in output_files:
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"   📄 {os.path.basename(file_path)} ({file_size:.1f} KB)")
        
        # Final statistics
        if args.ao1_dashboard or args.ao1_only:
            ao1_success = len(dashboard) if 'dashboard' in locals() else 0
            print()
            print("🎯 AO1 FINAL METRICS")
            print("-" * 25)
            print(f"AO1 Metrics Generated: {ao1_success}/4")
            print(f"Semantic Fields Used: {len(high_intelligence_fields)}")
            
            if ao1_success > 0:
                avg_confidence = sum(m.validation.confidence_score for m in dashboard.values()) / len(dashboard)
                print(f"Average Confidence: {avg_confidence:.1%}")
        
        efficiency_score = (results.high_intelligence_fields / max(results.analyzed_fields, 1)) * 100
        print()
        print("🎯 FINAL ANALYSIS METRICS")
        print("-" * 30)
        print(f"Semantic Analysis Efficiency: {efficiency_score:.1f}%")
        print(f"Fields per Second: {results.analyzed_fields / max(results.processing_time_seconds, 0.1):.1f}")
        
        print()
        print("🎉 Analysis complete! Your AO1 visibility metrics are ready.")
        
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