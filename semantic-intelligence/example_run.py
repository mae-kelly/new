#!/usr/bin/env python3

from database_connector import DatabaseConnector
from advanced_semantic_analyzer import AdvancedSemanticAnalyzer
from advanced_ao1_engine_part1 import AdvancedAO1Engine
from advanced_ao1_engine_part2 import AdvancedAO1EngineExecutor

def run_ao1_analysis(database_path: str):
    print("🧠 Starting Advanced AO1 Analysis...")
    
    # 1. Connect to database
    db_connector = DatabaseConnector(database_path)
    if not db_connector.connect():
        print("❌ Failed to connect to database")
        return
        
    print("✅ Connected to database")
    
    # 2. Discover schema
    schema = db_connector.discover_schema()
    print(f"📊 Discovered {sum(len(cols) for cols in schema.values())} fields in {len(schema)} tables")
    
    # 3. Analyze fields with advanced semantics
    analyzer = AdvancedSemanticAnalyzer()
    intelligent_fields = []
    
    field_count = 0
    for table_name, columns in schema.items():
        for column_name, data_type in columns:
            if field_count >= 100:  # Limit for demo
                break
                
            try:
                sample_values = db_connector.sample_field_data(table_name, column_name, 1000)
                field_intelligence = analyzer.analyze_field_deeply(
                    column_name, table_name, data_type, sample_values
                )
                
                if field_intelligence.intelligence_score > 0.5:
                    intelligent_fields.append(field_intelligence)
                    
                field_count += 1
                
                if field_count % 20 == 0:
                    print(f"   Analyzed {field_count} fields...")
                    
            except Exception as e:
                print(f"   Warning: Failed to analyze {table_name}.{column_name}: {e}")
                
        if field_count >= 100:
            break
    
    print(f"✅ Found {len(intelligent_fields)} high-intelligence fields")
    
    # 4. Generate AO1 Dashboard
    ao1_engine = AdvancedAO1Engine(db_connector)
    
    # Combine with part 2 functionality
    executor = AdvancedAO1EngineExecutor(ao1_engine)
    
    # Monkey patch the methods we need
    ao1_engine._build_intelligent_global_visibility = executor._build_intelligent_global_visibility
    ao1_engine._build_intelligent_platform_coverage = executor._build_intelligent_platform_coverage
    ao1_engine._build_intelligent_infrastructure_visibility = executor._build_intelligent_infrastructure_visibility
    ao1_engine._build_intelligent_silent_assets = executor._build_intelligent_silent_assets
    ao1_engine._build_intelligent_log_role_coverage = executor._build_intelligent_log_role_coverage
    ao1_engine._extract_visibility_percentage = executor._extract_visibility_percentage
    ao1_engine._calculate_dashboard_semantic_coherence = executor._calculate_dashboard_semantic_coherence
    
    print("🎯 Generating AO1 Dashboard...")
    dashboard = ao1_engine.generate_intelligent_ao1_dashboard(intelligent_fields)
    
    # 5. Display Results
    print("\n🎯 AO1 DASHBOARD RESULTS")
    print("=" * 50)
    print(f"Success Rate: {dashboard.success_rate:.1f}%")
    print(f"Semantic Coherence: {dashboard.semantic_coherence:.2f}")
    print()
    
    if dashboard.global_visibility_score:
        gv = dashboard.global_visibility_score
        visibility_pct = gv.extracted_values.get('visibility_percentage', 0)
        total_assets = gv.extracted_values.get('total_assets', 0)
        
        print(f"📊 Global Visibility: {visibility_pct:.1f}%")
        print(f"   Total Assets: {total_assets:,}")
        print(f"   Assessment: {gv.business_assessment}")
        print(f"   Confidence: {gv.validation_confidence:.2f}")
        
    if dashboard.platform_coverage:
        pc = dashboard.platform_coverage
        platforms = len(pc.extracted_values.get('platform_names', []))
        print(f"🔧 Platform Coverage: {platforms} platforms detected")
        
    if dashboard.silent_assets:
        sa = dashboard.silent_assets
        silent_count = sa.extracted_values.get('total_silent', 0)
        print(f"🔇 Silent Assets: {silent_count:,} assets with zero logging")
        
    print()
    print("🎉 Analysis Complete!")
    
    # 6. Save Queries (optional)
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if dashboard.global_visibility_score:
        with open(f'ao1_global_visibility_{timestamp}.sql', 'w') as f:
            f.write("-- AO1 Global Visibility Query\n")
            f.write(f"-- Generated: {datetime.now()}\n")
            f.write(f"-- Confidence: {dashboard.global_visibility_score.validation_confidence:.2f}\n")
            f.write(dashboard.global_visibility_score.query.sql)
        print(f"💾 Saved global visibility query to ao1_global_visibility_{timestamp}.sql")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python example_run.py <database_path>")
        print("Example: python example_run.py sample_data.sqlite")
        sys.exit(1)
        
    database_path = sys.argv[1]
    run_ao1_analysis(database_path)