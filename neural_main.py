import argparse
import logging
import json
import os
from datetime import datetime
import duckdb
import pandas as pd
from neural_engine_brilliant import BrilliantVisibilityMapper
from config import QuantumConfig

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/quantum_analysis.log'),
            logging.StreamHandler()
        ]
    )

def load_test_data() -> dict:
    tables = {}
    test_files = [
        'flow_correlation_events.csv',
        'enterprise_inventory_data.csv', 
        'endpoint_security_telemetry.csv'
    ]
    
    for file in test_files:
        filepath = f"test_data/{file}"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            table_name = file.replace('.csv', '')
            tables[table_name] = df
            logging.info(f"Loaded test table: {table_name} with {len(df)} rows")
    
    return tables

def generate_final_report(mappings, mode):
    high_confidence_mappings = [m for m in mappings if m.entanglement_strength > 0.5]
    
    metrics_coverage = {}
    for mapping in high_confidence_mappings:
        metric = mapping.target_metric
        if metric not in metrics_coverage:
            metrics_coverage[metric] = []
        metrics_coverage[metric].append({
            'source': mapping.source_coordinates,
            'table': mapping.table_name,
            'column': mapping.column_name,
            'confidence': float(mapping.entanglement_strength),
            'reasoning': mapping.reasoning_graph
        })
    
    return {
        'analysis_metadata': {
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'total_mappings': len(mappings),
            'high_confidence_mappings': len(high_confidence_mappings),
            'metrics_identified': len(metrics_coverage)
        },
        'visibility_metrics_coverage': metrics_coverage,
        'confidence_distribution': {
            'high_confidence': len([m for m in mappings if m.entanglement_strength > 0.7]),
            'medium_confidence': len([m for m in mappings if 0.4 <= m.entanglement_strength <= 0.7]),
            'low_confidence': len([m for m in mappings if m.entanglement_strength < 0.4])
        },
        'detailed_mappings': [
            {
                'table': m.table_name,
                'column': m.column_name,
                'metric': m.target_metric,
                'confidence': float(m.entanglement_strength),
                'detection_methods': m.reasoning_graph.get('detection_methods', []),
                'sample_values': m.reasoning_graph.get('sample_values', [])
            }
            for m in high_confidence_mappings
        ]
    }

def print_summary(report):
    print("\n" + "="*80)
    print("BRILLIANT QUANTUM VISIBILITY ANALYSIS")
    print("="*80)
    
    print(f"Analysis Mode: {report['analysis_metadata']['mode'].upper()}")
    print(f"Total Mappings Found: {report['analysis_metadata']['total_mappings']}")
    print(f"High Confidence Mappings: {report['analysis_metadata']['high_confidence_mappings']}")
    print(f"Visibility Metrics Identified: {report['analysis_metadata']['metrics_identified']}")
    
    print("\nDISCOVERED VISIBILITY METRICS:")
    for metric, sources in report['visibility_metrics_coverage'].items():
        best_source = max(sources, key=lambda x: x['confidence'])
        print(f"  * {metric.upper().replace('_', ' ')}")
        print(f"    Table: {best_source['table']}")
        print(f"    Column: {best_source['column']}")
        print(f"    Confidence: {best_source['confidence']:.3f}")
        print(f"    Detection: {'+'.join(best_source['reasoning']['detection_methods'])}")
        print(f"    Sample: {best_source['reasoning']['sample_values'][:2]}")
        print()
    
    print("DETAILED TABLE-COLUMN MAPPINGS:")
    for mapping in sorted(report['detailed_mappings'], key=lambda x: x['confidence'], reverse=True)[:10]:
        print(f"  {mapping['table']}.{mapping['column']} -> {mapping['metric']} ({mapping['confidence']:.3f})")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Brilliant Visibility ML Analysis')
    parser.add_argument('--mode', choices=['test', 'prod'], default='test')
    parser.add_argument('--output', default='outputs/analysis_report.json')
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting brilliant analysis in {args.mode} mode")
    
    if args.mode == 'test':
        tables = load_test_data()
    else:
        logger.error("Production mode not implemented in simplified version")
        return
    
    if not tables:
        logger.error("No tables loaded for analysis")
        return
    
    quantum_mapper = BrilliantVisibilityMapper()
    mappings = quantum_mapper.discover_mappings_with_brilliance(tables)
    
    report = generate_final_report(mappings, args.mode)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Analysis complete. Report saved to {args.output}")
    print_summary(report)

if __name__ == "__main__":
    main()
