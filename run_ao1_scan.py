import asyncio
import sys
import os
from connection_validator import BigQueryConnectionValidator
from hyperscale_scanner import HyperScaleBigQueryScanner

async def main():
    print("🚀 AO1 Enterprise BigQuery Visibility Scanner")
    print("="*60)
    
    validator = BigQueryConnectionValidator()
    if not validator.validate_complete_setup():
        print("\n❌ Connection validation failed")
        print("\nQuick fixes:")
        print("1. Set GOOGLE_APPLICATION_CREDENTIALS in .env")
        print("2. Ensure service account has required roles:")
        print("   - roles/bigquery.dataViewer")
        print("   - roles/bigquery.jobUser")
        print("   - roles/resourcemanager.projectViewer")
        print("3. Verify project access in Google Cloud Console")
        return 1
    
    print("\n🔍 Starting enterprise-scale AO1 visibility scan...")
    
    scanner = HyperScaleBigQueryScanner()
    
    try:
        results = await scanner.execute_enterprise_scan()
        scanner.save_results(results)
        
        overall_status = results['ao1_assessment']['overall_ao1_readiness']['status']
        
        if overall_status == 'READY':
            print("\n✅ AO1 IMPLEMENTATION READY")
            print("   Sufficient visibility coverage detected across enterprise")
        elif overall_status == 'PARTIAL':
            print("\n⚠️ PARTIAL AO1 READINESS")
            print("   Expand data collection in weak coverage areas")
        else:
            print("\n❌ AO1 NOT READY")
            print("   Insufficient visibility coverage for AO1 implementation")
        
        reliability = results['enterprise_summary']['scan_reliability']
        if reliability['success_rate'] < 80:
            print(f"\n⚠️ Scan reliability: {reliability['success_rate']:.1f}%")
            print("   Some resources had connection issues (see full report)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Enterprise scan failed: {e}")
        print("\nTroubleshooting:")
        print("- Check network connectivity to Google Cloud")
        print("- Verify service account permissions")
        print("- Ensure sufficient BigQuery quota")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
