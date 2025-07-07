#!/usr/bin/env python3
import sys
import importlib
import subprocess
import requests

def check_dependencies():
    """Verify all dependencies are installed"""
    required_packages = [
        'solana', 'web3', 'torch', 'numpy', 'pandas',
        'sklearn', 'requests', 'aiohttp', 'websockets'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    return len(missing) == 0

def check_apis():
    """Test API connectivity"""
    apis = {
        "Solana RPC": "https://api.mainnet-beta.solana.com",
        "Jupiter": "https://quote-api.jup.ag/v6/tokens",
    }
    
    all_good = True
    for name, url in apis.items():
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {name} API")
            else:
                print(f"⚠️  {name} API (status: {response.status_code})")
                all_good = False
        except:
            print(f"❌ {name} API")
            all_good = False
    
    return all_good

def main():
    print("🔍 Verifying system setup...")
    
    deps_ok = check_dependencies()
    apis_ok = check_apis()
    
    if deps_ok and apis_ok:
        print("\n✅ System verification passed!")
        return True
    else:
        print("\n❌ System verification failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
