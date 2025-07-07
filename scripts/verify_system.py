import sys
import importlib
import subprocess
import requests
def check_dependencies():
    apis = {
        "Solana RPC": "https:
        "Jupiter": "https:
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
