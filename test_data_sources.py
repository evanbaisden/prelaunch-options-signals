#!/usr/bin/env python3
"""
Quick test script to verify data sources are working.
Run this to check if your system can access the necessary data.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test if basic Python modules work."""
    print("Testing basic imports...")
    try:
        import datetime
        import csv
        print("[OK] Basic Python modules: OK")
        return True
    except Exception as e:
        print(f"[FAIL] Basic imports failed: {e}")
        return False

def test_events_file():
    """Test if events_master.csv exists and is readable."""
    print("Testing events master file...")
    events_file = Path("data/processed/events_master.csv")
    
    if not events_file.exists():
        print(f"[FAIL] Events file not found: {events_file}")
        return False
    
    try:
        with open(events_file, 'r') as f:
            lines = f.readlines()
            print(f"[OK] Events file: {len(lines)-1} events found")
            return True
    except Exception as e:
        print(f"[FAIL] Events file read error: {e}")
        return False

def test_raw_data_directory():
    """Test if raw data directory exists and has CSV files."""
    print("Testing raw data directory...")
    raw_dir = Path("data/raw")
    
    if not raw_dir.exists():
        print(f"[FAIL] Raw data directory not found: {raw_dir}")
        return False
    
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        print(f"[FAIL] No CSV files found in: {raw_dir}")
        return False
    
    print(f"[OK] Raw data directory: {len(csv_files)} CSV files found")
    return True

def test_yahoo_finance():
    """Test Yahoo Finance data access (basic HTTP check)."""
    print("Testing Yahoo Finance access...")
    try:
        import urllib.request
        import ssl
        
        # Create SSL context that allows unverified certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Test basic Yahoo Finance connectivity
        url = "https://finance.yahoo.com"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
            if response.status == 200:
                print("[OK] Yahoo Finance: Accessible")
                return True
            else:
                print(f"[FAIL] Yahoo Finance: HTTP {response.status}")
                return False
                
    except Exception as e:
        print(f"[WARN] Yahoo Finance: Cannot test connectivity ({e})")
        return False

def test_config_file():
    """Test if config/environment setup works."""
    print("Testing configuration...")
    
    env_example = Path(".env.example")
    if not env_example.exists():
        print("[FAIL] .env.example not found")
        return False
    
    env_file = Path(".env")
    if not env_file.exists():
        print("[WARN] .env file not found (should copy from .env.example)")
        # Try to read from .env.example instead
        try:
            with open(env_example) as f:
                content = f.read()
                if "EVENTS_CSV" in content:
                    print("[OK] Configuration template: OK")
                    return True
        except Exception as e:
            print(f"[FAIL] Config test failed: {e}")
            return False
    else:
        print("[OK] Configuration: .env file exists")
        return True

def test_results_directory():
    """Test if results directory is writable."""
    print("Testing results directory...")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Test write access
    test_file = results_dir / "test_write.tmp"
    try:
        test_file.write_text("test")
        test_file.unlink()  # Delete test file
        print("[OK] Results directory: Writable")
        return True
    except Exception as e:
        print(f"[FAIL] Results directory not writable: {e}")
        return False

def main():
    """Run all data source tests."""
    print("=" * 50)
    print("Pre-Launch Options Signals - Data Source Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_events_file,
        test_raw_data_directory,
        test_config_file,
        test_results_directory,
        test_yahoo_finance,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"[SUCCESS] ALL TESTS PASSED ({passed}/{total})")
        print("\n[OK] Your system is ready to run the analysis!")
        print("Next steps:")
        print("1. Copy .env.example to .env if you haven't")
        print("2. Run: python -m src.analysis run-all")
        
    elif passed >= total - 1:
        print(f"[WARN] MOSTLY READY ({passed}/{total})")
        print("\n[OK] Core functionality should work")
        print("[WARN] Some external data sources may have issues")
        
    else:
        print(f"[FAIL] ISSUES FOUND ({passed}/{total})")
        print("\n[WARN] Please fix the failed tests before running analysis")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)