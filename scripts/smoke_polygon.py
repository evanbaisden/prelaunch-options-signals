#!/usr/bin/env python3
"""
Smoke test for Polygon.io options data integration
"""
import sys
from pathlib import Path
from datetime import date

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase2.options_pipeline import OptionsPipeline

def main():
    """Run a basic smoke test of the options pipeline"""
    print("=== Polygon Options Pipeline Smoke Test ===")
    
    try:
        # Initialize pipeline
        print("Initializing options pipeline...")
        pipe = OptionsPipeline()
        print("[OK] Pipeline initialized successfully")
        
        # Test with Apple on a recent date (iPhone 15 announcement period)
        test_ticker = "AAPL"
        test_date = date(2023, 9, 7)  # A few days before iPhone 15 announcement
        
        print(f"\nTesting chain snapshot for {test_ticker} on {test_date}...")
        
        stats = pipe.chain_snapshot_for_event_day(test_ticker, test_date)
        
        print("[OK] Chain snapshot retrieved successfully")
        print("\nResults:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during smoke test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)