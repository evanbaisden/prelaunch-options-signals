#!/usr/bin/env python3
"""
Simple Polygon API connection test - just list contracts without volume data
"""
import sys
from pathlib import Path
from datetime import date, timedelta

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.providers.polygon_client import PolygonOptionsClient

def main():
    """Test basic Polygon connection and contract listing"""
    print("=== Simple Polygon Connection Test ===")
    
    try:
        # Initialize client
        client = PolygonOptionsClient()
        print("[OK] Client initialized")
        
        # Test listing contracts only (no volume data)
        ticker = "AAPL"
        test_date = date(2023, 9, 7)
        exp_max = test_date + timedelta(days=30)  # Limit to nearby expiries
        
        print(f"\nTesting contract listing for {ticker} on {test_date}...")
        print(f"Limiting to expiries through {exp_max}")
        
        contracts = client.list_contracts(
            underlying=ticker,
            as_of=test_date,
            exp_min=test_date,
            exp_max=exp_max
        )
        
        print(f"[OK] Retrieved {len(contracts)} contracts")
        
        # Show a few sample contracts
        print("\nSample contracts:")
        for i, contract in enumerate(contracts[:5]):
            print(f"  {i+1}. {contract}")
        
        if len(contracts) > 5:
            print(f"  ... and {len(contracts) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)