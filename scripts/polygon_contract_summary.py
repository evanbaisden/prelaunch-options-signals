#!/usr/bin/env python3
"""
Polygon contract summary - works with free tier limitations
"""
import sys
from pathlib import Path
from datetime import date, timedelta

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.providers.polygon_client import PolygonOptionsClient

def main():
    """Create options contract summary without detailed volume data"""
    print("=== Polygon Contract Summary (Free Tier Compatible) ===")
    
    try:
        client = PolygonOptionsClient()
        print("[OK] Client initialized")
        
        # Test around iPhone 15 announcement
        ticker = "AAPL"
        announcement_date = date(2023, 9, 12)
        
        # Test a few days around the announcement
        test_dates = [
            announcement_date - timedelta(days=3),
            announcement_date - timedelta(days=1), 
            announcement_date,
            announcement_date + timedelta(days=1),
            announcement_date + timedelta(days=3)
        ]
        
        print(f"\nAnalyzing {ticker} options around iPhone 15 announcement...")
        print(f"Announcement date: {announcement_date}")
        
        for test_date in test_dates:
            print(f"\n--- {test_date} ---")
            
            # Limit to nearby expiries to reduce contract count
            exp_min = test_date
            exp_max = test_date + timedelta(days=21)  # ~3 weeks out
            
            contracts = client.list_contracts(
                underlying=ticker,
                as_of=test_date,
                exp_min=exp_min,
                exp_max=exp_max
            )
            
            # Analyze contract mix without volume data
            call_count = sum(1 for c in contracts if c.get("type", "").upper().startswith("C"))
            put_count = len(contracts) - call_count
            
            # Group by expiration
            exp_dates = {}
            for c in contracts:
                exp = c.get("expiration", "unknown")
                if exp not in exp_dates:
                    exp_dates[exp] = {"calls": 0, "puts": 0}
                
                if c.get("type", "").upper().startswith("C"):
                    exp_dates[exp]["calls"] += 1
                else:
                    exp_dates[exp]["puts"] += 1
            
            print(f"  Total contracts: {len(contracts)}")
            print(f"  Calls: {call_count}, Puts: {put_count}")
            print(f"  Expiration dates: {len(exp_dates)}")
            
            # Show top 3 expiry dates by contract count
            top_exp = sorted(exp_dates.items(), 
                           key=lambda x: x[1]["calls"] + x[1]["puts"], 
                           reverse=True)[:3]
            
            for exp_date, counts in top_exp:
                total = counts["calls"] + counts["puts"]
                print(f"    {exp_date}: {total} contracts ({counts['calls']}C/{counts['puts']}P)")
        
        print("\n[SUCCESS] Contract analysis completed without volume data")
        print("Note: Volume data requires premium tier due to rate limits")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)