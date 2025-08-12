#!/usr/bin/env python3
"""
Quick test of Phase 2 options analysis infrastructure.
"""
import pandas as pd
from datetime import date, datetime
from src.phase2.options_data import OptionsDataManager, AlphaVantageOptionsProvider
from src.config import get_config

def test_basic_phase2():
    """Test basic Phase 2 functionality."""
    print("Testing Phase 2 Options Analysis Infrastructure")
    print("=" * 60)
    
    config = get_config()
    
    # Test 1: Options Data Manager
    print("\\n1. Testing OptionsDataManager initialization...")
    try:
        manager = OptionsDataManager(config)
        print(f"✅ Initialized with providers: {list(manager.providers.keys())}")
        print(f"   Provider priority: {manager.provider_order}")
    except Exception as e:
        print(f"❌ Failed to initialize OptionsDataManager: {e}")
        return False
    
    # Test 2: Alpha Vantage Provider  
    print("\\n2. Testing Alpha Vantage options data collection...")
    try:
        # Use iPhone 12 launch as test case
        test_ticker = 'AAPL'
        test_date = date(2020, 10, 13)  # iPhone 12 announcement
        
        print(f"   Collecting options data for {test_ticker} on {test_date}")
        print("   (This may take 10-15 seconds due to API rate limiting...)")
        
        # Get options data with cross-validation
        validation_results = manager.cross_validate_options_data(
            ticker=test_ticker,
            date=test_date,
            max_providers=1  # Just Alpha Vantage for now
        )
        
        if validation_results:
            provider_name = list(validation_results.keys())[0]
            df = validation_results[provider_name]
            
            print(f"✅ Collected {len(df)} contracts from {provider_name}")
            print(f"   Calls: {len(df[df['option_type'] == 'call'])}")
            print(f"   Puts: {len(df[df['option_type'] == 'put'])}")
            
            # Check Greeks availability
            greek_cols = ['delta', 'gamma', 'theta', 'vega']
            greeks_summary = {}
            for col in greek_cols:
                if col in df.columns:
                    available = df[col].notna().sum()
                    pct = available / len(df) * 100
                    greeks_summary[col] = f"{available}/{len(df)} ({pct:.0f}%)"
            
            print("   Greeks availability:", greeks_summary)
            
            # Show sample contract
            print("\\n   Sample contract:")
            sample = df.iloc[0]
            for col in ['strike', 'option_type', 'expiration', 'last_price', 'volume', 'delta']:
                if col in df.columns:
                    print(f"     {col}: {sample[col]}")
            
            return True
        else:
            print("❌ No options data collected")
            return False
            
    except Exception as e:
        print(f"❌ Options data collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_events_processing():
    """Test processing of events data."""
    print("\\n3. Testing events data processing...")
    
    try:
        config = get_config()
        
        # Load events
        events_path = config.events_csv
        df = pd.read_csv(events_path)
        print(f"✅ Loaded {len(df)} events from events file")
        
        # Show event breakdown
        event_counts = df['company'].value_counts()
        print("   Events by company:")
        for company, count in event_counts.items():
            print(f"     {company}: {count}")
        
        # Show date range
        df['announcement'] = pd.to_datetime(df['announcement']).dt.date
        min_date = df['announcement'].min()
        max_date = df['announcement'].max()
        print(f"   Date range: {min_date} to {max_date}")
        
        return True
        
    except Exception as e:
        print(f"❌ Events processing failed: {e}")
        return False

def main():
    """Run Phase 2 infrastructure tests."""
    print("Pre-Launch Options Signals - Phase 2 Infrastructure Test")
    print("=" * 70)
    
    tests = [
        test_basic_phase2,
        test_events_processing
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\\n✅ Phase 2 infrastructure is ready!")
        print("\\nNext steps:")
        print("1. Run comprehensive analysis: python -m src.phase2.run")
        print("2. Review results in results/phase2_run_*/ directory")
        
    elif passed >= total - 1:
        print(f"⚠️  MOSTLY READY ({passed}/{total})")
        print("\\n✅ Core Phase 2 functionality works")
        print("❗ Some components may need adjustment")
        
    else:
        print(f"❌ ISSUES FOUND ({passed}/{total})")
        print("\\n❗ Fix failed tests before running full analysis")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)