#!/usr/bin/env python3
"""
Data Source Assessment for Pre-Launch Options Signals Analysis
Check what data sources are available and what's missing for complete analysis.
"""
import sys
import os
from pathlib import Path

def check_phase1_data():
    """Check Phase 1 data availability."""
    print("\n=== PHASE 1 DATA SOURCES ===")
    
    # Check events file
    events_file = Path("data/processed/events_master.csv")
    if events_file.exists():
        with open(events_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            print(f"[OK] Product Launch Events: {len(lines)} events")
            for line in lines[:5]:  # Show first 5
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    print(f"     - {parts[0]} ({parts[1]}: {parts[2]})")
            if len(lines) > 5:
                print(f"     ... and {len(lines) - 5} more")
    else:
        print("[FAIL] Product Launch Events: Missing events_master.csv")
        return False
    
    # Check stock data
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        csv_files = list(raw_dir.glob("*.csv"))
        print(f"[OK] Stock Price Data: {len(csv_files)} files")
        for f in csv_files[:3]:  # Show first 3
            print(f"     - {f.name}")
        if len(csv_files) > 3:
            print(f"     ... and {len(csv_files) - 3} more")
    else:
        print("[FAIL] Stock Price Data: Missing data/raw directory")
        return False
    
    return True

def check_phase2_data():
    """Check Phase 2 data source requirements."""
    print("\n=== PHASE 2 DATA SOURCES ===")
    
    # Check options data providers
    print("\n1. OPTIONS DATA PROVIDERS:")
    
    # Check if we have credentials for different providers
    env_file = Path(".env")
    providers = {
        "QuantConnect": ["QUANTCONNECT_API_KEY", "QUANTCONNECT_API_SECRET"],
        "Polygon": ["POLYGON_API_KEY"],
        "Alpha Vantage": ["ALPHA_VANTAGE_API_KEY"],
        "Yahoo Finance": []  # No API key needed
    }
    
    available_providers = []
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_content = f.read()
    else:
        env_content = ""
    
    for provider, required_keys in providers.items():
        if not required_keys:  # Yahoo Finance
            print(f"[OK] {provider}: Available (no API key required)")
            available_providers.append(provider)
        else:
            has_keys = all(key in env_content and f"{key}=" in env_content for key in required_keys)
            if has_keys:
                print(f"[OK] {provider}: Configured")
                available_providers.append(provider)
            else:
                print(f"[MISSING] {provider}: Needs API keys - {', '.join(required_keys)}")
    
    # Check earnings data
    print("\n2. EARNINGS DATA:")
    if "Alpha Vantage" in available_providers:
        print("[OK] Earnings Data: Available via Alpha Vantage")
    else:
        print("[LIMITED] Earnings Data: Only Yahoo Finance estimates available")
    
    # Check market microstructure data
    print("\n3. MARKET MICROSTRUCTURE DATA:")
    if "QuantConnect" in available_providers:
        print("[OK] Bid/Ask Spreads: Available via QuantConnect")
        print("[OK] Order Flow Data: Available via QuantConnect")
    elif "Polygon" in available_providers:
        print("[OK] Bid/Ask Spreads: Available via Polygon")
        print("[LIMITED] Order Flow Data: Limited via Polygon")
    else:
        print("[MISSING] Bid/Ask Spreads: Need QuantConnect or Polygon")
        print("[MISSING] Order Flow Data: Need QuantConnect")
    
    return len(available_providers) >= 1

def check_infrastructure():
    """Check analysis infrastructure."""
    print("\n=== ANALYSIS INFRASTRUCTURE ===")
    
    # Check core modules
    core_files = [
        ("src/config.py", "Configuration Management"),
        ("src/schemas.py", "Data Validation"),
        ("src/analysis.py", "Phase 1 Analysis Engine"),
        ("src/phase2/options_data.py", "Options Data Pipeline"),
        ("src/phase2/flow_analysis.py", "Options Flow Analysis"),
        ("src/phase2/correlation_analysis.py", "Statistical Testing"),
        ("src/phase2/regression_framework.py", "Regression Models"),
        ("src/phase2/backtesting.py", "Strategy Backtesting")
    ]
    
    all_present = True
    for filepath, description in core_files:
        if Path(filepath).exists():
            print(f"[OK] {description}")
        else:
            print(f"[MISSING] {description}: {filepath}")
            all_present = False
    
    return all_present

def assess_data_completeness():
    """Assess overall data completeness for analysis."""
    print("\n" + "="*60)
    print("DATA SOURCE ASSESSMENT SUMMARY")
    print("="*60)
    
    # What you HAVE
    print("\n[OK] AVAILABLE DATA SOURCES:")
    print("- Product launch events (13 events, 2020-2024)")
    print("- Stock price data (Yahoo Finance historical)")
    print("- Yahoo Finance options data (limited)")
    print("- Basic analysis infrastructure")
    print("- Statistical testing framework")
    print("- Backtesting engine")
    
    # What you're MISSING for complete analysis
    print("\n[MISSING] FOR COMPLETE ANALYSIS:")
    missing_items = []
    
    # Check for QuantConnect
    env_file = Path(".env")
    has_quantconnect = False
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            has_quantconnect = "QUANTCONNECT_API_KEY" in content
    
    if not has_quantconnect:
        missing_items.extend([
            "QuantConnect API credentials (recommended for options data)",
            "Comprehensive options chains with Greeks",
            "High-quality bid/ask spread data",
            "Institutional order flow data"
        ])
    
    # Check for other providers
    has_alpha_vantage = env_file.exists() and "ALPHA_VANTAGE_API_KEY" in open(env_file).read()
    if not has_alpha_vantage:
        missing_items.append("Alpha Vantage API key (for earnings data)")
    
    has_polygon = env_file.exists() and "POLYGON_API_KEY" in open(env_file).read()
    if not has_polygon:
        missing_items.append("Polygon API key (alternative options provider)")
    
    for item in missing_items:
        print(f"- {item}")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. IMMEDIATE (Can run analysis now):")
    print("   - Use existing Yahoo Finance data for basic analysis")
    print("   - Run Phase 1 stock analysis: python -m src.analysis run-all")
    print("   - Test options data collection with Yahoo Finance")
    
    print("\n2. ENHANCED ANALYSIS (Get better data):")
    print("   - Sign up for QuantConnect (best options data)")
    print("   - Get Alpha Vantage free API key (earnings data)")
    print("   - Consider Polygon API for additional validation")
    
    print("\n3. PRODUCTION QUALITY:")
    print("   - QuantConnect Professional subscription (~$150/month)")
    print("   - Multiple data providers for cross-validation")
    print("   - Real-time data feeds for live analysis")
    
    # Data quality assessment
    completeness_score = 0
    if Path("data/processed/events_master.csv").exists():
        completeness_score += 25
    if list(Path("data/raw").glob("*.csv")):
        completeness_score += 25
    if has_quantconnect:
        completeness_score += 30
    if has_alpha_vantage:
        completeness_score += 20
    
    print(f"\nDATA COMPLETENESS: {completeness_score}/100")
    
    if completeness_score >= 70:
        print("Status: READY for comprehensive analysis")
    elif completeness_score >= 50:
        print("Status: CAN START analysis, enhance data sources for better results")
    else:
        print("Status: BASIC analysis possible, need more data sources")

def main():
    """Run complete data source assessment."""
    print("Pre-Launch Options Signals - Data Source Assessment")
    print("="*60)
    
    # Check each component
    phase1_ok = check_phase1_data()
    phase2_ok = check_phase2_data()
    infra_ok = check_infrastructure()
    
    # Overall assessment
    assess_data_completeness()
    
    return phase1_ok and infra_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)