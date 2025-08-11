#!/usr/bin/env python3
"""
Phase 1: Volume Information Leakage Analysis - Driver Script
Runs the complete analysis pipeline for all events
"""
import os
import sys

# Add src to Python path for imports
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "src"))

from phase1.section1_setup import VolumeInformationLeakageAnalyzer
from phase1.section7_main import run_multi_event_analysis, generate_cross_event_summary
from phase1.section8_save import save_analysis_results, create_summary_visualization

def main():
    """Run the complete Phase 1 analysis pipeline"""
    print("=== Phase 1: Volume Information Leakage Analysis ===")
    print("Initializing analyzer...")
    
    # Create analyzer with default parameters
    analyzer = VolumeInformationLeakageAnalyzer()
    
    print(f"Analyzing {len(analyzer.events)} events...")
    
    # Run multi-event analysis
    all_results = run_multi_event_analysis(analyzer)
    
    # Generate cross-event summary
    summary_stats = generate_cross_event_summary(all_results)
    
    # Save results
    if all_results:
        print("\nSaving analysis results...")
        saved = save_analysis_results(all_results, summary_stats, analyzer)
        
        # Create visualizations
        if saved and saved.get("summary_dataframe") is not None:
            print("Creating summary visualization...")
            create_summary_visualization(saved["summary_dataframe"])
        
        print(f"\nAnalysis complete! Results saved in:")
        print(f"  - data/processed/")
        print(f"  - results/")
    else:
        print("No results to save - analysis may have failed.")

if __name__ == "__main__":
    main()