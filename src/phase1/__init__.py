"""
Phase 1 Analysis Package - Modular prelaunch options signals analysis.

This package provides a complete analysis pipeline for studying stock price and volume 
patterns around technology product launches. The analysis is split into focused modules:

- baseline: Calculate statistical baselines and normal trading patterns
- anomalies: Detect volume and price anomalies using statistical methods  
- signals: Extract trading signals around announcement and release events
- outcomes: Compile results and generate reports
- run: Main orchestrator that coordinates all analysis components

Usage:
    from src.phase1 import Phase1Analyzer
    
    analyzer = Phase1Analyzer()
    results = analyzer.run_complete_analysis()
"""

from .run import Phase1Analyzer, main
from .baseline import calculate_baseline_metrics, calculate_period_averages
from .anomalies import detect_volume_spikes, detect_price_anomalies, identify_trading_anomalies
from .signals import extract_announcement_signal, extract_release_signal, compare_signals
from .outcomes import compile_analysis_results, export_results_to_csv, generate_summary_statistics

__version__ = "1.0.0"
__all__ = [
    "Phase1Analyzer",
    "main", 
    "calculate_baseline_metrics",
    "calculate_period_averages",
    "detect_volume_spikes",
    "detect_price_anomalies", 
    "identify_trading_anomalies",
    "extract_announcement_signal",
    "extract_release_signal",
    "compare_signals",
    "compile_analysis_results",
    "export_results_to_csv", 
    "generate_summary_statistics"
]