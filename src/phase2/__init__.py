"""
Phase 2: Options Analysis Module

Comprehensive options market analysis around product launch events.
Includes multi-provider data collection, flow analysis, statistical testing,
and trading strategy backtesting.

Key Components:
    options_data: Multi-provider options data collection (Alpha Vantage, Polygon, Yahoo)
    flow_analysis: Options flow and unusual activity detection
    earnings_data: Earnings correlation analysis
    correlation_analysis: Statistical correlation testing
    regression_framework: Predictive modeling
    backtesting: Trading strategy evaluation
    run: Main Phase 2 execution pipeline

Data Providers:
    - Alpha Vantage: Historical options data 2020-2024 (Primary)
    - Polygon: Recent options data validation  
    - Yahoo Finance: Backup options data

Usage:
    from src.phase2 import OptionsDataManager
    
    manager = OptionsDataManager()
    options_data = manager.cross_validate_options_data('AAPL', date(2020, 10, 13))

Status: ✅ PHASE 2 IMPLEMENTED AND WORKING
"""

__version__ = "2.0.0"
__status__ = "Production Ready"

from .options_data import OptionsDataManager, AlphaVantageOptionsProvider
from .flow_analysis import OptionsFlowAnalyzer
from .earnings_data import EarningsAnalyzer

__all__ = [
    "OptionsDataManager", 
    "AlphaVantageOptionsProvider",
    "OptionsFlowAnalyzer", 
    "EarningsAnalyzer"
]