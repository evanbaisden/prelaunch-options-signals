"""
Pre-Launch Options Signals Analysis Package

A comprehensive framework for analyzing options market signals around product launch events.
Provides both Phase 1 (stock analysis) and Phase 2 (options analysis) capabilities with
professional-grade data integration and statistical analysis.

Modules:
    analysis: Phase 1 stock price analysis around product launches
    phase1: Phase 1 analysis components (baseline, anomalies, signals, outcomes)  
    phase2: Phase 2 options analysis components (data, flow, correlation, backtesting)
    config: Configuration management
    schemas: Data validation
    visualizations: Chart generation
    common: Shared utilities and types
"""

__version__ = "2.0.0"
__author__ = "Pre-Launch Options Signals Research Project"
__description__ = "Options signals analysis around product launch events"

# Core exports
from .config import get_config

__all__ = ["get_config"]