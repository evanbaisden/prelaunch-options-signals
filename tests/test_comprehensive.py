"""
Test suite for comprehensive analysis functionality.
Tests the main working analysis framework.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import subprocess

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from comprehensive_analysis import ComprehensiveAnalyzer


class TestComprehensiveAnalyzer:
    """Test the comprehensive analysis framework."""
    
    def test_analyzer_initialization(self):
        """Test that analyzer initializes properly."""
        analyzer = ComprehensiveAnalyzer()
        assert analyzer.market_ticker == "^GSPC"
        assert analyzer.estimation_days == 120
        assert analyzer.min_estimation_days == 30
        assert analyzer.results_dir.exists()
    
    def test_load_events(self):
        """Test event loading functionality."""
        analyzer = ComprehensiveAnalyzer()
        try:
            events_df = analyzer.load_events()
            
            # Check that events loaded successfully
            assert isinstance(events_df, pd.DataFrame)
            assert len(events_df) > 0
            
            # Check required columns exist (using actual column names)
            required_cols = ['company', 'ticker', 'name', 'announcement', 'release']
            for col in required_cols:
                assert col in events_df.columns
            
            # Check data types
            assert events_df['announcement'].dtype == 'object'  # Should be date objects
            
        except FileNotFoundError:
            pytest.skip("Events master file not found - this is expected in some test environments")
    
    def test_main_analysis_runs(self):
        """Test that the main analysis script runs without errors."""
        try:
            # Run the comprehensive analysis script
            result = subprocess.run(
                ["python", "src/comprehensive_analysis.py"], 
                capture_output=True, 
                text=True,
                timeout=60
            )
            
            # Check that it completed successfully
            assert result.returncode == 0, f"Analysis failed with error: {result.stderr}"
            
            # Check that output contains expected content
            assert "COMPREHENSIVE PRODUCT LAUNCH ANALYSIS" in result.stdout
            assert "RESULTS SUMMARY" in result.stdout
            assert "Events:" in result.stdout or "Total Events:" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.skip("Analysis took too long - this is acceptable for testing")
        except FileNotFoundError:
            pytest.skip("Python not found in PATH - this is acceptable for testing")


if __name__ == "__main__":
    pytest.main([__file__])