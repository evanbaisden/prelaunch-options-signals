import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from analysis import PrelaunchAnalyzer

class TestPrelaunchAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        return PrelaunchAnalyzer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample stock data for testing"""
        dates = pd.date_range(start='2020-01-01', end='2020-03-31', freq='D')
        # Filter to business days only
        dates = dates[dates.dayofweek < 5]
        
        np.random.seed(42)  # For reproducible tests
        data = {
            'Date': dates,
            'Adj Close': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'Close': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'High': 105 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'Low': 95 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'Open': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'Volume': np.random.lognormal(15, 0.5, len(dates))  # Realistic volume distribution
        }
        
        return pd.DataFrame(data)
    
    def test_calculate_baseline_volume(self, analyzer, sample_data):
        """Test baseline volume calculation"""
        baseline = analyzer.calculate_baseline_volume(sample_data, window=20)
        
        # Check that baseline is a pandas Series
        assert isinstance(baseline, pd.Series)
        
        # Check that baseline has same length as input
        assert len(baseline) == len(sample_data)
        
        # Check that first 9 values are NaN (min_periods=window//2 = 10)
        assert baseline.iloc[:9].isna().all()
        
        # Check that remaining values are not NaN
        assert not baseline.iloc[10:].isna().any()
        
        # Check that baseline values are positive (volumes should be positive)
        assert (baseline.dropna() > 0).all()
    
    def test_detect_volume_anomalies(self, analyzer, sample_data):
        """Test volume anomaly detection"""
        # Add some artificial anomalies
        test_data = sample_data.copy()
        test_data.loc[30, 'Volume'] *= 10  # Create outlier
        test_data.loc[40, 'Volume'] *= 0.1  # Create low volume outlier
        
        anomalies = analyzer.detect_volume_anomalies(test_data, z_threshold=2.0)
        
        # Check that result is boolean Series
        assert isinstance(anomalies, pd.Series)
        assert anomalies.dtype == bool
        
        # Check that some anomalies were detected
        assert anomalies.sum() > 0
        
        # Check that our artificial outlier was detected
        assert anomalies.iloc[30] == True, "High volume outlier should be detected"
    
    def test_find_nearest_date_index(self, analyzer, sample_data):
        """Test date index finding function"""
        target_date = datetime(2020, 2, 15)
        
        idx = analyzer.find_nearest_date_index(sample_data, target_date)
        
        # Check that index is valid
        assert 0 <= idx < len(sample_data)
        
        # Check that returned date is close to target
        found_date = sample_data.iloc[idx]['Date']
        date_diff = abs((found_date - target_date).days)
        assert date_diff <= 3, f"Found date should be within 3 days of target, got {date_diff}"
    
    def test_load_stock_data_structure(self, analyzer, tmp_path):
        """Test stock data loading structure"""
        # Create mock CSV file with expected structure
        csv_content = """Price,Adj Close,Close,High,Low,Open,Volume
Ticker,TEST,TEST,TEST,TEST,TEST,TEST
Date,,,,,,
2020-01-01,100.0,100.0,105.0,95.0,98.0,1000000
2020-01-02,101.0,101.0,106.0,96.0,99.0,1100000
2020-01-03,99.0,99.0,104.0,94.0,100.0,900000"""
        
        test_file = tmp_path / "test_stock.csv"
        test_file.write_text(csv_content)
        
        # Temporarily change data directory
        original_dir = analyzer.data_dir
        analyzer.data_dir = tmp_path
        
        try:
            df = analyzer.load_stock_data("test_stock.csv")
            
            # Check structure
            expected_columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
            assert list(df.columns) == expected_columns
            
            # Check data types
            assert pd.api.types.is_datetime64_any_dtype(df['Date'])
            assert pd.api.types.is_numeric_dtype(df['Volume'])
            assert pd.api.types.is_numeric_dtype(df['Adj Close'])
            
            # Check data length
            assert len(df) == 3
            
        finally:
            analyzer.data_dir = original_dir
    
    def test_parameters_loading(self):
        """Test that parameters are loaded correctly"""
        analyzer = PrelaunchAnalyzer()
        
        # Check that parameters exist and have expected types
        assert isinstance(analyzer.params, dict)
        assert 'baseline_days' in analyzer.params
        assert 'z_thresholds' in analyzer.params
        
        # Check parameter types
        assert isinstance(analyzer.params['baseline_days'], int)
        assert isinstance(analyzer.params['z_thresholds'], dict)
        
        # Check that z_thresholds has expected keys
        expected_z_keys = ['low', 'med', 'high', 'extreme']
        assert all(key in analyzer.params['z_thresholds'] for key in expected_z_keys)
    
    def test_metadata_structure(self, analyzer):
        """Test that metadata is properly structured"""
        assert isinstance(analyzer.metadata, dict)
        
        required_keys = ['analysis_timestamp', 'parameters', 'data_sources', 'calculation_definitions']
        assert all(key in analyzer.metadata for key in required_keys)
        
        # Check calculation definitions
        calc_defs = analyzer.metadata['calculation_definitions']
        required_calc_keys = ['announcement_5day_return', 'release_5day_return', 'volume_spike_pct']
        assert all(key in calc_defs for key in required_calc_keys)

if __name__ == "__main__":
    pytest.main([__file__])