"""
Tests for baseline metrics calculation module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from src.phase1.baseline import (
    calculate_baseline_metrics,
    calculate_baseline_volume_ma,
    validate_baseline_data,
    calculate_period_averages
)
from src.common.types import BaselineMetrics


class TestCalculateBaselineMetrics:
    """Test baseline metrics calculation."""
    
    def test_calculate_baseline_metrics_normal_case(self, sample_stock_data, sample_launch_event):
        """Test baseline calculation with normal data."""
        baseline = calculate_baseline_metrics(
            sample_stock_data, 
            sample_launch_event.announcement,
            baseline_days=60
        )
        
        assert isinstance(baseline, BaselineMetrics)
        assert baseline.trading_days <= 60  # May be less due to weekends
        assert baseline.trading_days > 0
        assert baseline.avg_daily_return is not None
        assert baseline.avg_volume > 0
        assert baseline.volume_std >= 0
        assert baseline.price_volatility >= 0
        assert baseline.start_date < baseline.end_date
    
    def test_calculate_baseline_metrics_insufficient_data(self, sample_stock_data):
        """Test baseline calculation with insufficient data."""
        # Use announcement date very early in dataset
        early_date = sample_stock_data['Date'].iloc[5].date()
        
        baseline = calculate_baseline_metrics(
            sample_stock_data,
            early_date,
            baseline_days=60
        )
        
        # Should work but with fewer trading days
        assert baseline.trading_days < 60
        assert baseline.trading_days > 0
    
    def test_calculate_baseline_metrics_no_data(self, sample_stock_data):
        """Test baseline calculation with no available data."""
        # Use announcement date before dataset starts
        early_date = sample_stock_data['Date'].iloc[0].date() - timedelta(days=1)
        
        with pytest.raises(ValueError, match="No baseline data available"):
            calculate_baseline_metrics(
                sample_stock_data,
                early_date,
                baseline_days=60
            )


class TestCalculateBaselineVolumeMA:
    """Test baseline volume moving average calculation."""
    
    def test_baseline_volume_ma_normal(self, sample_stock_data):
        """Test normal volume moving average calculation."""
        volume_ma = calculate_baseline_volume_ma(sample_stock_data, window=20)
        
        assert len(volume_ma) == len(sample_stock_data)
        assert volume_ma.name == 'Volume'
        assert not volume_ma.iloc[-20:].isna().any()  # Last 20 values should not be NaN
    
    def test_baseline_volume_ma_missing_volume_column(self, sample_stock_data):
        """Test error handling for missing Volume column."""
        df_no_volume = sample_stock_data.drop('Volume', axis=1)
        
        with pytest.raises(ValueError, match="Volume column not found"):
            calculate_baseline_volume_ma(df_no_volume)
    
    def test_baseline_volume_ma_small_window(self, sample_stock_data):
        """Test volume MA with small window size."""
        volume_ma = calculate_baseline_volume_ma(sample_stock_data, window=5)
        
        assert len(volume_ma) == len(sample_stock_data)
        # First few values might be NaN due to min_periods
        assert not volume_ma.iloc[10:].isna().any()


class TestValidateBaselineData:
    """Test baseline data validation."""
    
    def test_validate_baseline_data_good_data(self, sample_stock_data):
        """Test validation with good quality data."""
        baseline_data = sample_stock_data.iloc[:60]
        result = validate_baseline_data(baseline_data)
        
        assert result is True
    
    def test_validate_baseline_data_insufficient_length(self, sample_stock_data):
        """Test validation with insufficient data length."""
        short_data = sample_stock_data.iloc[:20]
        result = validate_baseline_data(short_data, min_days=30)
        
        assert result is False
    
    def test_validate_baseline_data_missing_prices(self, sample_stock_data):
        """Test validation with missing price data."""
        bad_data = sample_stock_data.copy()
        bad_data.loc[10:15, 'Adj Close'] = np.nan
        
        result = validate_baseline_data(bad_data)
        
        assert result is False
    
    def test_validate_baseline_data_missing_volume(self, sample_stock_data):
        """Test validation with missing volume data."""
        bad_data = sample_stock_data.copy()
        bad_data.loc[5:8, 'Volume'] = np.nan
        
        result = validate_baseline_data(bad_data)
        
        assert result is False
    
    def test_validate_baseline_data_negative_prices(self, sample_stock_data):
        """Test validation with negative/zero prices."""
        bad_data = sample_stock_data.copy()
        bad_data.loc[5, 'Adj Close'] = 0
        bad_data.loc[6, 'Adj Close'] = -10
        
        result = validate_baseline_data(bad_data)
        
        assert result is False


class TestCalculatePeriodAverages:
    """Test period averages calculation."""
    
    def test_calculate_period_averages_normal(self, sample_stock_data, sample_launch_event):
        """Test period averages with normal data."""
        averages = calculate_period_averages(
            sample_stock_data,
            sample_launch_event.announcement,
            sample_launch_event.release
        )
        
        required_keys = [
            'pre_announce_avg_return',
            'pre_announce_avg_volume',
            'announce_to_release_avg_return',
            'announce_to_release_avg_volume', 
            'post_release_avg_return',
            'post_release_avg_volume'
        ]
        
        for key in required_keys:
            assert key in averages
            assert averages[key] is not None
        
        # Volume values should be positive
        assert averages['pre_announce_avg_volume'] > 0
        assert averages['announce_to_release_avg_volume'] > 0
        assert averages['post_release_avg_volume'] > 0
    
    def test_calculate_period_averages_edge_dates(self, sample_stock_data):
        """Test period averages with edge case dates."""
        # Use dates at the very start and end of dataset
        start_date = sample_stock_data['Date'].iloc[30].date()
        end_date = sample_stock_data['Date'].iloc[60].date()
        
        averages = calculate_period_averages(
            sample_stock_data,
            start_date,
            end_date,
            baseline_days=20,
            post_release_days=20
        )
        
        # Should handle edge cases gracefully
        assert all(key in averages for key in [
            'pre_announce_avg_return', 'pre_announce_avg_volume',
            'announce_to_release_avg_return', 'announce_to_release_avg_volume',
            'post_release_avg_return', 'post_release_avg_volume'
        ])
    
    def test_calculate_period_averages_returns_calculation(self, sample_stock_data, sample_launch_event):
        """Test that returns are calculated correctly."""
        averages = calculate_period_averages(
            sample_stock_data,
            sample_launch_event.announcement,
            sample_launch_event.release
        )
        
        # Returns should be reasonable (not extreme values)
        for return_key in ['pre_announce_avg_return', 'announce_to_release_avg_return', 'post_release_avg_return']:
            if averages[return_key] is not None and not np.isnan(averages[return_key]):
                assert abs(averages[return_key]) < 0.1  # Less than 10% daily return on average