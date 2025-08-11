"""
Tests for anomaly detection module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from src.phase1.anomalies import (
    detect_volume_spikes,
    detect_price_anomalies,
    identify_trading_anomalies,
    calculate_anomaly_scores
)


class TestDetectVolumeSpikes:
    """Test volume spike detection."""
    
    def test_detect_volume_spikes_normal(self, sample_stock_data, sample_launch_event):
        """Test volume spike detection with normal data."""
        results = detect_volume_spikes(
            sample_stock_data,
            sample_launch_event.announcement,
            window_before=5,
            window_after=5
        )
        
        required_keys = [
            'volume_spike_pct', 'volume_z_score', 'event_volume_avg',
            'baseline_volume_avg', 'significance_level', 'volume_anomaly_detected'
        ]
        
        for key in required_keys:
            assert key in results
        
        assert isinstance(results['volume_spike_pct'], (int, float))
        assert isinstance(results['volume_z_score'], (int, float))
        assert isinstance(results['volume_anomaly_detected'], bool)
        assert results['event_volume_avg'] > 0
        assert results['baseline_volume_avg'] > 0
    
    def test_detect_volume_spikes_with_spike(self, sample_stock_data, sample_launch_event):
        """Test volume spike detection with artificially added spike."""
        # Add volume spike around announcement
        df_with_spike = sample_stock_data.copy()
        
        # Find announcement date index
        announcement_idx = None
        for i, row in df_with_spike.iterrows():
            if row['Date'].date() == sample_launch_event.announcement:
                announcement_idx = i
                break
        
        if announcement_idx is not None:
            # Add 5x volume spike
            spike_indices = range(max(0, announcement_idx - 2), 
                                min(len(df_with_spike), announcement_idx + 3))
            df_with_spike.loc[spike_indices, 'Volume'] *= 5
        
        results = detect_volume_spikes(
            df_with_spike,
            sample_launch_event.announcement
        )
        
        # Should detect significant spike
        assert results['volume_spike_pct'] > 1.0  # More than 100% increase
        assert abs(results['volume_z_score']) > 1.0  # Should be significant
        assert results['volume_anomaly_detected'] is True
    
    def test_detect_volume_spikes_edge_cases(self, sample_stock_data):
        """Test volume spike detection with edge cases."""
        # Test with date at very start of data
        early_date = sample_stock_data['Date'].iloc[5].date()
        
        results = detect_volume_spikes(sample_stock_data, early_date)
        
        # Should handle gracefully even with limited baseline
        assert 'volume_spike_pct' in results
        assert 'volume_z_score' in results
    
    def test_detect_volume_spikes_significance_levels(self, sample_stock_data, sample_launch_event):
        """Test significance level classification."""
        # Test with different spike magnitudes
        df_extreme = sample_stock_data.copy()
        
        # Find announcement index
        announcement_idx = None
        for i, row in df_extreme.iterrows():
            if row['Date'].date() == sample_launch_event.announcement:
                announcement_idx = i
                break
        
        if announcement_idx is not None:
            # Create extreme volume spike
            df_extreme.loc[announcement_idx, 'Volume'] *= 20
            
            results = detect_volume_spikes(df_extreme, sample_launch_event.announcement)
            
            # Should detect high significance
            assert results['significance_level'] in ['low', 'med', 'high', 'extreme']
            if abs(results['volume_z_score']) > 5.0:
                assert results['significance_level'] == 'extreme'


class TestDetectPriceAnomalies:
    """Test price anomaly detection."""
    
    def test_detect_price_anomalies_normal(self, sample_stock_data, sample_launch_event):
        """Test price anomaly detection with normal data."""
        results = detect_price_anomalies(
            sample_stock_data,
            sample_launch_event.announcement,
            window_days=5
        )
        
        required_keys = [
            'price_5day_return', 'price_1day_return', 'abnormal_return',
            'baseline_volatility', 'baseline_mean_return'
        ]
        
        for key in required_keys:
            assert key in results
            assert isinstance(results[key], (int, float))
    
    def test_detect_price_anomalies_with_spike(self, sample_stock_data, sample_launch_event):
        """Test price anomaly detection with artificial price spike."""
        df_with_spike = sample_stock_data.copy()
        
        # Find announcement date
        announcement_idx = None
        for i, row in df_with_spike.iterrows():
            if row['Date'].date() == sample_launch_event.announcement:
                announcement_idx = i
                break
        
        if announcement_idx is not None and announcement_idx >= 5:
            # Create price spike (20% increase over 5 days)
            start_price = df_with_spike.loc[announcement_idx - 5, 'Adj Close']
            df_with_spike.loc[announcement_idx, 'Adj Close'] = start_price * 1.2
            
            results = detect_price_anomalies(
                df_with_spike,
                sample_launch_event.announcement,
                window_days=5
            )
            
            # Should detect significant return
            assert abs(results['price_5day_return']) > 0.1  # > 10% return
            assert results['abnormal_return'] != 0
    
    def test_detect_price_anomalies_edge_dates(self, sample_stock_data):
        """Test price anomaly detection with edge case dates."""
        # Test with date at very start (insufficient baseline)
        early_date = sample_stock_data['Date'].iloc[2].date()
        
        results = detect_price_anomalies(sample_stock_data, early_date)
        
        # Should handle gracefully
        assert 'price_5day_return' in results
        assert 'baseline_volatility' in results
        
        # Test with date at very end
        late_date = sample_stock_data['Date'].iloc[-2].date()
        
        results = detect_price_anomalies(sample_stock_data, late_date)
        assert 'price_5day_return' in results


class TestIdentifyTradingAnomalies:
    """Test comprehensive trading anomaly identification."""
    
    def test_identify_trading_anomalies_structure(self, sample_stock_data, sample_launch_event):
        """Test the structure of trading anomaly results."""
        anomalies = identify_trading_anomalies(
            sample_stock_data,
            sample_launch_event.announcement,
            sample_launch_event.release
        )
        
        required_keys = ['volume_anomalies', 'price_anomalies', 'combined_anomalies']
        
        for key in required_keys:
            assert key in anomalies
            assert isinstance(anomalies[key], list)
    
    def test_identify_trading_anomalies_proximity(self, sample_stock_data, sample_launch_event):
        """Test anomaly proximity classification."""
        # Add volume spikes near announcement and release dates
        df_with_anomalies = sample_stock_data.copy()
        
        # Find announcement and release indices
        announcement_idx = None
        release_idx = None
        
        for i, row in df_with_anomalies.iterrows():
            if row['Date'].date() == sample_launch_event.announcement:
                announcement_idx = i
            elif row['Date'].date() == sample_launch_event.release:
                release_idx = i
        
        # Add volume spikes
        if announcement_idx is not None:
            df_with_anomalies.loc[announcement_idx, 'Volume'] *= 10
        
        if release_idx is not None:
            df_with_anomalies.loc[release_idx, 'Volume'] *= 8
        
        anomalies = identify_trading_anomalies(
            df_with_anomalies,
            sample_launch_event.announcement,
            sample_launch_event.release
        )
        
        # Check that anomalies are properly classified by proximity
        for anomaly in anomalies['volume_anomalies']:
            assert 'event_proximity' in anomaly
            assert anomaly['event_proximity'] in ['announcement', 'release', 'other']
            assert 'days_to_announcement' in anomaly
            assert 'days_to_release' in anomaly


class TestCalculateAnomalyScores:
    """Test anomaly score calculation."""
    
    def test_calculate_anomaly_scores_empty(self):
        """Test anomaly scores with no anomalies."""
        empty_anomalies = {
            'volume_anomalies': [],
            'price_anomalies': [],
            'combined_anomalies': []
        }
        
        scores = calculate_anomaly_scores(empty_anomalies)
        
        assert scores['total_anomalies'] == 0
        assert scores['announcement_anomalies'] == 0
        assert scores['release_anomalies'] == 0
        assert scores['avg_announcement_z_score'] == 0.0
        assert scores['avg_release_z_score'] == 0.0
        assert scores['max_z_score'] == 0.0
    
    def test_calculate_anomaly_scores_with_data(self):
        """Test anomaly scores with sample anomalies."""
        sample_anomalies = {
            'volume_anomalies': [
                {
                    'event_proximity': 'announcement',
                    'metrics': {'volume_z_score': 3.5}
                },
                {
                    'event_proximity': 'release',
                    'metrics': {'volume_z_score': 2.8}
                },
                {
                    'event_proximity': 'other',
                    'metrics': {'volume_z_score': 1.9}
                }
            ],
            'price_anomalies': [],
            'combined_anomalies': []
        }
        
        scores = calculate_anomaly_scores(sample_anomalies)
        
        assert scores['total_anomalies'] == 3
        assert scores['announcement_anomalies'] == 1
        assert scores['release_anomalies'] == 1
        assert scores['avg_announcement_z_score'] == 3.5
        assert scores['avg_release_z_score'] == 2.8
        assert scores['max_z_score'] == 3.5
    
    def test_calculate_anomaly_scores_statistics(self):
        """Test statistical calculations in anomaly scores."""
        # Create anomalies with known statistics
        anomalies_with_stats = {
            'volume_anomalies': [
                {'event_proximity': 'announcement', 'metrics': {'volume_z_score': 4.0}},
                {'event_proximity': 'announcement', 'metrics': {'volume_z_score': 2.0}},  # avg = 3.0
                {'event_proximity': 'release', 'metrics': {'volume_z_score': 5.0}},
                {'event_proximity': 'release', 'metrics': {'volume_z_score': 1.0}}  # avg = 3.0
            ]
        }
        
        scores = calculate_anomaly_scores(anomalies_with_stats)
        
        assert scores['announcement_anomalies'] == 2
        assert scores['release_anomalies'] == 2
        assert scores['avg_announcement_z_score'] == 3.0
        assert scores['avg_release_z_score'] == 3.0
        assert scores['max_z_score'] == 5.0