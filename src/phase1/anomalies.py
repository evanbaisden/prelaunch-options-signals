"""
Anomaly detection for volume and price patterns in prelaunch analysis.
Detects unusual trading activity around product announcement and release events.
"""
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
from ..common.utils import safe_z_score, find_nearest_date_index, detect_volume_anomalies
from ..config import Config


def detect_volume_spikes(
    df: pd.DataFrame,
    event_date: date,
    window_before: int = 5,
    window_after: int = 5,
    baseline_window: int = 20
) -> Dict[str, float]:
    """
    Detect volume spikes around event dates.
    
    Args:
        df: Stock data DataFrame
        event_date: Event date to analyze
        window_before: Days before event to include
        window_after: Days after event to include  
        baseline_window: Rolling window for baseline calculation
        
    Returns:
        Dictionary with volume spike metrics
    """
    config = Config()
    
    # Find event date index
    event_idx = find_nearest_date_index(df, event_date)
    
    # Calculate baseline volume using rolling mean
    df_copy = df.copy()
    df_copy['Volume_Baseline'] = df_copy['Volume'].rolling(
        window=baseline_window, 
        min_periods=baseline_window//2
    ).mean()
    
    # Define event window
    start_idx = max(0, event_idx - window_before)
    end_idx = min(len(df_copy), event_idx + window_after + 1)
    event_window = df_copy.iloc[start_idx:end_idx]
    
    # Calculate volume metrics
    event_volume_avg = event_window['Volume'].mean()
    baseline_volume_avg = event_window['Volume_Baseline'].mean()
    
    # Volume spike percentage
    volume_spike_pct = (event_volume_avg / baseline_volume_avg - 1) if baseline_volume_avg > 0 else 0
    
    # Calculate volume ratios for z-score
    volume_ratios = df_copy['Volume'] / df_copy['Volume_Baseline']
    volume_ratios = volume_ratios.dropna()
    
    # Calculate z-score for event period
    event_volume_ratios = event_window['Volume'] / event_window['Volume_Baseline']
    event_avg_ratio = event_volume_ratios.mean()
    
    # Calculate z-score using historical distribution
    z_score = safe_z_score(volume_ratios).iloc[event_idx] if len(volume_ratios) > event_idx else 0
    
    # Determine significance level
    significance_level = None
    abs_z = abs(z_score)
    if abs_z >= config.z_threshold_extreme:
        significance_level = 'extreme'
    elif abs_z >= config.z_threshold_high:
        significance_level = 'high'
    elif abs_z >= config.z_threshold_med:
        significance_level = 'med'
    elif abs_z >= config.z_threshold_low:
        significance_level = 'low'
    
    return {
        'volume_spike_pct': volume_spike_pct,
        'volume_z_score': z_score,
        'event_volume_avg': event_volume_avg,
        'baseline_volume_avg': baseline_volume_avg,
        'significance_level': significance_level,
        'volume_anomaly_detected': abs_z >= config.z_threshold_low
    }


def detect_price_anomalies(
    df: pd.DataFrame,
    event_date: date,
    window_days: int = 5,
    baseline_days: int = 60
) -> Dict[str, float]:
    """
    Detect price movement anomalies around event dates.
    
    Args:
        df: Stock data DataFrame  
        event_date: Event date to analyze
        window_days: Days around event to calculate return
        baseline_days: Days before event for baseline calculation
        
    Returns:
        Dictionary with price anomaly metrics
    """
    # Find event date index
    event_idx = find_nearest_date_index(df, event_date)
    
    # Calculate baseline period
    baseline_start = max(0, event_idx - baseline_days)
    baseline_data = df.iloc[baseline_start:event_idx].copy()
    baseline_data['Daily_Return'] = baseline_data['Adj Close'].pct_change()
    
    # Calculate event window return (t-5 to t+0)
    start_idx = max(0, event_idx - window_days)
    if start_idx < len(df) and event_idx < len(df):
        start_price = df.iloc[start_idx]['Adj Close']
        event_price = df.iloc[event_idx]['Adj Close']
        event_return = (event_price - start_price) / start_price
    else:
        event_return = 0.0
    
    # Calculate 1-day return
    if event_idx > 0 and event_idx < len(df):
        prev_price = df.iloc[event_idx - 1]['Adj Close']
        curr_price = df.iloc[event_idx]['Adj Close']
        one_day_return = (curr_price - prev_price) / prev_price
    else:
        one_day_return = 0.0
    
    # Calculate baseline volatility
    baseline_volatility = baseline_data['Daily_Return'].std() if len(baseline_data) > 1 else 0.0
    baseline_mean_return = baseline_data['Daily_Return'].mean() if len(baseline_data) > 0 else 0.0
    
    # Calculate abnormal return (actual - expected)
    expected_return = baseline_mean_return * window_days  # Expected cumulative return
    abnormal_return = event_return - expected_return
    
    return {
        'price_5day_return': event_return,
        'price_1day_return': one_day_return,
        'abnormal_return': abnormal_return,
        'baseline_volatility': baseline_volatility,
        'baseline_mean_return': baseline_mean_return
    }


def identify_trading_anomalies(
    df: pd.DataFrame,
    announcement_date: date,
    release_date: date
) -> Dict[str, List[Dict]]:
    """
    Comprehensive anomaly detection across announcement and release periods.
    
    Args:
        df: Stock data DataFrame
        announcement_date: Product announcement date
        release_date: Product release date
        
    Returns:
        Dictionary with detected anomalies by category
    """
    config = Config()
    anomalies = {
        'volume_anomalies': [],
        'price_anomalies': [],
        'combined_anomalies': []
    }
    
    # Detect volume anomalies using the utility function
    volume_anomaly_mask = detect_volume_anomalies(df, z_threshold=config.z_threshold_low)
    
    # Find anomaly dates
    anomaly_dates = df[volume_anomaly_mask]['Date'].tolist()
    
    for anomaly_date in anomaly_dates:
        # Check if anomaly is near announcement or release
        days_to_announce = (pd.Timestamp(announcement_date) - anomaly_date).days
        days_to_release = (pd.Timestamp(release_date) - anomaly_date).days
        
        # Volume metrics for this anomaly
        volume_metrics = detect_volume_spikes(df, anomaly_date.date())
        
        anomaly_info = {
            'date': anomaly_date.date(),
            'days_to_announcement': days_to_announce,
            'days_to_release': days_to_release,
            'metrics': volume_metrics
        }
        
        # Categorize by proximity to events
        if abs(days_to_announce) <= 10:
            anomaly_info['event_proximity'] = 'announcement'
        elif abs(days_to_release) <= 10:
            anomaly_info['event_proximity'] = 'release'
        else:
            anomaly_info['event_proximity'] = 'other'
        
        anomalies['volume_anomalies'].append(anomaly_info)
    
    return anomalies


def calculate_anomaly_scores(anomalies: Dict[str, List[Dict]]) -> Dict[str, float]:
    """
    Calculate aggregate anomaly scores for the analysis period.
    
    Args:
        anomalies: Dictionary of detected anomalies
        
    Returns:
        Dictionary of anomaly scores and metrics
    """
    volume_anomalies = anomalies.get('volume_anomalies', [])
    
    # Count anomalies by proximity
    announcement_anomalies = [a for a in volume_anomalies if a['event_proximity'] == 'announcement']
    release_anomalies = [a for a in volume_anomalies if a['event_proximity'] == 'release']
    
    # Calculate average z-scores
    announcement_z_scores = [a['metrics']['volume_z_score'] for a in announcement_anomalies]
    release_z_scores = [a['metrics']['volume_z_score'] for a in release_anomalies]
    
    return {
        'total_anomalies': len(volume_anomalies),
        'announcement_anomalies': len(announcement_anomalies),
        'release_anomalies': len(release_anomalies),
        'avg_announcement_z_score': np.mean(announcement_z_scores) if announcement_z_scores else 0.0,
        'avg_release_z_score': np.mean(release_z_scores) if release_z_scores else 0.0,
        'max_z_score': max([a['metrics']['volume_z_score'] for a in volume_anomalies]) if volume_anomalies else 0.0
    }