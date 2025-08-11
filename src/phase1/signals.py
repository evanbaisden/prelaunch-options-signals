"""
Signal detection and extraction for prelaunch options analysis.
Focuses on announcement and release event signals.
"""
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, Dict, Tuple
from ..common.types import SignalMetrics, LaunchEvent
from ..common.utils import find_nearest_date_index
from ..config import Config
from .anomalies import detect_volume_spikes, detect_price_anomalies


def extract_announcement_signal(
    df: pd.DataFrame,
    event: LaunchEvent,
    window_before: int = 5,
    window_after: int = 20
) -> SignalMetrics:
    """
    Extract trading signals around product announcement.
    
    Args:
        df: Stock data DataFrame
        event: LaunchEvent object with announcement date
        window_before: Days before announcement to analyze
        window_after: Days after announcement to analyze
        
    Returns:
        SignalMetrics object with announcement signals
    """
    announcement_date = event.announcement
    
    # Get volume signals
    volume_metrics = detect_volume_spikes(
        df, announcement_date, 
        window_before=window_before,
        window_after=window_after
    )
    
    # Get price signals  
    price_metrics = detect_price_anomalies(df, announcement_date, window_days=5)
    
    return SignalMetrics(
        event_date=announcement_date,
        event_type='announcement',
        price_5day_return=price_metrics['price_5day_return'],
        price_1day_return=price_metrics['price_1day_return'],
        abnormal_return=price_metrics['abnormal_return'],
        volume_spike_pct=volume_metrics['volume_spike_pct'],
        volume_z_score=volume_metrics['volume_z_score'],
        volume_anomaly_detected=volume_metrics['volume_anomaly_detected'],
        significance_level=volume_metrics['significance_level']
    )


def extract_release_signal(
    df: pd.DataFrame,
    event: LaunchEvent,
    window_before: int = 5,
    window_after: int = 10
) -> SignalMetrics:
    """
    Extract trading signals around product release.
    
    Args:
        df: Stock data DataFrame
        event: LaunchEvent object with release date
        window_before: Days before release to analyze
        window_after: Days after release to analyze
        
    Returns:
        SignalMetrics object with release signals
    """
    release_date = event.release
    
    # Get volume signals
    volume_metrics = detect_volume_spikes(
        df, release_date,
        window_before=window_before, 
        window_after=window_after
    )
    
    # Get price signals
    price_metrics = detect_price_anomalies(df, release_date, window_days=5)
    
    return SignalMetrics(
        event_date=release_date,
        event_type='release',
        price_5day_return=price_metrics['price_5day_return'],
        price_1day_return=price_metrics['price_1day_return'],
        abnormal_return=price_metrics['abnormal_return'],
        volume_spike_pct=volume_metrics['volume_spike_pct'],
        volume_z_score=volume_metrics['volume_z_score'],
        volume_anomaly_detected=volume_metrics['volume_anomaly_detected'],
        significance_level=volume_metrics['significance_level']
    )


def calculate_signal_strength(signal: SignalMetrics) -> float:
    """
    Calculate overall signal strength score (0-100).
    
    Args:
        signal: SignalMetrics object
        
    Returns:
        Signal strength score
    """
    config = Config()
    
    # Price component (0-50 points)
    price_score = 0.0
    abs_return = abs(signal.price_5day_return)
    if abs_return > 0.05:  # >5% move
        price_score = min(25.0, abs_return * 500)  # Scale to 0-25
    
    abs_abnormal = abs(signal.abnormal_return) if signal.abnormal_return else 0.0
    if abs_abnormal > 0.02:  # >2% abnormal return
        price_score += min(25.0, abs_abnormal * 1250)  # Scale to 0-25
    
    # Volume component (0-50 points) 
    volume_score = 0.0
    abs_z_score = abs(signal.volume_z_score)
    
    if abs_z_score >= config.z_threshold_extreme:
        volume_score = 50.0
    elif abs_z_score >= config.z_threshold_high:
        volume_score = 40.0
    elif abs_z_score >= config.z_threshold_med:
        volume_score = 30.0
    elif abs_z_score >= config.z_threshold_low:
        volume_score = 20.0
    else:
        volume_score = min(20.0, abs_z_score * 12)  # Scale below threshold
    
    return min(100.0, price_score + volume_score)


def detect_multi_day_patterns(
    df: pd.DataFrame,
    event_date: date,
    window_days: int = 10
) -> Dict[str, any]:
    """
    Detect multi-day trading patterns around events.
    
    Args:
        df: Stock data DataFrame
        event_date: Event date to analyze around
        window_days: Days before/after event to analyze
        
    Returns:
        Dictionary with pattern analysis results
    """
    event_idx = find_nearest_date_index(df, event_date)
    
    # Extract window around event
    start_idx = max(0, event_idx - window_days)
    end_idx = min(len(df), event_idx + window_days + 1)
    window_data = df.iloc[start_idx:end_idx].copy()
    
    # Calculate daily metrics
    window_data['Daily_Return'] = window_data['Adj Close'].pct_change()
    window_data['Volume_MA20'] = window_data['Volume'].rolling(20, min_periods=10).mean()
    window_data['Volume_Ratio'] = window_data['Volume'] / window_data['Volume_MA20']
    
    # Pattern detection
    patterns = {}
    
    # Consecutive up/down days
    returns = window_data['Daily_Return'].dropna()
    up_days = (returns > 0).astype(int)
    down_days = (returns < 0).astype(int)
    
    # Find consecutive runs
    up_runs = []
    down_runs = []
    current_up_run = 0
    current_down_run = 0
    
    for is_up in up_days:
        if is_up:
            current_up_run += 1
            if current_down_run > 0:
                down_runs.append(current_down_run)
                current_down_run = 0
        else:
            current_down_run += 1
            if current_up_run > 0:
                up_runs.append(current_up_run)
                current_up_run = 0
    
    # Add final runs
    if current_up_run > 0:
        up_runs.append(current_up_run)
    if current_down_run > 0:
        down_runs.append(current_down_run)
    
    patterns['max_consecutive_up_days'] = max(up_runs) if up_runs else 0
    patterns['max_consecutive_down_days'] = max(down_runs) if down_runs else 0
    patterns['total_up_days'] = sum(up_days)
    patterns['total_down_days'] = sum(down_days)
    
    # Volume persistence
    high_volume_days = (window_data['Volume_Ratio'] > 1.5).sum()
    patterns['high_volume_days'] = high_volume_days
    
    # Volatility metrics
    patterns['return_volatility'] = returns.std()
    patterns['max_single_day_return'] = returns.abs().max()
    patterns['avg_absolute_return'] = returns.abs().mean()
    
    return patterns


def compare_signals(
    announcement_signal: Optional[SignalMetrics],
    release_signal: Optional[SignalMetrics]
) -> Dict[str, any]:
    """
    Compare announcement and release signals to identify patterns.
    
    Args:
        announcement_signal: Announcement signal metrics
        release_signal: Release signal metrics
        
    Returns:
        Dictionary with signal comparison results
    """
    if not announcement_signal or not release_signal:
        return {'comparison_available': False}
    
    comparison = {'comparison_available': True}
    
    # Price signal comparison
    announce_return = announcement_signal.price_5day_return
    release_return = release_signal.price_5day_return
    
    comparison['stronger_price_signal'] = 'announcement' if abs(announce_return) > abs(release_return) else 'release'
    comparison['price_signal_correlation'] = np.corrcoef([announce_return, release_return])[0, 1] if announce_return != 0 and release_return != 0 else 0.0
    comparison['price_consistency'] = 'consistent' if (announce_return > 0) == (release_return > 0) else 'mixed'
    
    # Volume signal comparison  
    announce_z = announcement_signal.volume_z_score
    release_z = release_signal.volume_z_score
    
    comparison['stronger_volume_signal'] = 'announcement' if abs(announce_z) > abs(release_z) else 'release'
    comparison['volume_signal_persistence'] = abs(announce_z - release_z) < 1.0  # Similar z-scores
    
    # Overall signal strength
    announce_strength = calculate_signal_strength(announcement_signal)
    release_strength = calculate_signal_strength(release_signal)
    
    comparison['stronger_overall_signal'] = 'announcement' if announce_strength > release_strength else 'release'
    comparison['signal_strength_ratio'] = announce_strength / release_strength if release_strength > 0 else float('inf')
    
    return comparison