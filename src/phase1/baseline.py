"""
Baseline metrics calculation for prelaunch options signals analysis.
Establishes normal trading patterns and statistical baselines.
"""
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional
from ..common.types import BaselineMetrics, StockData
from ..common.utils import safe_z_score, find_nearest_date_index


def calculate_baseline_metrics(
    df: pd.DataFrame, 
    announcement_date: date,
    baseline_days: int = 60
) -> BaselineMetrics:
    """
    Calculate baseline statistical metrics for the pre-announcement period.
    
    Args:
        df: Stock data DataFrame with Date, Adj Close, Volume columns
        announcement_date: Product announcement date
        baseline_days: Number of trading days before announcement to use
        
    Returns:
        BaselineMetrics object with statistical measures
    """
    # Find announcement date index
    announce_idx = find_nearest_date_index(df, announcement_date)
    
    # Extract baseline period (60 days before announcement)
    baseline_start = max(0, announce_idx - baseline_days)
    baseline_data = df.iloc[baseline_start:announce_idx].copy()
    
    if len(baseline_data) == 0:
        raise ValueError("No baseline data available before announcement date")
    
    # Calculate daily returns
    baseline_data['Daily_Return'] = baseline_data['Adj Close'].pct_change()
    
    # Calculate metrics
    avg_daily_return = baseline_data['Daily_Return'].mean()
    avg_volume = baseline_data['Volume'].mean()
    volume_std = baseline_data['Volume'].std()
    price_volatility = baseline_data['Daily_Return'].std()
    trading_days = len(baseline_data)
    
    start_date = baseline_data['Date'].iloc[0].date()
    end_date = baseline_data['Date'].iloc[-1].date()
    
    return BaselineMetrics(
        avg_daily_return=avg_daily_return,
        avg_volume=avg_volume,
        volume_std=volume_std,
        price_volatility=price_volatility,
        trading_days=trading_days,
        start_date=start_date,
        end_date=end_date
    )


def calculate_baseline_volume_ma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate baseline volume using rolling mean for anomaly detection.
    
    Args:
        df: DataFrame with Volume column
        window: Rolling window size for baseline calculation
        
    Returns:
        Series of baseline volume values
    """
    if 'Volume' not in df.columns:
        raise ValueError("Volume column not found in DataFrame")
    
    # Use minimum periods to handle early data points
    min_periods = min(window // 2, 10)
    return df['Volume'].rolling(window=window, min_periods=min_periods).mean()


def validate_baseline_data(baseline_data: pd.DataFrame, min_days: int = 30) -> bool:
    """
    Validate that baseline data meets minimum quality requirements.
    
    Args:
        baseline_data: Baseline period DataFrame
        min_days: Minimum number of trading days required
        
    Returns:
        True if baseline data is valid, False otherwise
    """
    if len(baseline_data) < min_days:
        return False
    
    # Check for missing price data
    if baseline_data['Adj Close'].isna().any():
        return False
    
    # Check for missing volume data
    if baseline_data['Volume'].isna().any():
        return False
    
    # Check for zero or negative prices
    if (baseline_data['Adj Close'] <= 0).any():
        return False
    
    return True


def calculate_period_averages(
    df: pd.DataFrame,
    announcement_date: date,
    release_date: date,
    baseline_days: int = 60,
    post_release_days: int = 30
) -> dict:
    """
    Calculate average returns and volumes across key periods.
    
    Args:
        df: Stock data DataFrame
        announcement_date: Product announcement date
        release_date: Product release date
        baseline_days: Days before announcement for baseline
        post_release_days: Days after release to analyze
        
    Returns:
        Dictionary of period averages
    """
    # Find key date indices
    announce_idx = find_nearest_date_index(df, announcement_date)
    release_idx = find_nearest_date_index(df, release_date)
    
    # Calculate daily returns
    df_copy = df.copy()
    df_copy['Daily_Return'] = df_copy['Adj Close'].pct_change()
    
    # Define periods
    pre_announce_start = max(0, announce_idx - baseline_days)
    pre_announce_data = df_copy.iloc[pre_announce_start:announce_idx]
    
    announce_to_release_data = df_copy.iloc[announce_idx:release_idx]
    
    post_release_end = min(len(df_copy), release_idx + post_release_days)
    post_release_data = df_copy.iloc[release_idx:post_release_end]
    
    return {
        'pre_announce_avg_return': pre_announce_data['Daily_Return'].mean(),
        'pre_announce_avg_volume': pre_announce_data['Volume'].mean(),
        'announce_to_release_avg_return': announce_to_release_data['Daily_Return'].mean(),
        'announce_to_release_avg_volume': announce_to_release_data['Volume'].mean(),
        'post_release_avg_return': post_release_data['Daily_Return'].mean(),
        'post_release_avg_volume': post_release_data['Volume'].mean()
    }