"""
Utility functions for prelaunch options signals analysis.
IO helpers, date guards, statistical utilities, and caching.
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
from dotenv import load_dotenv


def load_env_file(env_path: Optional[str] = None) -> bool:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. If None, searches for .env in current directory.
        
    Returns:
        True if .env file was loaded successfully, False otherwise.
    """
    if env_path is None:
        env_path = ".env"
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
        return True
    return False


def setup_logging(level: str = "INFO", format_str: Optional[str] = None) -> None:
    """
    Configure logging for the analysis pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Custom format string for log messages
    """
    if format_str is None:
        format_str = "%(asctime)s %(levelname)s %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[logging.StreamHandler()]
    )


def validate_date_range(start_date: Union[str, date], end_date: Union[str, date]) -> Tuple[date, date]:
    """
    Validate and convert date range inputs.
    
    Args:
        start_date: Start date (string or date object)
        end_date: End date (string or date object)
        
    Returns:
        Tuple of validated date objects (start_date, end_date)
        
    Raises:
        ValueError: If dates are invalid or start_date > end_date
    """
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date).date()
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date).date()
    
    if start_date > end_date:
        raise ValueError(f"Start date {start_date} cannot be after end date {end_date}")
    
    return start_date, end_date


def safe_z_score(values: pd.Series, robust: bool = True) -> pd.Series:
    """
    Calculate z-scores with robust handling of edge cases.
    
    Args:
        values: Series of numerical values
        robust: If True, uses median and MAD for robust z-score calculation
        
    Returns:
        Series of z-scores, with NaN for invalid inputs
    """
    if len(values) < 2:
        return pd.Series([np.nan] * len(values), index=values.index)
    
    if robust:
        # Robust z-score using median absolute deviation
        median = values.median()
        mad = (values - median).abs().median()
        if mad == 0:
            return pd.Series([0.0] * len(values), index=values.index)
        z_scores = (values - median) / (1.4826 * mad)  # 1.4826 is consistency factor
    else:
        # Standard z-score
        mean = values.mean()
        std = values.std()
        if std == 0:
            return pd.Series([0.0] * len(values), index=values.index)
        z_scores = (values - mean) / std
    
    return z_scores


def find_nearest_date_index(df: pd.DataFrame, target_date: date, date_col: str = 'Date') -> int:
    """
    Find the index of the row with the date closest to target_date.
    
    Args:
        df: DataFrame with date column
        target_date: Target date to find
        date_col: Name of date column
        
    Returns:
        Index of nearest date
        
    Raises:
        ValueError: If date column not found or DataFrame is empty
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    # Convert target to pandas datetime for comparison
    target_dt = pd.Timestamp(target_date)
    date_diffs = (df[date_col] - target_dt).abs()
    return date_diffs.idxmin()


def calculate_baseline_volume(df: pd.DataFrame, window: int = 20, min_periods: Optional[int] = None) -> pd.Series:
    """
    Calculate baseline volume using rolling mean for anomaly detection.
    
    Args:
        df: DataFrame with Volume column
        window: Rolling window size
        min_periods: Minimum periods required for calculation
        
    Returns:
        Series of baseline volume values
    """
    if 'Volume' not in df.columns:
        raise ValueError("Volume column not found in DataFrame")
    
    if min_periods is None:
        min_periods = window // 2
    
    return df['Volume'].rolling(window=window, min_periods=min_periods).mean()


def detect_volume_anomalies(
    df: pd.DataFrame, 
    z_threshold: float = 2.0, 
    baseline_window: int = 20,
    robust: bool = True
) -> pd.Series:
    """
    Detect volume anomalies using z-score method.
    
    Args:
        df: DataFrame with Volume column
        z_threshold: Z-score threshold for anomaly detection
        baseline_window: Window for baseline calculation
        robust: Use robust z-score calculation
        
    Returns:
        Boolean series indicating anomalies
    """
    baseline = calculate_baseline_volume(df, window=baseline_window)
    volume_ratio = df['Volume'] / baseline
    z_scores = safe_z_score(volume_ratio, robust=robust)
    return z_scores.abs() > z_threshold


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def cache_results(func):
    """Simple decorator for caching expensive function results."""
    cache = {}
    
    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper


def validate_stock_data(df: pd.DataFrame, ticker: str) -> Dict[str, bool]:
    """
    Validate stock data quality and return quality checks.
    
    Args:
        df: Stock data DataFrame
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary of quality check results
    """
    checks = {}
    
    # Required columns
    required_cols = ['Date', 'Adj Close', 'Volume', 'High', 'Low', 'Open']
    checks['has_required_columns'] = all(col in df.columns for col in required_cols)
    
    # Data completeness
    checks['no_missing_prices'] = not df['Adj Close'].isna().any()
    checks['no_missing_volumes'] = not df['Volume'].isna().any()
    checks['no_zero_prices'] = not (df['Adj Close'] <= 0).any()
    checks['no_zero_volumes'] = not (df['Volume'] <= 0).any()
    
    # Date continuity (allowing for weekends/holidays)
    if 'Date' in df.columns and len(df) > 1:
        date_gaps = df['Date'].diff().dt.days
        max_gap = date_gaps.max()
        checks['reasonable_date_gaps'] = max_gap <= 7  # No gaps > 1 week
    else:
        checks['reasonable_date_gaps'] = True
    
    # Price sanity checks
    if checks['no_missing_prices']:
        price_changes = df['Adj Close'].pct_change().abs()
        checks['no_extreme_price_moves'] = not (price_changes > 0.5).any()  # No >50% daily moves
    else:
        checks['no_extreme_price_moves'] = False
    
    # Volume sanity checks
    if checks['no_missing_volumes']:
        volume_median = df['Volume'].median()
        volume_outliers = df['Volume'] > (volume_median * 100)
        checks['reasonable_volume_outliers'] = volume_outliers.sum() / len(df) < 0.01  # <1% extreme outliers
    else:
        checks['reasonable_volume_outliers'] = False
    
    return checks


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_large_number(value: float, unit: str = "M") -> str:
    """Format large numbers with appropriate units."""
    if unit == "M":
        return f"{value / 1e6:.1f}M"
    elif unit == "K":
        return f"{value / 1e3:.1f}K"
    elif unit == "B":
        return f"{value / 1e9:.2f}B"
    else:
        return f"{value:,.0f}"