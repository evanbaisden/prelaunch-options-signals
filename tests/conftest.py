"""
Pytest configuration and shared fixtures for prelaunch options signals tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import tempfile
import os

# Removed imports for types that are no longer used


@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    n_days = len(dates)
    np.random.seed(42)  # For reproducible tests
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)  # Small daily returns
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate volume data
    base_volume = 1000000
    volume_multiplier = np.random.lognormal(0, 0.5, n_days)
    volumes = (base_volume * volume_multiplier).astype(int)
    
    # Add some volume spikes
    spike_days = np.random.choice(n_days, size=10, replace=False)
    volumes[spike_days] *= np.random.uniform(3, 8, 10)
    
    df = pd.DataFrame({
        'Date': dates,
        'Adj Close': prices,
        'Close': prices,
        'High': prices * np.random.uniform(1.00, 1.05, n_days),
        'Low': prices * np.random.uniform(0.95, 1.00, n_days),
        'Open': prices * np.random.uniform(0.98, 1.02, n_days),
        'Volume': volumes
    })
    
    return df


# Removed fixtures for legacy types that are no longer used


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        yield temp_path


@pytest.fixture
def sample_csv_file(temp_data_dir, sample_stock_data):
    """Create sample CSV file with proper format."""
    csv_path = temp_data_dir / "test_data.csv"
    
    # Write CSV with proper header format (matching real data structure)
    with open(csv_path, 'w') as f:
        f.write("Price,Adj Close,Close,High,Low,Open,Volume\n")
        f.write("Ticker,TEST,TEST,TEST,TEST,TEST,TEST\n")
        f.write("Date,,,,,,\n")
    
    # Append actual data
    sample_stock_data.to_csv(csv_path, mode='a', header=False, index=False)
    
    return csv_path


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        'BASELINE_DAYS': '60',
        'SIGNAL_WINDOW_ANNOUNCE': '5,20',
        'SIGNAL_WINDOW_RELEASE': '5,10',
        'Z_THRESHOLD_LOW': '1.645',
        'Z_THRESHOLD_MED': '2.326',
        'Z_THRESHOLD_HIGH': '2.576',
        'Z_THRESHOLD_EXTREME': '5.0',
        'DATA_RAW_DIR': 'test_data',
        'RESULTS_DIR': 'test_results',
        'LOG_LEVEL': 'WARNING'
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def stock_data_with_quality_issues():
    """Generate stock data with various quality issues for testing validation."""
    dates = pd.date_range(start='2020-01-01', end='2020-06-30', freq='D')
    dates = dates[dates.weekday < 5]
    
    n_days = len(dates)
    
    df = pd.DataFrame({
        'Date': dates,
        'Adj Close': np.random.uniform(50, 150, n_days),
        'Close': np.random.uniform(50, 150, n_days),
        'High': np.random.uniform(100, 200, n_days),
        'Low': np.random.uniform(10, 100, n_days),
        'Open': np.random.uniform(50, 150, n_days),
        'Volume': np.random.randint(10000, 5000000, n_days)
    })
    
    # Introduce quality issues
    # Missing values
    df.loc[10:15, 'Adj Close'] = np.nan
    df.loc[20:22, 'Volume'] = np.nan
    
    # Zero/negative prices
    df.loc[30, 'Adj Close'] = 0
    df.loc[31, 'Adj Close'] = -10
    
    # Extreme price movements
    df.loc[40, 'Adj Close'] = df.loc[39, 'Adj Close'] * 2.5  # 150% increase
    
    # Date gaps (remove some dates)
    df = df.drop(df.index[50:60])  # 10-day gap
    
    return df