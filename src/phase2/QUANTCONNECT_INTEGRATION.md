# QuantConnect Integration Guide

This guide explains how to integrate the Phase 2 options analysis with QuantConnect for real-time and historical options data.

## Overview

The Phase 2 system includes QuantConnect integration for:
- **Historical options data collection** via `OptionsDataCollectionAlgorithm`
- **Real-time signal detection** via `OptionsSignalDetectionAlgorithm`  
- **Data export and analysis** using QuantConnect's object store

## Architecture

```
Local Analysis System ←→ QuantConnect Cloud Platform
     ↓                           ↓
Phase 2 Analysis              Options Data Collection
     ↓                           ↓
Results & Signals ←→ ObjectStore/API ←→ Live Trading Algorithm
```

## Setup Instructions

### 1. QuantConnect Account Setup

1. Create account at [QuantConnect.com](https://www.quantconnect.com)
2. Subscribe to options data (required for historical and live data)
3. Note your API credentials for local integration

### 2. Deploy Data Collection Algorithm

1. Copy `quantconnect_algorithm.py` to QuantConnect IDE
2. Use `OptionsDataCollectionAlgorithm` for historical data collection:

```python
# In QuantConnect IDE
from quantconnect_algorithm import OptionsDataCollectionAlgorithm

# The algorithm will automatically:
# - Collect options chains for AAPL, NVDA, MSFT around product launches
# - Filter for relevant strikes and expirations  
# - Export data to QuantConnect object store
# - Run for date range 2020-2024
```

### 3. Configure Local Integration

Update your `.env` file:

```bash
# QuantConnect Integration
QUANTCONNECT_API_KEY=your_api_key_here
QUANTCONNECT_API_SECRET=your_secret_here
QUANTCONNECT_PROJECT_ID=your_project_id
QUANTCONNECT_ORGANIZATION_ID=your_org_id

# Data source priority (QuantConnect first)
OPTIONS_DATA_PROVIDERS=quantconnect,yahoo,polygon
```

### 4. Run Phase 2 Analysis

```bash
# Run Phase 2 with QuantConnect data
python -m src.phase2.run --data-source quantconnect

# Or run full pipeline
make phase2
```

## Data Collection Algorithm Features

### OptionsDataCollectionAlgorithm

**Purpose**: Collect comprehensive options data around product launch events

**Features**:
- **Target Events**: Pre-configured with Apple iPhone, NVIDIA RTX, Microsoft Xbox launches
- **Data Collection Window**: 60 days before to 30 days after each event
- **Options Filtering**: Strikes -20% to +20%, expirations up to 6 months
- **Data Export**: Automatic CSV export to QuantConnect object store
- **Greeks Collection**: Delta, Gamma, Theta, Vega, Rho when available

**Collected Data Points**:
```python
{
    'timestamp': datetime,
    'underlying_ticker': str,
    'underlying_price': float,
    'symbol': str,           # Option contract symbol
    'strike': float,
    'expiry': datetime,
    'option_type': str,      # 'call' or 'put'
    'bid': float,
    'ask': float,
    'last_price': float,
    'volume': int,
    'open_interest': int,
    'implied_volatility': float,
    'delta': float,
    'gamma': float,
    'theta': float,
    'vega': float,
    'rho': float
}
```

### OptionsSignalDetectionAlgorithm

**Purpose**: Real-time detection of unusual options activity

**Features**:
- **Signal Detection**: Put/call ratios, volume spikes, IV anomalies
- **Real-time Processing**: 15-minute signal detection intervals
- **Baseline Comparison**: Rolling 20-period historical comparisons
- **Alert Generation**: Configurable thresholds for signal strength

## Data Integration Workflow

### 1. Historical Analysis Mode

```python
from src.phase2.options_data import OptionsDataManager

# Initialize with QuantConnect provider
manager = OptionsDataManager(config)

# Collect options data for launch events
for event in launch_events:
    options_data = manager.get_options_for_event(event)
    # Data automatically sourced from QuantConnect if available
```

### 2. Live Signal Detection Mode

```python
from src.phase2.quantconnect_algorithm import OptionsSignalDetectionAlgorithm

# Deploy to QuantConnect for live trading
# Algorithm will:
# - Monitor options flow in real-time
# - Detect unusual activity patterns
# - Generate trading signals
# - Log alerts for manual review
```

## Data Quality and Coverage

### QuantConnect Options Data Benefits

1. **Comprehensive Coverage**: Full options chains for major US equities
2. **High Quality**: Exchange-sourced data with bid/ask spreads
3. **Historical Depth**: Data back to 2010+ for major stocks
4. **Greeks Calculation**: Built-in Greeks and implied volatility
5. **Real-time Access**: Sub-second latency for live algorithms

### Data Validation

The system includes automatic data quality checks:

```python
# Validation performed automatically
validation_results = provider.validate_options_data(options_df)
# Checks: data completeness, reasonable strikes, valid expirations, 
#         non-negative prices, implied volatility ranges
```

## API Integration (Future Enhancement)

For direct API access without algorithms:

```python
# Future: Direct QuantConnect API integration
from quantconnect.api import Api

api = Api()
api.authenticate(user_id, token)

# Get historical options data
options_data = api.get_options_data(
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2020-12-31',
    resolution='daily'
)
```

## Cost Considerations

### QuantConnect Pricing (Approximate)

- **Basic Plan**: $20/month - Limited data access
- **Professional**: $50/month - Full options data access
- **Team**: $100/month - Multiple algorithms, more compute
- **Data Costs**: Options data ~$100/month additional

### Cost Optimization

1. **Targeted Collection**: Only collect data around launch events
2. **Efficient Filtering**: Use strike/expiration filters to reduce data volume
3. **Batch Processing**: Run collection algorithms periodically vs. continuously
4. **Local Caching**: Store collected data locally to avoid re-downloading

## Troubleshooting

### Common Issues

1. **Missing Data**: Check QuantConnect subscription includes options data
2. **API Limits**: Monitor usage against plan limits
3. **Algorithm Timeouts**: Optimize data collection loops
4. **Export Failures**: Verify object store permissions

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger('src.phase2.options_data').setLevel(logging.DEBUG)

# Test QuantConnect connection
provider = QuantConnectOptionsProvider(config)
test_chain = provider.get_options_chain('AAPL', date(2024, 1, 19))
```

## Performance Optimization

### Algorithm Optimization

1. **Selective Processing**: Only process options near launch events
2. **Efficient Filtering**: Use QuantConnect's built-in filters
3. **Batch Exports**: Export data weekly rather than daily
4. **Memory Management**: Clear old data periodically

### Local Processing

1. **Parallel Analysis**: Process multiple tickers concurrently
2. **Data Caching**: Cache QuantConnect data locally
3. **Incremental Updates**: Only fetch new data since last run
4. **Compression**: Use compressed formats for large datasets

## Next Steps

1. **Deploy Collection Algorithm**: Start collecting historical options data
2. **Validate Data Quality**: Compare with existing Yahoo Finance data
3. **Implement Live Signals**: Deploy real-time detection algorithm
4. **Backtest Strategies**: Use collected data for strategy backtesting
5. **API Integration**: Implement direct API access for local analysis

---

For questions or support, refer to:
- [QuantConnect Documentation](https://www.quantconnect.com/docs/)
- [QuantConnect Community](https://www.quantconnect.com/forum/)
- Phase 2 system logs: `results/phase2_run_*/phase2_run.log`