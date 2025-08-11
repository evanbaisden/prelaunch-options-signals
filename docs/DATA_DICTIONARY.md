# Data Dictionary

This document describes the data schemas and field definitions used throughout the prelaunch options signals analysis.

## Raw Data Files

### Stock Price CSV Format (`data/raw/*.csv`)

**File naming convention**: `{company}_{product}_raw.csv`

**Structure**:
```
Row 1: Price,Adj Close,Close,High,Low,Open,Volume
Row 2: Ticker,{TICKER},{TICKER},{TICKER},{TICKER},{TICKER},{TICKER}
Row 3: Date,,,,,,
Row 4+: {YYYY-MM-DD},{float},{float},{float},{float},{float},{int}
```

**Fields**:
- `Date`: Trading date (YYYY-MM-DD format)
- `Adj Close`: Split/dividend adjusted closing price (USD)
- `Close`: Raw closing price (USD)
- `High`: Daily high price (USD)
- `Low`: Daily low price (USD)  
- `Open`: Opening price (USD)
- `Volume`: Trading volume (shares)

**Data source**: Yahoo Finance (yfinance library)
**Coverage**: Daily OHLCV data for analysis periods around product launches

### Events Master CSV (`data/processed/events_master.csv`)

**Schema**:
```csv
name,company,ticker,announcement,release,next_earnings,category
```

**Fields**:
- `name`: Product name (e.g., "iPhone 12", "RTX 30 Series")
- `company`: Company name (Apple, Microsoft, NVIDIA)
- `ticker`: Stock ticker symbol (AAPL, MSFT, NVDA)
- `announcement`: Product announcement date (YYYY-MM-DD)
- `release`: Product release/launch date (YYYY-MM-DD)
- `next_earnings`: Next earnings announcement after product launch (YYYY-MM-DD or null)
- `category`: Product category (Consumer Hardware, Gaming Hardware, Semiconductor Hardware)

## Analysis Output Files

### Phase 1 Summary CSV (`results/phase1_summary_stats_{timestamp}.csv`)

**Schema**:
```csv
product,company,ticker,announcement_date,release_date,category,baseline_avg_return,baseline_avg_volume,baseline_volatility,announcement_5day_return,announcement_volume_spike,announcement_z_score,release_5day_return,release_volume_spike,release_z_score,pre_announce_avg_return,announce_to_release_avg_return,post_release_avg_return,pre_announce_avg_volume,announce_to_release_avg_volume,post_release_avg_volume,data_quality_score,analysis_timestamp,notes
```

**Field Definitions**:

#### Event Information
- `product`: Product name
- `company`: Company name  
- `ticker`: Stock ticker symbol
- `announcement_date`: Product announcement date (YYYY-MM-DD)
- `release_date`: Product release date (YYYY-MM-DD)
- `category`: Product category

#### Baseline Metrics (60 trading days pre-announcement)
- `baseline_avg_return`: Average daily return during baseline period (decimal, e.g., 0.0015 = 0.15%)
- `baseline_avg_volume`: Average daily volume during baseline period (shares)
- `baseline_volatility`: Standard deviation of daily returns during baseline period

#### Event-Specific Signals
- `announcement_5day_return`: Cumulative return from t-5 to t+0 around announcement (decimal)
- `announcement_volume_spike`: Volume spike percentage during announcement period (decimal)
- `announcement_z_score`: Z-score of announcement period volume vs baseline
- `release_5day_return`: Cumulative return from t-5 to t+0 around release (decimal)  
- `release_volume_spike`: Volume spike percentage during release period (decimal)
- `release_z_score`: Z-score of release period volume vs baseline

#### Period Averages
- `pre_announce_avg_return`: Average daily return 60 days before announcement (decimal)
- `announce_to_release_avg_return`: Average daily return from announcement to release (decimal)
- `post_release_avg_return`: Average daily return 30 days after release (decimal)
- `pre_announce_avg_volume`: Average daily volume 60 days before announcement (shares)
- `announce_to_release_avg_volume`: Average daily volume from announcement to release (shares)
- `post_release_avg_volume`: Average daily volume 30 days after release (shares)

#### Quality Control
- `data_quality_score`: Data quality score from 0.0 to 1.0 based on completeness and sanity checks
- `analysis_timestamp`: Analysis execution timestamp (ISO format)
- `notes`: Additional notes or warnings about the analysis

### Phase 1 Metadata JSON (`results/phase1_summary_stats_{timestamp}_metadata.json`)

**Schema**:
```json
{
  "analysis_timestamp": "2024-01-01T12:00:00.000000",
  "parameters": {
    "baseline_days": 60,
    "signal_window_announce": [5, 20],
    "signal_window_release": [5, 10],
    "z_thresholds": {
      "low": 1.645,
      "med": 2.326,
      "high": 2.576,
      "extreme": 5.0
    }
  },
  "data_sources": ["yfinance"],
  "calculation_definitions": {
    "announcement_5day_return": "Cumulative return from close[t-5] to close[t+0] on announcement day",
    "release_5day_return": "Cumulative return from close[t-5] to close[t+0] on release day",
    "volume_spike_pct": "(volume[period] / baseline_mean) - 1, where baseline = 20-day MA",
    "pre_announce_return": "Average daily return 60 trading days before announcement",
    "announce_to_release_return": "Average daily return from announcement to release day",
    "post_release_return": "Average daily return 30 trading days after release",
    "data_source": "yfinance adjusted close prices (split/dividend adjusted), raw volumes"
  }
}
```

## Statistical Definitions

### Return Calculations
- **Daily Return**: `(price[t] - price[t-1]) / price[t-1]`
- **5-Day Cumulative Return**: `(price[t+0] - price[t-5]) / price[t-5]`
- **Average Daily Return**: Arithmetic mean of daily returns over specified period

### Volume Metrics
- **Baseline Volume**: 20-day rolling average volume
- **Volume Spike %**: `(period_volume - baseline_volume) / baseline_volume * 100`
- **Volume Z-Score**: `(volume_ratio - mean(volume_ratio)) / std(volume_ratio)` where volume_ratio = volume / baseline_volume

### Statistical Thresholds
- **Z-Score Low (1.645)**: 95th percentile (one-tailed test)
- **Z-Score Medium (2.326)**: 99th percentile (one-tailed test)
- **Z-Score High (2.576)**: 99% confidence (two-tailed test) / 99.5% one-tailed
- **Z-Score Extreme (5.0)**: Extreme outliers

## Data Quality Checks

### Automated Validation
- **Required Columns**: Presence of Date, Adj Close, Volume, High, Low, Open
- **Completeness**: No missing values in critical fields
- **Sanity Checks**: No zero/negative prices or volumes
- **Date Continuity**: No gaps > 7 days in trading dates
- **Price Movements**: No single-day moves > 50%
- **Volume Outliers**: < 1% of observations exceed 100x median volume

### Quality Score Calculation
Data quality score (0.0 to 1.0) based on percentage of quality checks passed:
- 1.0: All checks passed
- 0.8-0.99: Minor issues (some gaps or outliers)
- 0.6-0.79: Moderate issues (missing data or extreme values)
- < 0.6: Major data quality concerns

## File Naming Conventions

### Raw Data
- **Stock Data**: `{company}_{product}_raw.csv`
- **Events**: `events_master.csv`

### Results
- **Summary Stats**: `phase1_summary_stats_{YYYYMMDD_HHMMSS}.csv`
- **Metadata**: `phase1_summary_stats_{YYYYMMDD_HHMMSS}_metadata.json`
- **Visualizations**: `{chart_type}_{timestamp}.{format}`

### Timestamps
All timestamps use format: `YYYYMMDD_HHMMSS` (e.g., `20240101_120000`)

## Version History

- **v1.0**: Initial data dictionary for Phase 1 analysis
- **v1.1**: Added quality control metrics and validation rules
- **v1.2**: Standardized field naming and statistical definitions