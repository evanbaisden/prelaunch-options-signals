# Prelaunch Options Signals Analysis - Methodology

This document outlines the analytical methodology used for detecting trading signals around technology product launch events.

## Table of Contents

- [Data Sources](#data-sources)
- [Baseline Window Rationale](#baseline-window-rationale)
- [Statistical Significance Thresholds](#statistical-significance-thresholds)
- [Event Windows](#event-windows)
- [Bootstrap Validation](#bootstrap-validation)
- [Output Structure](#output-structure)
- [Limitations and Assumptions](#limitations-and-assumptions)

## Data Sources

### Primary Data
- **Stock Price Data**: Historical OHLCV (Open, High, Low, Close, Volume) data
  - Source: Yahoo Finance (via yfinance library)
  - Frequency: Daily
  - Adjustment: Split and dividend adjusted closing prices
  - Coverage: Technology companies (AAPL, NVDA, MSFT)

### Event Data
- **Product Launch Events**: Manually curated dataset of technology product announcements and releases
  - Format: CSV with announcement dates, release dates, and next earnings dates
  - Coverage: Major consumer technology products (iPhone, RTX GPUs, Xbox consoles)
  - Validation: Cross-referenced with official company announcements

### Data Quality Considerations
- Missing data handling: Forward-fill for gaps < 3 days, exclude for longer gaps
- Trading day alignment: Events matched to nearest trading day
- Corporate actions: Prices adjusted for splits and dividends
- Volume normalization: Raw volume data used without adjustment

## Baseline Window Rationale

The baseline period establishes "normal" trading behavior before product launch events.

### Window Options
- **30 Days**: Captures recent market conditions, sensitive to short-term volatility
- **60 Days**: Balances recency with statistical stability (default)
- **90 Days**: Provides robust statistical baseline, may include outdated market conditions

### Selection Criteria
The 60-day default baseline provides:
- Sufficient sample size for statistical validity (n ≥ 30 trading days)
- Seasonal adjustment (captures ~3 months of market behavior)
- Excludes prior earnings cycles that might confound results
- Balances recency vs. stability trade-off

### Implementation Notes
- Baseline calculated as trailing N trading days before announcement
- Excludes weekends, holidays, and market closures
- Volume baseline uses rolling 20-day moving average within baseline period
- Returns baseline calculated as mean daily return over baseline period

## Statistical Significance Thresholds

Z-score thresholds determine statistical significance of volume and price anomalies.

### Threshold Levels
- **Low (1.645)**: 90% confidence level (one-tailed test)
  - Use case: Initial screening for potential signals
  - Interpretation: 10% probability of false positive
  
- **Medium (2.326)**: 99% confidence level (one-tailed test)  
  - Use case: Standard significance testing (default)
  - Interpretation: 1% probability of false positive
  
- **High (2.576)**: 99% confidence level (two-tailed test)
  - Use case: Conservative signal detection
  - Interpretation: 0.5% probability of false positive per tail

### Statistical Assumptions
- Normal distribution of baseline metrics (tested via Shapiro-Wilk)
- Independent observations (no autocorrelation assumption)
- Stationary baseline period (no structural breaks)
- Homoscedasticity (constant variance over baseline period)

### Implementation Notes
- Z-scores calculated as: (observed - baseline_mean) / baseline_std
- Volume Z-scores use log-transformed data for normality
- Multiple testing correction not applied (exploratory analysis)

## Event Windows

Event windows define the time periods around product launches for analysis.

### Window Specifications
- **±1 Day**: Captures immediate market reaction
  - Use case: High-frequency signal detection
  - Rationale: News dissemination and algorithmic trading effects
  
- **±2 Days**: Accounts for information processing lag
  - Use case: Short-term momentum analysis  
  - Rationale: Human trader reaction time and settlement cycles
  
- **±5 Days**: Captures extended market adjustment
  - Use case: Medium-term trend analysis (default)
  - Rationale: Full price discovery and position adjustment period

### Window Selection Rationale
- Pre-event period: Captures potential information leakage or anticipation
- Post-event period: Measures market reaction and price adjustment
- Symmetric windows: Control for general market volatility
- Business day alignment: Events matched to nearest trading day

### Implementation Notes
- Windows calculated in trading days (exclude weekends/holidays)
- Event date (T=0) included in analysis
- Overlapping windows handled via separate analysis runs
- Weekend events shifted to next trading day

## Bootstrap Validation

Bootstrap resampling provides robust statistical inference for small sample sizes.

### Methodology (Planned)
- **Sample Size**: N=1000 bootstrap replications
- **Resampling Strategy**: Stratified by company and time period
- **Confidence Intervals**: 95% bias-corrected and accelerated (BCa)
- **Test Statistics**: Mean returns, volume ratios, correlation coefficients

### Applications (Future Implementation)
- Validate baseline parameter selection (30/60/90 day comparison)
- Test robustness of Z-score thresholds
- Estimate confidence intervals for event window effects
- Control for multiple comparisons across products/events

### Limitations
- Assumes exchangeability of observations within strata
- May not capture regime changes or structural breaks
- Bootstrap validity requires sufficient sample diversity

## Output Structure

Analysis results are saved in timestamped directory structure for reproducibility.

### Directory Structure
```
results/
└── run_YYYYMMDD_HHMMSS/
    ├── run.log                         # Execution log
    ├── phase1_summary.csv              # Main results table
    ├── phase1_summary_metadata.json    # Configuration and metadata
    ├── volume_summary.png              # Volume spike overview
    ├── volume_analysis.png             # Detailed volume analysis  
    └── returns_summary.png             # Returns comparison chart
```

### Results Table Schema
Key metrics included in `phase1_summary.csv`:
- **Event Information**: Product name, company, dates
- **Baseline Metrics**: Average returns, volume, volatility
- **Signal Metrics**: Z-scores, anomaly flags, significance levels
- **Period Returns**: Pre-announcement, announcement-to-release, post-release
- **Quality Metrics**: Data completeness, calculation confidence

### Visualization Artifacts
- **Volume Summary**: Bar charts of volume spikes by product
- **Volume Analysis**: Time series of volume patterns around events
- **Returns Summary**: Comparative returns across event windows

### Metadata Tracking
Configuration snapshot includes:
- Analysis parameters (baseline days, thresholds, windows)
- Data sources and file paths
- Statistical assumptions and methods
- Execution timestamp and environment details

## Limitations and Assumptions

### Data Limitations
- **Survivorship Bias**: Analysis limited to successful/current products
- **Sample Size**: Small number of events per company/product category
- **Time Period**: Limited historical coverage (varies by product)
- **Market Conditions**: Results may not generalize across market regimes

### Methodological Assumptions
- **Efficient Markets**: Price changes reflect information processing
- **Event Isolation**: Product launches are primary information events
- **Linear Relationships**: Z-score methodology assumes linear signal detection
- **Independence**: Events treated as independent observations

### Statistical Caveats
- **Multiple Testing**: No correction for multiple comparisons
- **Outlier Sensitivity**: Z-score methods sensitive to extreme observations
- **Distribution Assumptions**: Normality assumptions may not hold for all metrics
- **Temporal Dependence**: Autocorrelation in financial time series not fully addressed

### Future Enhancements
- Robust statistical methods for non-normal distributions
- Time-varying volatility models (GARCH, etc.)
- Cross-sectional analysis across technology sectors
- Integration with options market data for signal validation
- Machine learning approaches for pattern recognition

---

*Last Updated: December 2024*
*Version: 1.0*