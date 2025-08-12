# Event Study Methodology

This document outlines the statistical methodology used for analyzing technology product launch effects on equity markets.

## Research Design

**Framework**: Market-adjusted event study methodology following Brown & Warner (1985) and MacKinlay (1997).

**Hypothesis Testing**: Tests the null hypothesis that product launch announcements do not generate systematic abnormal returns in equity markets.

**Sample**: 34 major technology product launches from six companies (Apple, NVIDIA, Microsoft, Tesla, AMD, Sony) spanning 2020-2024.

## Data Sources

### Primary Data
- **Equity Data**: Yahoo Finance via yfinance API
  - Daily OHLCV data with adjustments for splits and dividends
  - Market benchmark: S&P 500 Index (^GSPC)
  - Coverage: Technology sector companies

### Event Data
- **Product Launches**: Manually curated dataset of announcement dates
  - Source: Official company press releases and financial news
  - Categories: Consumer electronics, semiconductor hardware, gaming platforms, electric vehicles
  - Quality Control: Cross-referenced with multiple sources for date accuracy

## Event Study Methodology

### Market Model Estimation
- **Estimation Window**: ~83 trading days ending 30 days before announcement (range: 81-85 days)
- **Model**: CAPM regression R_stock = α + β × R_market + ε
- **Parameters**: Estimated via ordinary least squares (OLS)
- **Quality Control**: Minimum 30 days of data required for estimation

### Abnormal Return Calculation
- **Formula**: AR_t = R_stock_t - (α̂ + β̂ × R_market_t)
- **Event Windows**: Multiple windows from -5 to +5 days relative to announcement
- **Cumulative Abnormal Returns**: Sum of daily abnormal returns within each window

### Statistical Testing
- **Primary Test**: One-sample t-test against null hypothesis of zero abnormal returns
- **Confidence Intervals**: 95% CI using Student's t-distribution
- **Significance Level**: α = 0.05 for statistical inference
- **Power Analysis**: Cohen's d effect size calculation for adequacy assessment

## Analysis Parameters

### Event Windows
- **Primary Analysis**: 5-day pre-announcement window (-5 to 0 days)
- **Additional Windows**: -3:0, -1:0, 0:+1, 0:+3, 0:+5 days
- **Rationale**: Captures potential information leakage and market reaction patterns

### Data Quality Controls
- **Missing Data**: Listwise deletion for incomplete observations
- **Outlier Detection**: Examination of standardized residuals
- **Market Model Fit**: R-squared assessment for explanatory power
- **Temporal Alignment**: Events matched to nearest trading day

## Statistical Framework

### Assumptions
- **Independence**: Product launch events treated as independent observations
- **Normality**: Daily returns approximately normally distributed
- **Stationarity**: Market model parameters stable over estimation period
- **Market Efficiency**: Prices reflect available information

### Hypothesis Testing
- **Null Hypothesis**: E[AR_t] = 0 (no systematic abnormal returns)
- **Alternative Hypothesis**: E[AR_t] ≠ 0 (systematic abnormal returns exist)
- **Test Statistic**: t = (mean_AR - 0) / (std_AR / √n)
- **Critical Values**: ±1.96 for two-tailed test at 5% significance level

### Power Analysis
- **Sample Size**: N = 34 events
- **Effect Size**: Cohen's d = |mean_AR| / std_AR
- **Power Calculation**: Ability to detect economically meaningful effects (≥2%)
- **Type II Error**: Beta risk assessment for null finding interpretation

## Output Specifications

### Primary Results
- **Statistical Summary**: Sample size, mean, standard deviation, confidence intervals
- **Significance Tests**: T-statistics, p-values, significance flags
- **Effect Sizes**: Cohen's d for practical significance assessment
- **Company Breakdown**: Subgroup analysis by firm

### Detailed Output
- **Event-Level Results**: Individual abnormal returns and statistical metrics
- **Market Model Parameters**: Alpha, beta, R-squared for each event
- **Data Quality Indicators**: Estimation period length and model fit statistics
- **Robustness Checks**: Raw returns comparison and volume analysis

## Limitations

### Methodological Constraints
- **Daily Frequency**: Cannot capture intraday market reactions
- **Event Isolation**: Potential confounding with earnings or other corporate events
- **Sample Concentration**: Focus on technology sector limits generalizability
- **Time Period**: 2020-2024 represents specific market conditions

### Statistical Limitations
- **Multiple Testing**: No adjustment for multiple event window comparisons
- **Cross-Sectional Dependence**: Potential correlation across events not modeled
- **Non-Normality**: Return distributions may exhibit fat tails or skewness
- **Regime Changes**: Market efficiency may vary over time and conditions

## Academic Standards

### Reproducibility
- **Code Availability**: Complete analysis framework documented and executable
- **Data Sources**: Publicly available financial data from established providers
- **Parameter Transparency**: All analysis parameters explicitly specified
- **Version Control**: Results timestamped and archived for verification

### Validation
- **Cross-Validation**: Results verified with alternative data sources where available
- **Sensitivity Analysis**: Robustness to estimation window and parameter choices
- **Literature Consistency**: Methodology follows established academic conventions
- **Statistical Software**: Analysis conducted using Python scientific computing stack

---

**Last Updated**: August 2025  
**Version**: 2.0 (Final)  
**Compliance**: Academic research standards for financial event studies