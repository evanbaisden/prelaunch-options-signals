# Methodology Documentation

## Event Study Framework

### Estimation Window
- **Period**: -120 to -6 trading days relative to launch
- **Market Model**: Simple linear regression against market returns
- **Parameters**: Alpha (intercept), Beta (market sensitivity), R-squared (fit)

### Event Windows Analyzed
- Pre-launch: (-5,0), (-3,0), (-1,0)  
- Post-launch: (0,+1), (0,+3), (0,+5)

### Statistical Tests Applied
1. **Parametric**: Two-sided t-tests
2. **Non-parametric**: 
   - Corrado (1989) rank test
   - Corrado-Zivney (1992) sign test
3. **Bootstrap**: 1,000 replications for confidence intervals

## Options Anomaly Detection

### Z-Score Thresholds
- 90th percentile: 1.645
- 99th percentile: 2.326  
- 99.5th percentile: 2.576
- High anomaly: ≥5.0

### Structural Indicators
- OTM concentration: >65%
- Near-dated expiry: >80%
- Put-call ratio extremes: <0.30 or >1.50
- Bid-ask spread increases: relative changes

## Cross-sectional Analysis

### Model Specifications
1. **Basic**: Beta, R-squared, Volume spike
2. **Options**: Basic + PCR, OTM ratio, High volume anomaly  
3. **Company**: Basic + Firm fixed effects
4. **Full**: All variables + COVID controls + Market cap proxy

### Statistical Adjustments
- Robust standard errors clustered by issuer
- Bootstrap standard errors (1,000 replications)
- Coefficient of variation for stability assessment