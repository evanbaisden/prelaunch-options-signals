# Project Summary for Submission

**Student:** Evan Baisden  
**Course:** Independent Study in Finance  
**Professor:** Professor Cheah  
**Submission Date:** August 26, 2025

## Research Question

Do options markets show predictive signals around major technology product launches?

## Key Findings

✅ **No Statistical Significance**: Pre-launch abnormal returns (+0.62%, +0.68%, +0.32%) are statistically indistinguishable from zero across parametric and non-parametric tests

✅ **Limited Economic Significance**: Trading strategies achieve gross Sharpe ratios around 0.2 but become unprofitable after transaction costs

✅ **Market Efficiency**: Evidence supports semi-strong form efficiency in options markets around product launches

✅ **Selection Bias Concerns**: Deflated Sharpe ratios (-7 to -12) indicate severe overfitting in strategy backtests

## Dataset Summary

- **34 product launch events** (June 2020 - March 2024)
- **6 technology companies**: Apple, NVIDIA, Microsoft, Tesla, AMD, Sony
- **92,076 option-chain observations** (60 trading days pre-launch per event)
- **610+ quarterly earnings records**

## Methodology Highlights

### Statistical Rigor
- Event study framework with (-120, -6) estimation windows
- Multiple event windows: (-5,0), (-3,0), (-1,0), (0,+1), (0,+3), (0,+5)
- Parametric (t-tests) and non-parametric tests (Corrado rank, sign tests)
- Bootstrap confidence intervals (1,000 replications)

### Options Anomaly Detection
- Issuer-normalized z-scores for volume, open interest, implied volatility
- Structural indicators: OTM concentration, expiry structure, put-call ratios
- High-anomaly threshold: z-score ≥ 5.0 (3 events identified)

### Cross-sectional Analysis
- Multiple regression specifications with R² progression: 2% → 5% → 34% → 45%
- Company fixed effects dominate options variables in explanatory power
- Robust standard errors clustered by issuer

### Economic Analysis  
- Three trading strategies: long-all, long-short volume, options anomaly
- Transaction cost modeling: retail (0.40%), institutional (0.17%), high-frequency (0.08%)
- Deflated Sharpe ratios to address selection bias

## Academic Contributions

1. **Comprehensive dataset** of technology product launches with options data
2. **Methodological framework** combining parametric and non-parametric tests
3. **Economic significance assessment** with realistic transaction costs
4. **Market efficiency evidence** in derivative markets around corporate events

## Files Submitted

### Main Analysis
- **Research Paper** (PDF format) - Complete study with all findings
- `results/final_archive/final_analysis_results.json` - Core event study dataset

### Statistical Analysis  
- `results/statistical_tests/comprehensive_summary.json` - Validated paper statistics
- `results/statistical_tests/nonparametric_tests.json` - Rank tests, sign tests, bootstrap CIs
- `results/statistical_tests/cross_sectional_analysis.json` - Regression results with controls
- `results/statistical_tests/economic_significance.json` - Trading strategies, Sharpe ratios
- `results/statistical_tests/robustness_analysis.json` - Sensitivity and stability tests

### Supporting Documentation
- `README.md` - Project overview and structure
- `documentation/methodology_notes.md` - Detailed methodology
- `documentation/data_sources.md` - Data collection details  
- `documentation/code_documentation.md` - Code explanations

### Analysis Code (for reproducibility)
- `analysis/nonparametric_tests.py`
- `analysis/summary_statistics.py`
- `analysis/cross_sectional_analysis.py`
- `analysis/economic_significance.py`
- `analysis/simple_robustness.py`

## Verification and Reproducibility

All analysis is reproducible with provided code and data. Key statistics from the paper are validated in the supplementary analysis files. The study follows best practices for financial research including robust standard errors, multiple model specifications, and comprehensive robustness checks.

---

*This project represents a comprehensive investigation into options market efficiency around corporate events, contributing both methodological insights and empirical evidence to the finance literature.*