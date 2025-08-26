# Pre-Launch Options Signals: Comprehensive Research Study

**Author:** Evan Baisden  
**Course:** Independent Study in Finance  
**Professor:** Professor Cheah  
**Date:** August 26, 2025  

## Project Overview

This study explores whether unusual options market activity contains information about the outcomes of upcoming technology product launches. The research analyzes 34 major product launches between June 2020 and March 2024, examining options market signals and their predictive power for stock returns and earnings surprises.

## Key Findings

- **No Statistical Significance**: Pre-launch abnormal returns are statistically indistinguishable from zero across all tests
- **Limited Economic Significance**: Gross Sharpe ratios around 0.2 become negative after transaction costs
- **Market Efficiency**: Evidence supports semi-strong form efficiency in options markets
- **Selection Bias**: Deflated Sharpe ratios (-7 to -12) indicate severe overfitting concerns

## Project Structure

```
prelaunch-options-signals/
├── README.md                           # This overview document
├── FINAL_PAPER.pdf                     # Main research paper (PDF)
├── FINAL_PAPER.md                      # Main research paper (Markdown)
├── data/                               # Raw and processed data
│   ├── raw/
│   │   └── options/                    # Original options data
│   └── processed/                      # Cleaned datasets
├── analysis/                           # Analysis code
│   ├── nonparametric_tests.py          # Statistical tests implementation
│   ├── summary_statistics.py           # Summary tables generation
│   ├── cross_sectional_analysis.py     # Regression analysis
│   ├── economic_significance.py        # Trading strategy analysis
│   └── simple_robustness.py            # Robustness checks
├── results/                            # All analysis results
│   ├── final_archive/                  # Core event study results
│   │   ├── final_analysis_results.json # Main 34-event dataset
│   │   └── comprehensive_analysis_*.json
│   ├── statistical_tests/              # Supplementary analysis results
│   │   ├── comprehensive_summary.json  # Validated statistics
│   │   ├── nonparametric_tests.json   # Rank tests, sign tests
│   │   ├── cross_sectional_analysis.json # Regression results
│   │   ├── economic_significance.json  # Sharpe ratios, transaction costs
│   │   ├── robustness_analysis.json   # Robustness checks
│   │   └── comprehensive_analysis_report.md # Executive summary
│   ├── options/                        # Options analysis data
│   ├── equity/                         # Equity analysis results
│   └── earnings/                       # Earnings data
└── documentation/                      # Additional documentation
    ├── methodology_notes.md            # Detailed methodology
    ├── data_sources.md                 # Data collection details
    ├── code_documentation.md           # Code explanations
    ├── best_practices.md               # Research methodology standards
    ├── DATA_DICTIONARY.md              # Variable definitions
    └── methodology_detailed.md         # Comprehensive methodology
```

## Data Summary

- **34 events** across 6 technology companies (Apple, NVIDIA, Microsoft, Tesla, AMD, Sony)
- **92,076 option-chain observations** covering 60 trading days pre-launch
- **610+ quarterly earnings records** for earnings surprise analysis
- **Period:** June 2020 - March 2024

## Key Results Files

### Core Analysis
- `results/final_archive/final_analysis_results.json` - Main event study results
- `results/statistical_tests/comprehensive_summary.json` - Validated paper statistics

### Statistical Tests
- `results/statistical_tests/nonparametric_tests.json` - Corrado rank & sign tests
- `results/statistical_tests/robustness_analysis.json` - Outlier sensitivity analysis

### Economic Analysis
- `results/statistical_tests/economic_significance.json` - Trading strategy performance
- `results/statistical_tests/cross_sectional_analysis.json` - Regression results

## Reproducibility

All analysis code is provided in the `analysis/` directory. To reproduce results:

1. Ensure Python 3.13+ with required packages (numpy, pandas, scipy, scikit-learn)
2. Run analysis scripts in the following order:
   ```bash
   python analysis/nonparametric_tests.py
   python analysis/summary_statistics.py  
   python analysis/cross_sectional_analysis.py
   python analysis/economic_significance.py
   python analysis/simple_robustness.py
   ```

## Academic Contributions

1. **Methodological**: Comprehensive framework combining parametric and non-parametric tests
2. **Empirical**: Large-scale analysis of options market efficiency around product launches  
3. **Practical**: Clear evidence that transaction costs eliminate apparent trading opportunities
4. **Theoretical**: Support for semi-strong form market efficiency in derivative markets

## Software Requirements

- Python 3.13+
- numpy 2.3.2+
- pandas 2.3.1+
- scipy 1.16.1+
- scikit-learn 1.7.1+
- yfinance 0.2.65+

## Contact

**Evan Baisden**  
Independent Study in Finance  
[Contact information as appropriate]

---

*This research was conducted as part of an independent study under the supervision of Professor Cheah. All code and data are available for academic use and verification.*