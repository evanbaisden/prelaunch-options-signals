# Code Documentation

## Analysis Scripts Overview

### Core Analysis Scripts

#### 1. `nonparametric_tests.py`
**Purpose**: Implements Corrado rank test, sign test, and bootstrap confidence intervals

**Key Functions**:
- `corrado_rank_test()`: Implements Corrado (1989) rank test
- `sign_test()`: Implements Corrado-Zivney (1992) sign test  
- `bootstrap_confidence_interval()`: 1,000-replication bootstrap
- `run_comprehensive_tests()`: Main execution function

**Output**: `results/statistical_tests/nonparametric_tests.json`

#### 2. `summary_statistics.py`
**Purpose**: Generates summary tables and validates paper statistics

**Key Functions**:
- `generate_appendix_a_table()`: Creates Appendix A descriptive statistics
- `generate_earnings_surprise_analysis()`: Earnings surprise rate analysis
- `calculate_high_anomaly_events()`: Identifies high-anomaly events
- `create_comprehensive_summary()`: Main summary generation

**Output**: `results/statistical_tests/comprehensive_summary.json`

#### 3. `cross_sectional_analysis.py`
**Purpose**: Regression analysis with control variables and heterogeneity tests

**Key Functions**:
- `create_regression_dataset()`: Builds regression dataset with controls
- `run_baseline_regressions()`: Multiple model specifications
- `heterogeneity_analysis()`: Analysis by company, category, time period
- `test_key_hypotheses()`: Specific hypothesis testing

**Output**: `results/statistical_tests/cross_sectional_analysis.json`

#### 4. `economic_significance.py`
**Purpose**: Trading strategy analysis, Sharpe ratios, transaction costs

**Key Functions**:
- `calculate_trading_strategy_returns()`: Strategy return calculation
- `calculate_sharpe_ratios()`: Performance metrics including deflated Sharpe
- `calculate_transaction_costs()`: Cost modeling across investor types
- `net_performance_analysis()`: Post-cost performance assessment

**Output**: `results/statistical_tests/economic_significance.json`

#### 5. `simple_robustness.py`
**Purpose**: Robustness checks and sensitivity analysis

**Key Functions**:
- `outlier_sensitivity_analysis()`: Tests sensitivity to outliers
- `cross_sectional_robustness()`: Stability across subsamples
- `bootstrap_stability_test()`: Bootstrap-based stability assessment
- `alternative_estimation_windows()`: Window sensitivity analysis

**Output**: `results/statistical_tests/robustness_analysis.json`

## Execution Order

Run scripts in the following sequence for complete analysis:

```bash
1. python analysis/nonparametric_tests.py
2. python analysis/summary_statistics.py
3. python analysis/cross_sectional_analysis.py  
4. python analysis/economic_significance.py
5. python analysis/simple_robustness.py
```

## Dependencies

### Required Python Packages
- `numpy >= 2.3.2`: Numerical computations
- `pandas >= 2.3.1`: Data manipulation
- `scipy >= 1.16.1`: Statistical functions
- `scikit-learn >= 1.7.1`: Machine learning utilities
- `json`: Data serialization
- `pathlib`: File path handling
- `datetime`: Date/time utilities

### Input Data Requirements
- `results/final_archive/final_analysis_results.json`: Main event study results
- `results/options/options_analysis_data.json`: Options anomaly data (optional)
- `results/earnings/earnings_timing_summary.json`: Earnings data (optional)

## Output File Structure

All analysis outputs are saved in `results/statistical_tests/`:

- **Statistical Tests**: Non-parametric test results, bootstrap CIs
- **Summary Statistics**: Paper validation, descriptive tables  
- **Regression Analysis**: Cross-sectional models, R² progression
- **Economic Analysis**: Sharpe ratios, transaction costs, trading performance
- **Robustness Checks**: Sensitivity analysis, stability assessment

## Error Handling

Scripts include error handling for:
- Missing input files (graceful degradation)
- Data quality issues (NaN handling)
- JSON serialization (type conversion)
- Statistical edge cases (empty samples)

## Reproducibility Notes

- All random seeds are set (seed=42) for reproducible bootstrap results
- Analysis parameters are documented in output metadata
- Intermediate results are saved for debugging and verification