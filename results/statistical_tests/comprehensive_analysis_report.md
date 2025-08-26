# Pre-Launch Options Signals: Comprehensive Statistical Analysis
## Supplementary Analyses Building on Core Event Study

**Generated:** August 26, 2025  
**Analysis Period:** June 2020 - March 2024  
**Sample Size:** 34 technology product launch events  

---

## Executive Summary

This report presents comprehensive supplementary analyses that build on the core event study results. The analyses confirm the paper's main findings while providing additional robustness checks, non-parametric tests, and economic significance assessments.

### Key Validation Results
- **Sample Statistics Confirmed**: Mean pre-launch abnormal returns of +0.62% (-5,0), +0.68% (-3,0), and +0.32% (-1,0) match paper exactly
- **Earnings Surprise Rate**: 82.4% positive earnings surprises confirmed
- **High Anomaly Events**: 3 events identified (iPhone 12, Tesla Model 3 Highland, Vision Pro)
- **Statistical Significance**: Results remain non-significant across parametric and non-parametric tests

---

## 1. Non-Parametric Statistical Tests

### Corrado Rank Test Results
The Corrado (1989) rank test provides distribution-free testing of abnormal returns:

| Window | Rank Test Statistic | P-value | Significant |
|--------|-------------------|---------|-------------|
| (-5,0) | 0.000 | 1.000 | No |
| (-3,0) | 0.000 | 1.000 | No |
| (-1,0) | 0.000 | 1.000 | No |

### Sign Test Results
The Corrado-Zivney (1992) sign test examines the proportion of positive abnormal returns:

| Window | Proportion Positive | Test Statistic | P-value | Significant |
|--------|-------------------|----------------|---------|-------------|
| (-5,0) | 50.0% | 0.000 | 1.000 | No |
| (-3,0) | 50.0% | 0.000 | 1.000 | No |
| (-1,0) | 44.1% | -0.688 | 0.493 | No |

### Bootstrap Confidence Intervals
1,000 bootstrap replications provide bias-corrected confidence intervals:
- Pre-launch (-5,0): 95% CI = [-0.56%, +1.81%]
- Pre-launch (-3,0): 95% CI = [-0.37%, +1.73%]
- Pre-launch (-1,0): 95% CI = [-0.71%, +1.36%]

**Finding**: Non-parametric tests confirm the absence of statistically significant pre-launch abnormal returns, consistent with semi-strong market efficiency.

---

## 2. Robustness Analysis

### Outlier Sensitivity
Testing robustness to statistical outliers:

| Exclusion Rule | Events Removed | Mean AR (-5,0) | Impact |
|----------------|----------------|----------------|---------|
| Full Sample | 0 | +0.621% | Baseline |
| Exclude >2σ | 1 | +0.361% | -42% |
| Exclude >2.5σ | 0 | +0.621% | No change |
| Exclude >3σ | 0 | +0.621% | No change |

### Cross-Sectional Stability
Results vary significantly across companies:

| Company | Events | Mean AR (-5,0) | Standard Deviation |
|---------|--------|----------------|-------------------|
| Sony | 3 | +5.24% | High |
| NVIDIA | 7 | +1.99% | Medium |
| AMD | 5 | +1.29% | Medium |
| Tesla | 5 | +0.29% | Medium |
| Microsoft | 5 | -1.06% | Medium |
| Apple | 9 | -1.24% | High |

### Bootstrap Stability Assessment
- Coefficient of variation: 1.009
- Results classification: **Unstable**
- Interpretation: High variability across bootstrap samples suggests fragile statistical relationships

**Finding**: Results are sensitive to outliers and exhibit substantial heterogeneity across firms, indicating limited generalizability.

---

## 3. Enhanced Cross-Sectional Analysis

### Regression Results with Control Variables

**Model Specifications Tested:**
1. **Basic**: Beta, R-squared, Volume spike
2. **Options**: Basic + Put-call ratio, OTM ratio, High volume anomaly
3. **Company Controls**: Basic + Company dummies
4. **Full Model**: All variables + COVID period + Market cap proxy

### Key Regression Findings

| Model | R² | Observations | Key Significant Variables |
|-------|----|--------------|-----------------------------|
| Basic | 0.022 | 34 | None strongly significant |
| Options | 0.053 | 34 | OTM ratio (+), High volume anomaly (-) |
| Company Controls | 0.344 | 34 | Apple (-), Microsoft (-) |
| **Full Model** | **0.446** | **34** | **Put-call ratio (-), NVIDIA (+)** |

### Cross-Sectional Patterns
- **Company Effects**: Strong firm-specific heterogeneity (R² increases from 2.2% to 34.4%)
- **Options Variables**: Put-call ratio negatively predicts returns (consistent with contrarian effect)
- **Volume Effects**: Mixed results across specifications

**Finding**: Firm-specific effects dominate. Lower put-call ratios (more call buying) weakly associated with higher abnormal returns.

---

## 4. Economic Significance Analysis

### Trading Strategy Performance (Gross Returns)

| Strategy | Annual Return | Sharpe Ratio | Win Rate | Trades |
|----------|---------------|--------------|----------|---------|
| **Long All Events** | **+3.7%** | **0.20** | **50.0%** | 34 |
| Long-Short Volume | +0.0% | -0.23 | 47.1% | 34 |
| Options Anomaly | +0.2% | -0.84 | 11.5% | 26 |

### Transaction Cost Analysis

| Investor Type | Cost per Trade | Annual Cost | Net Sharpe (Long All) |
|---------------|----------------|-------------|---------------------|
| **Retail Investor** | 0.40% | 2.4% | **-0.08** |
| **Institutional** | 0.17% | 1.0% | **+0.08** |
| **High Frequency** | 0.08% | 0.5% | **+0.15** |

### Deflated Sharpe Ratios (Selection Bias Adjusted)
Following Bailey & López de Prado (2014):
- Long All: -7.46 (severely deflated)
- Long-Short Volume: -9.96 (severely deflated)
- Options Anomaly: -11.82 (severely deflated)

### Economic Interpretation
- **Gross Profitability**: Marginal positive returns from simple long strategy
- **Net Profitability**: Depends critically on transaction costs
- **Institutional Advantage**: Lower costs enable modest net positive returns
- **Selection Bias**: Severe deflation of Sharpe ratios suggests overfitting risks

**Finding**: Economic significance is limited. Only institutional investors with low transaction costs might achieve modest net profits.

---

## 5. Heterogeneity Analysis

### Product Category Analysis
| Category | Events | Mean AR (-5,0) | Key Companies |
|----------|--------|----------------|---------------|
| Consumer Hardware | 8 | -0.45% | Apple, Sony |
| Semiconductor Hardware | 7 | +1.85% | NVIDIA, AMD |
| AI Hardware | 3 | +3.24% | NVIDIA, AMD |
| Gaming Hardware | 3 | +1.63% | Microsoft, Sony |
| Software Platform | 2 | -0.53% | Microsoft |

### Time Period Analysis
- **Pre-COVID** (< March 2020): Limited sample
- **COVID Period** (Mar 2020 - Jun 2021): Mixed results
- **Post-COVID** (> Jun 2021): Continuation of patterns

**Finding**: Semiconductor and AI hardware announcements show stronger positive abnormal returns, suggesting sector-specific information effects.

---

## 6. Hypothesis Testing Summary

### Primary Hypotheses from Paper

| Hypothesis | Test Method | Result | P-value | Conclusion |
|------------|-------------|--------|---------|-----------|
| H1: Options volume predicts returns | Correlation | -0.219 | n/a | Weak negative relationship |
| H2: Put-call ratios predict returns | Correlation | -0.018 | n/a | No relationship |
| H3: High anomaly events differ | T-test | 0.312 | 0.312 | Not significant |

### Additional Tests
- **Earnings Surprise Rate**: 82.4% vs 50% null (p < 0.001), vs 70% tech baseline (p = 0.26)
- **Market Efficiency**: No significant abnormal returns across all tests
- **Cross-Sectional Predictability**: Limited explanatory power (R² < 45%)

---

## 7. Limitations and Caveats

### Statistical Limitations
1. **Small Sample Size**: 34 events limits statistical power
2. **Multiple Testing**: Deflated Sharpe ratios suggest overfitting concerns
3. **Heterogeneity**: Large firm and time-period effects reduce generalizability
4. **Outlier Sensitivity**: Results change with single outlier removal

### Economic Limitations
1. **Transaction Costs**: Erode most gross profits for typical investors
2. **Implementation Challenges**: Real-time options anomaly detection is complex
3. **Capacity Constraints**: Strategies may not scale to large capital
4. **Selection Bias**: Ex-post strategy selection inflates apparent performance

### Data Limitations
1. **Options Data Quality**: Some missing or incomplete observations
2. **Survivorship Bias**: Focus on successful tech companies
3. **Event Definition**: Subjective launch date selection
4. **Market Microstructure**: Ignores intraday timing and execution effects

---

## 8. Conclusions and Implications

### Academic Implications
1. **Market Efficiency**: Evidence supports semi-strong form efficiency in options markets
2. **Information Processing**: Markets appear to incorporate launch information efficiently
3. **Cross-Sectional Variation**: Firm-specific factors dominate systematic patterns
4. **Methodology**: Non-parametric tests provide valuable robustness checks

### Practical Implications
1. **Trading Strategy Viability**: Limited for retail investors due to transaction costs
2. **Institutional Opportunities**: Modest profits possible with sophisticated implementation
3. **Risk Management**: High variability requires careful position sizing
4. **Market Timing**: Pre-launch windows show no consistent exploitable patterns

### Future Research Directions
1. **Larger Samples**: Expand to other sectors and smaller companies
2. **High-Frequency Data**: Examine intraday options flow patterns
3. **Machine Learning**: Apply advanced pattern recognition techniques
4. **International Markets**: Test patterns in global technology markets

---

## 8. Technical Appendices

### Appendix A: Complete Summary Statistics
*[Detailed statistical tables available in comprehensive_summary.json]*

### Appendix B: Regression Output Details  
*[Full regression results available in cross_sectional_analysis.json]*

### Appendix C: Bootstrap Distributions
*[Bootstrap samples and confidence intervals in nonparametric_tests.json]*

### Appendix D: Economic Calculations
*[Detailed Sharpe ratio and transaction cost calculations in economic_significance.json]*

---

**Final Assessment**: This comprehensive analysis confirms the paper's core findings while revealing the fragility of apparent abnormal returns. The evidence supports market efficiency with limited exploitable opportunities, particularly after accounting for realistic transaction costs and selection biases. The work provides a solid foundation for understanding options market behavior around technology product launches.

---
*Analysis completed using Python 3.13 with numpy, pandas, scipy, and scikit-learn*  
*All code and data available in the analysis/ directory*