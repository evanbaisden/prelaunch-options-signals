# Pre-Launch Options Signals: Event Study Analysis

**An empirical examination of technology product launch effects on equity and options markets**

## 🎯 Research Objective

This study investigates whether major technology product launches create systematic abnormal returns in equity markets and corresponding signals in options markets. Using comprehensive event study methodology, the analysis examines **34 product launch events** across six major technology companies from 2020-2024.

## 🔬 Methodology

**Event Study Framework**: Market-adjusted abnormal returns calculated using CAPM regression with 120-day estimation windows. Statistical significance tested using one-sample t-tests with appropriate confidence intervals and power analysis.

**Sample**: 34 major product launches from Apple, NVIDIA, Microsoft, Tesla, AMD, and Sony.

**Data Sources**: 
- Equity data: Yahoo Finance API
- Options data: Alpha Vantage Historical Options API  
- Market benchmark: S&P 500 Index

## 📊 Key Findings

**Equity Results**: No statistically significant abnormal returns detected in the 5-day pre-announcement window (p=0.3121, N=34 events). Mean abnormal return of 0.62% with 95% CI [-0.61%, 1.85%].

**Options Results**: 44,206 contracts analyzed with call-dominated flow (P/C ratio 0.55) and high volume activity (1.29M avg contracts/event).

**Market Efficiency**: Results support semi-strong form market efficiency for anticipated technology product launches.

## 🚀 Quick Start

### Run Complete Analysis
```bash
# Install dependencies
pip install -r requirements.txt

# Run comprehensive analysis (34 events)
python src/comprehensive_analysis.py

# View results
cat results/final_analysis_results.csv
```

### Expected Output
- Console: Real-time analysis progress for all 34 events
- `results/final_analysis_results.csv`: Statistical results
- `results/final_research_report.md`: Complete academic report

## 📁 Project Structure

```
prelaunch-options-signals/
├── src/
│   ├── comprehensive_analysis.py     # Main analysis script (RUN THIS)
│   ├── phase1/                       # Equity analysis components
│   ├── phase2/                       # Options analysis components
│   └── common/                       # Shared utilities
├── data/
│   ├── processed/events_master.csv   # 34 events dataset
│   └── raw/options/                  # Historical options data
├── results/                          # Analysis outputs
├── docs/                            # Methodology documentation
├── tests/                           # Test suite
└── requirements.txt                 # Dependencies
```

## 🔍 Data Coverage

**Events**: 34 carefully curated technology product launches (2020-2024)
**Companies**: Apple, NVIDIA, Microsoft, Tesla, AMD, Sony
**Options Data**: 26/34 events have complete historical options chains
**Equity Data**: 100% coverage via Yahoo Finance API

## 📈 Academic Standards

**Statistical Rigor**: Follows Brown & Warner (1985) event study methodology
**Power Analysis**: N=34 provides adequate power for medium effect sizes
**Reproducibility**: Complete code documentation and data sources
**Quality Controls**: Outlier detection, missing data handling, validation

## 🔧 Technical Requirements

**Python**: 3.8+
**Key Dependencies**: pandas, numpy, scipy, yfinance, matplotlib

## 📚 Research Contribution

This study contributes to the event study literature by providing a well-powered analysis of technology product launch effects using modern data and methodology. The null findings support market efficiency theory while establishing a robust framework for future research.

**Limitations**: Focus on major product launches, daily frequency data, technology sector concentration.

---

**Research Ethics**: Uses publicly available historical market data for academic research. No proprietary information or investment advice provided.