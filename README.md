# Prelaunch Options Signals Analysis

A comprehensive analysis of stock price and volume patterns around major technology product launches to identify potential trading signals and information leakage.

## Project Overview

This project examines the stock market behavior around significant tech product announcements and releases, analyzing eight major product launches across three companies:

**Microsoft**: Xbox Series X/S (2020)
**NVIDIA**: RTX 30 Series (2020), RTX 40 Series (2022), RTX 40 SUPER (2024)  
**Apple**: iPhone 12 (2020), iPhone 13 (2021), iPhone 14 (2022), iPhone 15 (2023)

## Key Findings

### Market Patterns Discovered
- NVIDIA products show 2-3x higher volatility than Microsoft around launch events
- Volume spikes averaging 32-96% during announcement-to-release periods for successful launches¹
- Pre-announcement momentum is the strongest predictor of post-announcement performance
- "Sell-the-news" effects occur in 75% of analyzed launches

### Trading Signal Performance
- RTX 30 Series: Strongest signal (+8.43% on announcement, +96% volume spike)
- RTX 40 SUPER: Most consistent performance (+5.51% announcement, +5.48% release)
- RTX 40 Series: Highest risk (-12.94% post-release volatility)
- Xbox Series X/S: Classic contrarian setup (-7.03% announcement, +2.22% recovery)
- iPhone 12: Best Apple signal (+7.02% announcement)
- iPhone 15: Weakest performance (-7.06% announcement) indicating "iPhone fatigue"

## Project Structure

```
prelaunch-options-signals/
├── data/
│   ├── raw/                    # Raw stock price data CSV files
│   └── processed/              # Cleaned and processed data
├── src/
│   ├── analysis.py            # Core analysis engine
│   └── visualizations.py      # Chart and graph generation
├── notebooks/
│   └── Phase_1.ipynb          # Interactive analysis notebook
├── results/
│   ├── analysis_report.md     # Comprehensive findings report
│   ├── *.png                  # Generated visualization charts
│   └── events_master.csv      # Master events database
└── requirements.txt           # Python dependencies
```

## Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd prelaunch-options-signals
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the analysis**
```bash
python src/analysis.py
```

4. **Generate visualizations**
```bash
python src/visualizations.py
```

## How to Reproduce Phase 1 Results

### Option 1: One Command Reproduction
```bash
# Complete Phase 1 analysis with all outputs
python src/analysis.py && python src/visualizations.py
```

### Option 2: Step-by-Step Reproduction
```bash
# 1. Set up environment (optional - will use defaults)
cp .env.example .env
# Edit .env if you want to change analysis parameters

# 2. Run core analysis
python src/analysis.py

# 3. Generate visualizations 
python src/visualizations.py

# 4. Run unit tests
python -m pytest tests/

# 5. Check results
ls results/
# Expected files:
# - phase1_summary_[timestamp].csv (reproducible results)
# - phase1_summary_[timestamp]_metadata.json (parameters used)
# - analysis_report.md (comprehensive findings)
# - *.png files (visualization charts)
```

### Option 3: Jupyter Notebook
```bash
# Launch notebook for interactive analysis
jupyter notebook notebooks/Phase_1.ipynb
```

### Expected Results Structure
After running the analysis, you should have:
```
results/
├── phase1_summary_20250811_HHMMSS.csv          # Quantitative results
├── phase1_summary_20250811_HHMMSS_metadata.json # Analysis parameters
├── analysis_report.md                           # Comprehensive report
├── price_movements_comparison.png               # Price charts
├── volume_analysis.png                         # Volume patterns
├── returns_summary.png                         # Returns comparison
└── volume_summary.png                          # Volume summary
```

### Result Verification
To verify results match the reference analysis:
```bash
# Check CSV structure and dimensions
python -c "import pandas as pd; df = pd.read_csv('results/phase1_summary_[timestamp].csv'); print('Columns:', len(df.columns)); print('Products:', len(df))"

# Expected output: Columns: 11, Products: 8
# CSV should contain analysis results for all 8 product launches
```

## Research Standards

This project follows academic research standards with:
- Reproducible results and timestamped outputs
- Comprehensive methodology documentation  
- Statistical significance testing and robustness checks
- Professional presentation suitable for academic review

---

**Footnotes:**
¹ Volume spikes calculated as `(period_volume - baseline_volume) / baseline_volume * 100` where baseline is 60-day pre-announcement average.

**Disclaimer**: This analysis is for market microstructure research purposes only. Past performance does not guarantee future results. Always consult with qualified professionals before making investment decisions.
