# Project Structure

This document describes the clean, research-ready structure of the prelaunch options signals analysis project.

## Directory Structure

```
prelaunch-options-signals/
├── .env.example                    # Environment configuration template
├── README.md                       # Project overview and usage instructions
├── requirements.txt                # Python dependencies with pinned versions
├── PROJECT_STRUCTURE.md           # This file
├── data/
│   ├── processed/
│   │   └── events_master.csv       # Master events database
│   └── raw/                        # Stock price data (8 CSV files)
│       ├── apple_iphone_12_raw.csv
│       ├── apple_iphone_13_raw.csv
│       ├── apple_iphone_14_raw.csv
│       ├── apple_iphone_15_raw.csv
│       ├── microsoft_xbox_series_x-s_raw.csv
│       ├── nvidia_rtx_30_series_raw.csv
│       ├── nvidia_rtx_40_series_raw.csv
│       └── nvidia_rtx_40_super_raw.csv
├── notebooks/
│   └── Phase_1.ipynb              # Interactive analysis framework
├── results/
│   ├── analysis_report.md          # Comprehensive research findings
│   ├── phase1_summary_[timestamp].csv    # Quantitative results
│   ├── phase1_summary_[timestamp]_metadata.json  # Analysis parameters
│   ├── price_movements_comparison.png     # Price movement visualizations
│   ├── returns_summary.png        # Returns comparison charts
│   ├── volume_analysis.png         # Volume pattern analysis
│   └── volume_summary.png          # Volume summary statistics
├── src/
│   ├── analysis.py                 # Core analysis engine
│   └── visualizations.py          # Chart and graph generation
└── tests/
    ├── __init__.py
    └── test_analysis.py            # Unit tests (6 tests)
```

## Key Files Description

### Core Analysis
- **src/analysis.py**: Main analysis engine with environment-driven configuration
- **src/visualizations.py**: Professional visualization generation
- **tests/test_analysis.py**: Comprehensive unit test suite

### Data
- **data/raw/**: 8 CSV files containing historical stock price data
- **data/processed/events_master.csv**: Event metadata and timing information

### Results
- **results/analysis_report.md**: Professional research report with findings
- **results/phase1_summary_[timestamp].csv**: Reproducible quantitative results
- **results/*.png**: High-resolution visualization outputs

### Configuration
- **.env.example**: Template for environment variables and analysis parameters
- **requirements.txt**: Pinned Python dependencies for reproducibility

## Usage

### Quick Start
```bash
pip install -r requirements.txt
python src/analysis.py
python src/visualizations.py
```

### Testing
```bash
python -m pytest tests/ -v
```

### Analysis Parameters
All analysis parameters are configurable via environment variables (see .env.example).

## Data Coverage

- **8 product launches** across 3 companies
- **4 years** of data (2020-2024)
- **3 market sectors**: Gaming hardware, semiconductors, consumer electronics
- **Comprehensive metrics**: Price movements, volume patterns, statistical significance

## Research Standards

- Reproducible results with timestamped outputs
- Professional academic tone throughout
- Comprehensive unit test coverage
- Clear methodology documentation
- Version-controlled dependencies