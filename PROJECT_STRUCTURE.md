# Project Structure

This document describes the research-ready structure of the prelaunch options signals analysis project.

## Directory Structure

```
prelaunch-options-signals/
├── .env                            # Local environment variables (not in git)
├── .env.example                    # Environment configuration template
├── Makefile                        # Development automation commands
├── pyproject.toml                  # Python project configuration (Ruff, Black)
├── pytest.ini                     # Test configuration
├── README.md                       # Project overview and usage instructions
├── requirements.txt                # Python dependencies with pinned versions
├── PROJECT_STRUCTURE.md           # This file
├── data/
│   ├── processed/
│   │   └── events_master.csv       # Master events database (validated schema)
│   └── raw/                        # Stock price data (8 CSV files)
│       ├── apple_iphone_12_raw.csv
│       ├── apple_iphone_13_raw.csv
│       ├── apple_iphone_14_raw.csv
│       ├── apple_iphone_15_raw.csv
│       ├── microsoft_xbox_series_x-s_raw.csv
│       ├── nvidia_rtx_30_series_raw.csv
│       ├── nvidia_rtx_40_series_raw.csv
│       └── nvidia_rtx_40_super_raw.csv
├── docs/
│   ├── BestPractices.md            # Development best practices
│   ├── DATA_DICTIONARY.md          # Data schema and field definitions
│   └── METHODOLOGY.md              # Statistical methodology documentation
├── notebooks/
│   └── Phase_1.ipynb              # Interactive analysis framework
├── results/
│   ├── run_YYYYMMDD_HHMMSS/       # Timestamped analysis runs
│   │   ├── run.log                 # Execution log
│   │   ├── phase1_summary.csv      # Quantitative results
│   │   ├── phase1_summary_metadata.json  # Configuration snapshot
│   │   ├── volume_summary.png      # Volume spike overview
│   │   ├── volume_analysis.png     # Detailed volume analysis
│   │   └── returns_summary.png     # Returns comparison charts
│   └── [legacy files...]          # Historical outputs
├── scripts/
│   └── run_phase1.py              # Batch execution script
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── analysis.py                 # Core analysis engine with CLI
│   ├── config.py                   # Centralized configuration management
│   ├── logging_setup.py            # Logging and reproducibility setup
│   ├── schemas.py                  # Data validation schemas (Pandera/Pydantic)
│   ├── visualizations.py          # Pure visualization functions (matplotlib)
│   ├── common/
│   │   ├── __init__.py
│   │   ├── types.py                # Data type definitions
│   │   └── utils.py                # Utility functions
│   ├── phase1/
│   │   ├── __init__.py
│   │   ├── anomalies.py            # Anomaly detection algorithms
│   │   ├── baseline.py             # Baseline calculation methods
│   │   ├── outcomes.py             # Outcome measurement functions
│   │   ├── run.py                  # Phase 1 execution pipeline
│   │   └── signals.py              # Signal extraction logic
│   └── phase2/
│       ├── __init__.py
│       └── README.md               # Phase 2 preparation notes
└── tests/
    ├── __init__.py
    ├── conftest.py                 # Test configuration and fixtures
    ├── test_analysis.py            # Core analysis tests
    ├── test_anomalies.py           # Anomaly detection tests
    ├── test_baseline.py            # Baseline calculation tests
    └── test_integration.py         # Integration tests
```

## Key Files Description

### Core Infrastructure
- **src/config.py**: Centralized configuration with environment variable loading
- **src/logging_setup.py**: Logging setup and reproducibility (seed management)
- **src/schemas.py**: Data validation using Pandera/Pydantic schemas
- **Makefile**: Development workflow automation (setup, lint, format, test, run)

### Analysis Engine
- **src/analysis.py**: Main analysis engine with CLI interface (`python -m src.analysis`)
- **src/visualizations.py**: Pure visualization functions (matplotlib only)
- **src/phase1/**: Phase 1 analysis modules (baseline, signals, anomalies, outcomes)
- **src/common/**: Shared utilities and data types

### Data Management
- **data/raw/**: Historical stock price data (OHLCV format)
- **data/processed/events_master.csv**: Validated event metadata with schema enforcement
- **schemas.py**: Data contracts for CSV validation

### Results & Outputs
- **results/run_YYYYMMDD_HHMMSS/**: Timestamped analysis runs with complete reproducibility
- **run.log**: Execution logging for debugging and audit trails
- **phase1_summary.csv**: Quantitative results with metadata
- **Visualization artifacts**: volume_summary.png, volume_analysis.png, returns_summary.png

### Documentation
- **docs/METHODOLOGY.md**: Statistical methodology and rationale
- **docs/DATA_DICTIONARY.md**: Data schema and field definitions
- **docs/BestPractices.md**: Development guidelines

### Testing & Quality
- **tests/**: Comprehensive test suite with multiple test types
- **pyproject.toml**: Code quality configuration (Ruff + Black, line length 100)
- **pytest.ini**: Test configuration with quiet mode and warning filters

## Usage

### Development Workflow
```bash
# Setup
make setup                          # Install dependencies and dev tools
cp .env.example .env               # Configure environment variables

# Quality Assurance  
make lint                          # Run Ruff + Black checks
make format                        # Auto-format code
make test                          # Run test suite

# Analysis Execution
make run                           # Full analysis (all events)
python -m src.analysis run --event-id aapl_iphone_12  # Single event
python -m src.analysis run-all --baseline-days 90     # Custom parameters
```

### CLI Interface
```bash
# Available commands
python -m src.analysis run --event-id EVENT_ID       # Single event analysis
python -m src.analysis run-all                       # All events analysis

# Options
--baseline-days N                  # Baseline period (default: 60)
--z-thresholds 1.96 2.58          # Statistical thresholds
--windows "-1:+1" "-5:+5"         # Event windows
```

## Phase Coverage Status

### ✅ Phase 1 (Complete)
- **✅ Product launch calendar**: 8 events across AAPL, NVDA, MSFT (2020-2024)
- **✅ Data collection pipeline**: Yahoo Finance integration, CSV validation
- **✅ Baseline metrics**: 30/60/90-day baseline calculations
- **✅ Anomaly detection**: Z-score based volume/price spike identification  
- **✅ Statistical framework**: Significance testing, event windows (±1/±2/±5)
- **✅ Analytical infrastructure**: CLI, configuration, reproducible outputs

### ⚠️ Phase 1 (Gaps)
- **❌ Options data integration**: Missing core "options signals" requirement
- **❌ Extended product coverage**: Need Tesla models, gaming consoles
- **❌ Options flow analysis**: Unusual options activity detection

### 🔄 Phase 2 (Setup Ready)
- **✅ Statistical testing framework**: Z-scores, significance levels, event studies
- **✅ Data pipeline extensibility**: Modular design for additional data sources
- **✅ Reproducible methodology**: Documented, version-controlled, configurable
- **✅ Results infrastructure**: Timestamped runs, metadata tracking

### 📋 Phase 2 (Needs Development)
- **❌ Options market data sources**: Options chains, volume, open interest, IV
- **❌ Earnings integration**: Quarterly earnings vs. estimates correlation
- **❌ Regression framework**: Statistical correlation testing infrastructure
- **❌ Trading strategy backtesting**: Risk-adjusted performance evaluation
- **❌ Academic paper framework**: Literature synthesis and citation management

## Data Coverage

### Current Scope
- **8 product launches** across 3 companies (AAPL, NVDA, MSFT)
- **4 years** of data coverage (2020-2024)
- **Stock market data**: Daily OHLCV from Yahoo Finance
- **Event types**: Consumer electronics, gaming hardware, semiconductors

### Phase 2 Expansion Needed
- **Options market data**: Chains, volume, open interest, implied volatility
- **Additional companies**: Tesla (TSLA), gaming console manufacturers
- **Earnings data**: Quarterly results vs. analyst estimates
- **Market microstructure**: Bid-ask spreads, order flow, institutional activity

## Research Standards

- **Reproducibility**: Timestamped outputs, configuration snapshots, seed management
- **Code quality**: Linting (Ruff), formatting (Black), comprehensive testing
- **Documentation**: Methodology, data dictionaries, API documentation
- **Version control**: Git-based, dependency pinning, environment management
- **Academic rigor**: Statistical methodology documentation, assumption validation

## Next Steps for Complete Implementation

### Immediate (Phase 1 Completion)
1. **Options data integration**: Add options chain data sources
2. **Product calendar expansion**: Tesla models, gaming consoles  
3. **Options anomaly detection**: Unusual options activity screening

### Medium-term (Phase 2 Foundation)
1. **Earnings data pipeline**: Quarterly earnings integration
2. **Regression framework**: Correlation testing infrastructure
3. **Strategy backtesting**: Performance evaluation engine

### Long-term (Phase 2 Completion)
1. **Academic paper framework**: Literature synthesis tools
2. **Advanced analytics**: Machine learning, market microstructure analysis
3. **Production deployment**: Real-time signal detection system