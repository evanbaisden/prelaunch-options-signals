# Data Sources and Collection

## Primary Data Sources

### Options Data
- **Source**: Alpha Vantage API
- **Coverage**: 60 trading days pre-launch per event
- **Variables**: Daily volume, open interest, implied volatility, Greeks
- **Total**: 92,076 option-chain observations

### Equity Data  
- **Source**: Yahoo Finance API
- **Purpose**: Daily stock returns for abnormal return calculation
- **Coverage**: Full sample period (June 2020 - March 2024)

### Earnings Data
- **Source**: Manual collection from company reports
- **Variables**: Quarterly earnings surprises, announcement dates
- **Total**: 610+ quarterly records

## Sample Construction

### Event Selection Criteria
- Major technology product launches
- High-profile announcements with clear launch dates
- Sufficient options trading volume
- Clean event windows (no confounding events)

### Companies Included
1. **Apple** (9 events): iPhone, Mac, Watch, Vision Pro launches
2. **NVIDIA** (7 events): RTX series, AI hardware launches  
3. **Microsoft** (5 events): Xbox, Surface, Windows, AI software
4. **Tesla** (5 events): Vehicle launches, Battery Day, AI Day
5. **AMD** (5 events): CPU and GPU launches, AI hardware
6. **Sony** (3 events): PlayStation, PSVR launches

### Product Categories
- Consumer Hardware
- Semiconductor Hardware  
- AI Hardware
- Gaming Hardware
- Software Platform
- Electric Vehicle
- Mixed Reality
- Energy Technology
- AI Software
- Computing Hardware
- Wearables
- AI Technology

## Data Quality Measures

### Completeness Checks
- Missing data imputation strategies
- Data availability validation
- Cross-source verification where possible

### Quality Scores
- Event-level data quality scores (0-1 scale)
- Based on completeness, consistency, timing accuracy
- Average quality score: ~0.70

## Data Processing Pipeline

1. **Raw Data Collection**: API calls, manual verification
2. **Cleaning**: Remove invalid observations, handle missing data  
3. **Transformation**: Calculate returns, z-scores, anomaly flags
4. **Integration**: Merge options, equity, and earnings datasets
5. **Validation**: Quality checks, outlier detection, consistency tests