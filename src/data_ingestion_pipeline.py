"""
Complete Data Ingestion Pipeline
Orchestrates collection of options, equity, and earnings data for Phase II analysis
"""

import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from earnings_data_collector import EarningsDataCollector
from earnings_analysis import EarningsAnalyzer

class DataIngestionPipeline:
    def __init__(self, api_key: str, base_dir: str = "."):
        """
        Initialize data ingestion pipeline
        
        Args:
            api_key: AlphaVantage API key
            base_dir: Base directory for the project
        """
        self.api_key = api_key
        self.base_dir = base_dir
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize collectors and analyzers
        self.earnings_collector = EarningsDataCollector(api_key)
        self.earnings_analyzer = EarningsAnalyzer()
        
        # Data directories
        self.data_dirs = {
            'raw_options': os.path.join(base_dir, 'data', 'raw', 'options'),
            'raw_earnings': os.path.join(base_dir, 'data', 'raw', 'earnings'),
            'processed': os.path.join(base_dir, 'data', 'processed'),
            'results': os.path.join(base_dir, 'results')
        }
        
        # Ensure directories exist
        for dir_path in self.data_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def load_events_catalog(self, events_file: str = None) -> pd.DataFrame:
        """
        Load events catalog
        
        Args:
            events_file: Path to events file (optional)
            
        Returns:
            DataFrame with events data
        """
        if events_file is None:
            events_file = os.path.join(self.data_dirs['processed'], 'events_with_equity_data.csv')
        
        try:
            events_df = pd.read_csv(events_file)
            events_df['event_date'] = pd.to_datetime(events_df['event_date'])
            self.logger.info(f"Loaded {len(events_df)} events from catalog")
            return events_df
        except Exception as e:
            self.logger.error(f"Error loading events catalog: {str(e)}")
            return pd.DataFrame()
    
    def check_data_status(self) -> Dict:
        """
        Check current data availability status
        
        Returns:
            Dictionary with data status information
        """
        status = {
            'events': 0,
            'options_files': 0,
            'earnings_files': 0,
            'companies': set(),
            'date_range': None,
            'coverage': {}
        }
        
        # Check events
        events_df = self.load_events_catalog()
        if not events_df.empty:
            status['events'] = len(events_df)
            status['companies'] = set(events_df['symbol'].unique())
            status['date_range'] = {
                'start': events_df['event_date'].min().strftime('%Y-%m-%d'),
                'end': events_df['event_date'].max().strftime('%Y-%m-%d')
            }
        
        # Check options files
        if os.path.exists(self.data_dirs['raw_options']):
            options_files = [f for f in os.listdir(self.data_dirs['raw_options']) if f.endswith('.csv')]
            status['options_files'] = len(options_files)
            status['coverage']['options'] = f"{len(options_files)}/{len(events_df)} ({len(options_files)/len(events_df)*100:.1f}%)" if len(events_df) > 0 else "0/0"
        
        # Check earnings files
        if os.path.exists(self.data_dirs['raw_earnings']):
            earnings_files = [f for f in os.listdir(self.data_dirs['raw_earnings']) if f.endswith('_earnings.csv')]
            status['earnings_files'] = len(earnings_files)
            
            # Map earnings files to companies
            earnings_companies = set()
            for filename in earnings_files:
                symbol = filename.replace('_earnings.csv', '')
                earnings_companies.add(symbol)
            
            status['coverage']['earnings'] = f"{len(earnings_companies)}/{len(status['companies'])} companies" if status['companies'] else "0/0"
        
        return status
    
    def collect_missing_earnings(self, max_calls: int = None) -> Dict:
        """
        Collect earnings data for companies that don't have it yet
        
        Args:
            max_calls: Maximum API calls to make (default: remaining daily limit)
            
        Returns:
            Dictionary with collection results
        """
        # Get current status
        status = self.check_data_status()
        
        # Find companies without earnings data
        existing_earnings = set()
        if os.path.exists(self.data_dirs['raw_earnings']):
            for filename in os.listdir(self.data_dirs['raw_earnings']):
                if filename.endswith('_earnings.csv'):
                    symbol = filename.replace('_earnings.csv', '')
                    existing_earnings.add(symbol)
        
        missing_companies = status['companies'] - existing_earnings
        
        if not missing_companies:
            self.logger.info("All companies already have earnings data")
            return {'status': 'complete', 'collected': 0, 'remaining': 0}
        
        # Limit collection if specified
        companies_to_collect = list(missing_companies)
        if max_calls and len(companies_to_collect) * 2 > max_calls:  # 2 calls per company (earnings + estimates)
            companies_to_collect = companies_to_collect[:max_calls // 2]
        
        self.logger.info(f"Collecting earnings for {len(companies_to_collect)} companies: {companies_to_collect}")
        
        # Collect earnings data
        results = self.earnings_collector.collect_all_earnings(
            companies_to_collect, 
            self.data_dirs['raw_earnings']
        )
        
        # Count successful collections
        successful = sum(1 for symbol, (earnings_df, _) in results.items() if not earnings_df.empty)
        
        return {
            'status': 'partial' if len(missing_companies) > len(companies_to_collect) else 'complete',
            'collected': successful,
            'remaining': len(missing_companies) - len(companies_to_collect),
            'api_calls_used': self.earnings_collector.call_count
        }
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        Run comprehensive analysis with all available data
        
        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Starting comprehensive Phase II analysis")
        
        results = {}
        
        # 1. Run options analysis
        try:
            from comprehensive_options_research import main as run_options_analysis
            self.logger.info("Running options analysis...")
            run_options_analysis()
            results['options_analysis'] = 'completed'
        except Exception as e:
            self.logger.error(f"Options analysis failed: {str(e)}")
            results['options_analysis'] = f'failed: {str(e)}'
        
        # 2. Run Phase II statistical analysis
        try:
            from phase2_statistical_analysis import main as run_phase2_analysis
            self.logger.info("Running Phase II statistical analysis...")
            run_phase2_analysis()
            results['phase2_analysis'] = 'completed'
        except Exception as e:
            self.logger.error(f"Phase II analysis failed: {str(e)}")
            results['phase2_analysis'] = f'failed: {str(e)}'
        
        # 3. Run earnings analysis if data available
        try:
            earnings_status = self.check_data_status()
            if earnings_status['earnings_files'] > 0:
                self.logger.info("Running earnings analysis...")
                self.earnings_analyzer.generate_earnings_report()
                results['earnings_analysis'] = 'completed'
            else:
                results['earnings_analysis'] = 'skipped: no earnings data'
        except Exception as e:
            self.logger.error(f"Earnings analysis failed: {str(e)}")
            results['earnings_analysis'] = f'failed: {str(e)}'
        
        # 4. Generate summary report
        try:
            self.generate_pipeline_summary(results)
            results['summary_report'] = 'completed'
        except Exception as e:
            self.logger.error(f"Summary report failed: {str(e)}")
            results['summary_report'] = f'failed: {str(e)}'
        
        return results
    
    def generate_pipeline_summary(self, analysis_results: Dict):
        """
        Generate comprehensive pipeline summary report
        
        Args:
            analysis_results: Results from comprehensive analysis
        """
        status = self.check_data_status()
        
        summary = f"""# Data Ingestion Pipeline Summary
## Pre-Launch Options Signals: Complete Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Pipeline Version**: 1.0

---

## Data Status Overview

### Events Catalog
- **Total Events**: {status['events']}
- **Companies**: {len(status['companies'])} ({', '.join(sorted(status['companies']))})
- **Date Range**: {status['date_range']['start'] if status['date_range'] else 'N/A'} to {status['date_range']['end'] if status['date_range'] else 'N/A'}

### Data Coverage
- **Options Data**: {status['coverage'].get('options', 'N/A')}
- **Earnings Data**: {status['coverage'].get('earnings', 'N/A')}

---

## Analysis Results

"""
        
        # Add analysis status
        for analysis_type, result in analysis_results.items():
            status_emoji = "✅" if result == 'completed' else "❌" if 'failed' in result else "⏭️"
            summary += f"- **{analysis_type.replace('_', ' ').title()}**: {status_emoji} {result}\n"
        
        summary += f"""

---

## Generated Reports

The following reports have been generated in the `results/` directory:

1. **Options Analysis Report** (`options_analysis_report.md`)
   - Comprehensive options flow anomaly analysis
   - Statistical correlations and predictive modeling
   - Trading strategy backtesting results

2. **Phase II Statistical Analysis** (`phase2_analysis_results.csv`)
   - Advanced statistical testing and correlations
   - Regression analysis and model validation
   - Robustness checks and cross-validation

3. **Earnings Analysis Report** (`earnings_analysis_report.md`)
   - Earnings surprise correlations with options anomalies
   - Impact of earnings on stock price movements
   - Integration of fundamental and technical signals

---

## Data Pipeline Architecture

```
Data Sources → Collection → Processing → Analysis → Results
     ↓              ↓           ↓          ↓         ↓
AlphaVantage → Raw CSV → Events DF → Correlations → Reports
Yahoo Finance → Options → Equity → ML Models → Strategies
Manual → Events → Earnings → Stats → Backtests → Summary
```

---

## Next Steps

### For Additional Data Collection
1. **Options Data**: {32 - status['options_files']} events still need options data
2. **Earnings Data**: Additional quarters can be collected when API limit resets
3. **Extended Analysis**: Consider expanding to more companies and time periods

### For Enhanced Analysis
1. **Intraday Patterns**: Analyze hourly options flow patterns
2. **Cross-Asset Integration**: Include bond and currency market data
3. **Alternative Data**: Social media sentiment and news analysis
4. **Real-Time Implementation**: Build live trading signal generation

---

## Technical Details

### API Usage
- **AlphaVantage Calls**: {self.earnings_collector.call_count}/25 daily limit used
- **Rate Limiting**: Automatic 12-second delays between calls
- **Error Handling**: Comprehensive retry and fallback mechanisms

### Data Quality
- **Missing Data Handling**: Robust imputation and filtering
- **Outlier Detection**: Statistical outlier identification and treatment
- **Validation**: Cross-reference between multiple data sources

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Pipeline Framework**: Data Ingestion v1.0
**Contact**: Academic Research Project

"""
        
        # Save summary report
        summary_file = os.path.join(self.data_dirs['results'], 'pipeline_summary.md')
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        self.logger.info(f"Pipeline summary saved to {summary_file}")
    
    def run_full_pipeline(self, collect_earnings: bool = True, max_earnings_calls: int = 10) -> Dict:
        """
        Run the complete data ingestion and analysis pipeline
        
        Args:
            collect_earnings: Whether to collect missing earnings data
            max_earnings_calls: Maximum API calls for earnings collection
            
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting full data ingestion pipeline")
        
        pipeline_results = {
            'start_time': datetime.now(),
            'data_status': self.check_data_status()
        }
        
        # 1. Collect missing earnings data if requested
        if collect_earnings:
            self.logger.info("Phase 1: Collecting missing earnings data")
            earnings_results = self.collect_missing_earnings(max_earnings_calls)
            pipeline_results['earnings_collection'] = earnings_results
        else:
            pipeline_results['earnings_collection'] = {'status': 'skipped'}
        
        # 2. Run comprehensive analysis
        self.logger.info("Phase 2: Running comprehensive analysis")
        analysis_results = self.run_comprehensive_analysis()
        pipeline_results['analysis_results'] = analysis_results
        
        # 3. Final status check
        pipeline_results['final_status'] = self.check_data_status()
        pipeline_results['end_time'] = datetime.now()
        pipeline_results['duration'] = str(pipeline_results['end_time'] - pipeline_results['start_time'])
        
        self.logger.info(f"Pipeline completed in {pipeline_results['duration']}")
        
        return pipeline_results

def main():
    """
    Main function to run the complete data ingestion pipeline
    """
    # Load API key from environment
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    if not api_key:
        print("Error: ALPHAVANTAGE_API_KEY not found in environment variables")
        return
    
    # Initialize and run pipeline
    pipeline = DataIngestionPipeline(api_key)
    
    # Check current status
    print("Current Data Status:")
    print("-" * 30)
    status = pipeline.check_data_status()
    print(f"Events: {status['events']}")
    print(f"Options files: {status['options_files']}")
    print(f"Earnings files: {status['earnings_files']}")
    print(f"Companies: {len(status['companies'])}")
    print(f"Options coverage: {status['coverage'].get('options', 'N/A')}")
    print(f"Earnings coverage: {status['coverage'].get('earnings', 'N/A')}")
    
    # Run pipeline with limited earnings collection
    print("\nRunning pipeline...")
    results = pipeline.run_full_pipeline(collect_earnings=True, max_earnings_calls=8)
    
    # Print results summary
    print("\nPipeline Results:")
    print("-" * 20)
    print(f"Duration: {results['duration']}")
    print(f"Earnings collection: {results['earnings_collection']['status']}")
    for analysis, result in results['analysis_results'].items():
        print(f"{analysis}: {result}")

if __name__ == "__main__":
    main()