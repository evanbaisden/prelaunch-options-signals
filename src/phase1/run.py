"""
Phase 1 Analysis Orchestrator - Main execution engine for prelaunch options signals.
Coordinates baseline calculation, anomaly detection, signal extraction, and results compilation.
"""
import pandas as pd
import logging
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional

from ..common.types import LaunchEvent, StockData, AnalysisResults
from ..common.utils import validate_stock_data, ensure_directory, load_env_file, setup_logging
from ..config import Config

from .baseline import calculate_baseline_metrics, calculate_period_averages, validate_baseline_data
from .anomalies import identify_trading_anomalies, calculate_anomaly_scores
from .signals import extract_announcement_signal, extract_release_signal, compare_signals
from .outcomes import compile_analysis_results, export_results_to_csv, generate_summary_statistics, create_analysis_report


class Phase1Analyzer:
    """
    Main analyzer class for Phase 1 prelaunch options signals analysis.
    Coordinates all analysis modules and produces final results.
    """
    
    def __init__(self, data_dir: Optional[str] = None, results_dir: Optional[str] = None):
        """
        Initialize Phase 1 analyzer.
        
        Args:
            data_dir: Directory containing raw stock data CSV files
            results_dir: Directory for output results and reports
        """
        # Load environment configuration
        load_env_file()
        self.config = Config()
        
        # Set up logging
        setup_logging(self.config.log_level, self.config.log_format)
        
        # Configure directories
        self.data_dir = Path(data_dir or self.config.data_raw_dir)
        self.results_dir = Path(results_dir or self.config.results_dir)
        ensure_directory(self.results_dir)
        
        # Initialize results storage
        self.results: List[AnalysisResults] = []
        self.errors: List[Dict[str, str]] = []
        
        logging.info(f"Phase1Analyzer initialized - data: {self.data_dir}, results: {self.results_dir}")
    
    def load_stock_data(self, filename: str, ticker: str) -> StockData:
        """
        Load and validate stock data from CSV file.
        
        Args:
            filename: CSV filename in data directory
            ticker: Stock ticker symbol
            
        Returns:
            StockData object with validated data
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data validation fails
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Load CSV with proper format (skip header rows)
        df = pd.read_csv(filepath, skiprows=2)
        df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        
        # Clean and format data
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
        df = df.reset_index(drop=True)
        
        # Validate data quality
        quality_checks = validate_stock_data(df, ticker)
        
        # Get date range
        date_range = (df['Date'].iloc[0].date(), df['Date'].iloc[-1].date())
        
        return StockData(
            df=df,
            ticker=ticker,
            date_range=date_range,
            data_quality_checks=quality_checks
        )
    
    def analyze_product_launch(
        self, 
        event: LaunchEvent,
        data_filename: str,
        skip_on_error: bool = True
    ) -> Optional[AnalysisResults]:
        """
        Analyze a single product launch event.
        
        Args:
            event: LaunchEvent object with product details
            data_filename: CSV filename containing stock data
            skip_on_error: If True, log errors and continue; if False, raise exceptions
            
        Returns:
            AnalysisResults object or None if analysis failed
        """
        try:
            logging.info(f"Analyzing {event.name} ({event.company})")
            
            # Load stock data
            stock_data = self.load_stock_data(data_filename, event.ticker)
            
            # Validate baseline data availability
            baseline_start_idx = None
            for i, row in stock_data.df.iterrows():
                if row['Date'].date() >= event.announcement - pd.Timedelta(days=self.config.baseline_days):
                    baseline_start_idx = max(0, i - self.config.baseline_days)
                    break
            
            if baseline_start_idx is None:
                raise ValueError(f"Insufficient data for baseline period before {event.announcement}")
            
            # Calculate baseline metrics
            baseline = calculate_baseline_metrics(
                stock_data.df, 
                event.announcement, 
                self.config.baseline_days
            )
            
            # Validate baseline quality
            baseline_data = stock_data.df.iloc[baseline_start_idx:baseline_start_idx + self.config.baseline_days]
            if not validate_baseline_data(baseline_data):
                logging.warning(f"Baseline data quality issues for {event.name}")
            
            # Extract signals
            announcement_signal = None
            release_signal = None
            
            try:
                announcement_signal = extract_announcement_signal(
                    stock_data.df, 
                    event,
                    window_before=self.config.signal_window_announce[0],
                    window_after=self.config.signal_window_announce[1]
                )
            except Exception as e:
                logging.warning(f"Failed to extract announcement signal for {event.name}: {e}")
            
            try:
                release_signal = extract_release_signal(
                    stock_data.df,
                    event, 
                    window_before=self.config.signal_window_release[0],
                    window_after=self.config.signal_window_release[1]
                )
            except Exception as e:
                logging.warning(f"Failed to extract release signal for {event.name}: {e}")
            
            # Calculate period averages
            period_averages = calculate_period_averages(
                stock_data.df,
                event.announcement,
                event.release,
                self.config.baseline_days
            )
            
            # Compile results
            results = compile_analysis_results(
                event=event,
                stock_data=stock_data,
                baseline=baseline,
                announcement_signal=announcement_signal,
                release_signal=release_signal,
                period_averages=period_averages
            )
            
            logging.info(f"Successfully analyzed {event.name}")
            return results
            
        except Exception as e:
            error_msg = f"Analysis failed for {event.name}: {str(e)}"
            logging.error(error_msg)
            self.errors.append({
                'product': event.name,
                'company': event.company,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            if skip_on_error:
                return None
            else:
                raise
    
    def run_complete_analysis(self) -> Dict[str, any]:
        """
        Run complete Phase 1 analysis for all configured products.
        
        Returns:
            Dictionary with analysis summary and file paths
        """
        logging.info("Starting complete Phase 1 analysis")
        
        # Define product events (same as original analysis.py)
        events = [
            # Microsoft Xbox
            LaunchEvent(
                name="Xbox Series X/S",
                company="Microsoft", 
                ticker="MSFT",
                announcement=date(2020, 9, 9),
                release=date(2020, 11, 10),
                category="Gaming Hardware"
            ),
            
            # NVIDIA RTX Series
            LaunchEvent(
                name="RTX 30 Series",
                company="NVIDIA",
                ticker="NVDA", 
                announcement=date(2020, 9, 1),
                release=date(2020, 9, 17),
                category="Semiconductor Hardware"
            ),
            LaunchEvent(
                name="RTX 40 Series",
                company="NVIDIA",
                ticker="NVDA",
                announcement=date(2022, 9, 20),
                release=date(2022, 10, 12),
                category="Semiconductor Hardware"
            ),
            LaunchEvent(
                name="RTX 40 SUPER",
                company="NVIDIA", 
                ticker="NVDA",
                announcement=date(2024, 1, 8),
                release=date(2024, 1, 17),
                category="Semiconductor Hardware"
            ),
            
            # Apple iPhone
            LaunchEvent(
                name="iPhone 12",
                company="Apple",
                ticker="AAPL",
                announcement=date(2020, 10, 13),
                release=date(2020, 10, 23),
                category="Consumer Hardware"
            ),
            LaunchEvent(
                name="iPhone 13",
                company="Apple",
                ticker="AAPL", 
                announcement=date(2021, 9, 14),
                release=date(2021, 9, 24),
                category="Consumer Hardware"
            ),
            LaunchEvent(
                name="iPhone 14",
                company="Apple",
                ticker="AAPL",
                announcement=date(2022, 9, 7), 
                release=date(2022, 9, 16),
                category="Consumer Hardware"
            ),
            LaunchEvent(
                name="iPhone 15",
                company="Apple",
                ticker="AAPL",
                announcement=date(2023, 9, 12),
                release=date(2023, 9, 22),
                category="Consumer Hardware"
            )
        ]
        
        # Map events to data files  
        data_files = {
            "Xbox Series X/S": "microsoft_xbox_series_x-s_raw.csv",
            "RTX 30 Series": "nvidia_rtx_30_series_raw.csv",
            "RTX 40 Series": "nvidia_rtx_40_series_raw.csv", 
            "RTX 40 SUPER": "nvidia_rtx_40_super_raw.csv",
            "iPhone 12": "apple_iphone_12_raw.csv",
            "iPhone 13": "apple_iphone_13_raw.csv",
            "iPhone 14": "apple_iphone_14_raw.csv",
            "iPhone 15": "apple_iphone_15_raw.csv"
        }
        
        # Analyze each product
        self.results = []
        successful_analyses = 0
        
        for event in events:
            data_filename = data_files.get(event.name)
            if data_filename:
                result = self.analyze_product_launch(event, data_filename)
                if result:
                    self.results.append(result)
                    successful_analyses += 1
            else:
                logging.warning(f"No data file configured for {event.name}")
        
        # Generate summary statistics
        summary_stats = generate_summary_statistics(self.results)
        
        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = self.results_dir / f"phase1_summary_stats_{timestamp}"
        
        csv_path, metadata_path = export_results_to_csv(self.results, output_base, include_metadata=True)
        report_path = create_analysis_report(self.results, summary_stats, 
                                           self.results_dir / f"phase1_analysis_report_{timestamp}")
        
        # Create summary
        analysis_summary = {
            'total_products_configured': len(events),
            'successful_analyses': successful_analyses,
            'failed_analyses': len(self.errors),
            'output_files': {
                'summary_csv': str(csv_path),
                'metadata_json': str(metadata_path) if metadata_path else None,
                'analysis_report': str(report_path)
            },
            'summary_statistics': summary_stats,
            'errors': self.errors if self.errors else None
        }
        
        logging.info(f"Phase 1 analysis complete: {successful_analyses}/{len(events)} products analyzed successfully")
        
        return analysis_summary


def main():
    """Main entry point for Phase 1 analysis."""
    try:
        analyzer = Phase1Analyzer()
        results = analyzer.run_complete_analysis()
        
        print("\n" + "="*60)
        print("PHASE 1 ANALYSIS COMPLETE")
        print("="*60)
        print(f"Products analyzed: {results['successful_analyses']}/{results['total_products_configured']}")
        print(f"Results saved to: {results['output_files']['summary_csv']}")
        print(f"Report saved to: {results['output_files']['analysis_report']}")
        
        if results['errors']:
            print(f"\nErrors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error['product']}: {error['error']}")
        
        return results
        
    except Exception as e:
        logging.error(f"Phase 1 analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()