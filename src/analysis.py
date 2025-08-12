import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Optional
import os
from dotenv import load_dotenv

# Import new modules
from .config import get_config
from .logging_setup import setup_logging, set_seeds
from .schemas import validate_events_df

# Load environment variables
load_dotenv()

class PrelaunchAnalyzer:
    def __init__(self, data_dir=None, results_dir=None, config=None):
        # Load config
        self.config = config or get_config()
        
        # Set up directories
        self.data_dir = Path(data_dir or self.config.raw_data_dir)
        
        # Create timestamped results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(results_dir or self.config.results_dir) / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging to file in results directory
        self.logger = setup_logging(self.config.log_level)
        log_handler = logging.FileHandler(self.run_dir / 'run.log')
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(log_handler)
        
        # Set seeds for reproducibility
        set_seeds(self.config.seed)
        
        # Load and validate events data
        self.events_df = self._load_and_validate_events()
        
        # Initialize results storage
        self.results = {}
        
        # Analysis parameters from config
        self.params = {
            'baseline_days': self.config.baseline_days,
            'signal_window_announce_before': 5,  # TODO: make configurable
            'signal_window_announce_after': 20,   # TODO: make configurable  
            'signal_window_release_before': 5,    # TODO: make configurable
            'signal_window_release_after': 10,    # TODO: make configurable
            'z_thresholds': {
                'low': self.config.z_thresholds[0],
                'med': self.config.z_thresholds[1], 
                'high': self.config.z_thresholds[2],
                'extreme': self.config.z_thresholds[3] if len(self.config.z_thresholds) > 3 else 5.0
            }
        }
        
        # Initialize metadata
        self.metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'parameters': self.params,
            'config': {
                'events_csv': self.config.events_csv,
                'baseline_days': self.config.baseline_days,
                'z_thresholds': self.config.z_thresholds,
                'event_windows': self.config.event_windows,
                'seed': self.config.seed
            },
            'data_sources': [],
            'calculation_definitions': {
                'announcement_5day_return': 'Cumulative return from close[t-5] to close[t+0] on announcement day',
                'release_5day_return': 'Cumulative return from close[t-5] to close[t+0] on release day',
                'volume_spike_pct': '(volume[period] / baseline_mean) - 1, where baseline = 20-day MA',
                'pre_announce_return': 'Average daily return 60 trading days before announcement',
                'announce_to_release_return': 'Average daily return from announcement to release day',
                'post_release_return': 'Average daily return 30 trading days after release',
            }
        }
        
    def _load_and_validate_events(self) -> pd.DataFrame:
        """Load and validate events_master.csv."""
        events_path = Path(self.config.events_csv)
        if not events_path.exists():
            self.logger.warning(f"Events file not found: {events_path}. Using empty DataFrame.")
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(events_path)
            self.logger.info(f"Loaded {len(df)} events from {events_path}")
            
            # Validate schema
            validated_df = validate_events_df(df)
            self.logger.info("Events data validation passed")
            return validated_df
            
        except Exception as e:
            self.logger.error(f"Events data validation failed: {e}")
            raise
    
    def load_stock_data(self, filename):
        """Load and clean stock data from CSV file"""
        filepath = self.data_dir / filename
        df = pd.read_csv(filepath, skiprows=2)  # Skip the first 2 rows (headers)
        
        # The first column contains dates, rename it
        df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        
        # Clean and format the data
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
        df = df.reset_index(drop=True)
        
        return df
    
    def analyze_xbox_data(self):
        """Analyze Microsoft Xbox Series X/S launch data"""
        print("Analyzing Microsoft Xbox Series X/S launch data...")
        
        # Load the data
        xbox_df = self.load_stock_data("microsoft_xbox_series_x-s_raw.csv")
        
        # Key dates for Xbox Series X/S
        announcement_date = datetime(2020, 9, 9)  # Xbox Series X/S announcement
        release_date = datetime(2020, 11, 10)     # Launch date
        
        # Calculate price movements
        xbox_df['Price_Change'] = xbox_df['Adj Close'].pct_change()
        xbox_df['Volume_MA'] = xbox_df['Volume'].rolling(window=20).mean()
        xbox_df['Volume_Ratio'] = xbox_df['Volume'] / xbox_df['Volume_MA']
        
        # Find key periods
        announce_idx = self.find_nearest_date_index(xbox_df, announcement_date)
        release_idx = self.find_nearest_date_index(xbox_df, release_date)
        
        print(f"Announcement date index: {announce_idx}")
        print(f"Release date index: {release_idx}")
        
        # Pre-announcement analysis (60 days before announcement)
        pre_announce_start = max(0, announce_idx - 60)
        pre_announce_data = xbox_df.iloc[pre_announce_start:announce_idx]
        
        # Post-announcement to release
        announce_to_release = xbox_df.iloc[announce_idx:release_idx]
        
        # Post-release (30 days after release)
        post_release_end = min(len(xbox_df), release_idx + 30)
        post_release_data = xbox_df.iloc[release_idx:post_release_end]
        
        # Calculate statistics
        stats = {
            'pre_announce_avg_return': pre_announce_data['Price_Change'].mean(),
            'pre_announce_avg_volume': pre_announce_data['Volume'].mean(),
            'announce_to_release_avg_return': announce_to_release['Price_Change'].mean(),
            'announce_to_release_avg_volume': announce_to_release['Volume'].mean(),
            'post_release_avg_return': post_release_data['Price_Change'].mean(),
            'post_release_avg_volume': post_release_data['Volume'].mean(),
        }
        
        # Price performance around key dates
        if announce_idx >= 5:
            pre_announce_price = xbox_df.iloc[announce_idx - 5]['Adj Close']
            announce_price = xbox_df.iloc[announce_idx]['Adj Close']
            announce_return = (announce_price - pre_announce_price) / pre_announce_price
            stats['announcement_5day_return'] = announce_return
        
        if release_idx >= 5:
            pre_release_price = xbox_df.iloc[release_idx - 5]['Adj Close']
            release_price = xbox_df.iloc[release_idx]['Adj Close']
            release_return = (release_price - pre_release_price) / pre_release_price
            stats['release_5day_return'] = release_return
        
        self.results['xbox'] = stats
        return stats
    
    def analyze_nvidia_rtx30_data(self):
        """Analyze NVIDIA RTX 30 series launch data"""
        print("Analyzing NVIDIA RTX 30 series launch data...")
        
        rtx30_df = self.load_stock_data("nvidia_rtx_30_series_raw.csv")
        
        # Key dates for RTX 30 series
        announcement_date = datetime(2020, 9, 1)   # RTX 30 series announcement
        release_date = datetime(2020, 9, 17)       # RTX 3080 launch
        
        return self._analyze_nvidia_data(rtx30_df, announcement_date, release_date, 'rtx30')
    
    def analyze_nvidia_rtx40_data(self):
        """Analyze NVIDIA RTX 40 series launch data"""
        print("Analyzing NVIDIA RTX 40 series launch data...")
        
        rtx40_df = self.load_stock_data("nvidia_rtx_40_series_raw.csv")
        
        # Key dates for RTX 40 series  
        announcement_date = datetime(2022, 9, 20)  # RTX 40 series announcement
        release_date = datetime(2022, 10, 12)      # RTX 4090/4080 launch
        
        return self._analyze_nvidia_data(rtx40_df, announcement_date, release_date, 'rtx40')
    
    def analyze_nvidia_rtx40_super_data(self):
        """Analyze NVIDIA RTX 40 SUPER series launch data"""
        print("Analyzing NVIDIA RTX 40 SUPER series launch data...")
        
        rtx40_super_df = self.load_stock_data("nvidia_rtx_40_super_raw.csv")
        
        # Key dates for RTX 40 SUPER series
        announcement_date = datetime(2024, 1, 8)   # RTX 40 SUPER announcement
        release_date = datetime(2024, 1, 17)       # RTX 40 SUPER launch
        
        return self._analyze_nvidia_data(rtx40_super_df, announcement_date, release_date, 'rtx40_super')
    
    def analyze_iphone_12_data(self):
        """Analyze Apple iPhone 12 launch data"""
        print("Analyzing Apple iPhone 12 launch data...")
        
        iphone12_df = self.load_stock_data("apple_iphone_12_raw.csv")
        
        # Key dates for iPhone 12
        announcement_date = datetime(2020, 10, 13)  # iPhone 12 announcement
        release_date = datetime(2020, 10, 23)       # iPhone 12 launch
        
        return self._analyze_apple_data(iphone12_df, announcement_date, release_date, 'iphone12')
    
    def analyze_iphone_13_data(self):
        """Analyze Apple iPhone 13 launch data"""
        print("Analyzing Apple iPhone 13 launch data...")
        
        iphone13_df = self.load_stock_data("apple_iphone_13_raw.csv")
        
        # Key dates for iPhone 13
        announcement_date = datetime(2021, 9, 14)   # iPhone 13 announcement
        release_date = datetime(2021, 9, 24)        # iPhone 13 launch
        
        return self._analyze_apple_data(iphone13_df, announcement_date, release_date, 'iphone13')
    
    def analyze_iphone_14_data(self):
        """Analyze Apple iPhone 14 launch data"""
        print("Analyzing Apple iPhone 14 launch data...")
        
        iphone14_df = self.load_stock_data("apple_iphone_14_raw.csv")
        
        # Key dates for iPhone 14
        announcement_date = datetime(2022, 9, 7)    # iPhone 14 announcement
        release_date = datetime(2022, 9, 16)        # iPhone 14 launch
        
        return self._analyze_apple_data(iphone14_df, announcement_date, release_date, 'iphone14')
    
    def analyze_iphone_15_data(self):
        """Analyze Apple iPhone 15 launch data"""
        print("Analyzing Apple iPhone 15 launch data...")
        
        iphone15_df = self.load_stock_data("apple_iphone_15_raw.csv")
        
        # Key dates for iPhone 15
        announcement_date = datetime(2023, 9, 12)   # iPhone 15 announcement
        release_date = datetime(2023, 9, 22)        # iPhone 15 launch
        
        return self._analyze_apple_data(iphone15_df, announcement_date, release_date, 'iphone15')
    
    def _analyze_nvidia_data(self, df, announcement_date, release_date, product_key):
        """Common analysis function for NVIDIA data"""
        # Calculate price movements
        df['Price_Change'] = df['Adj Close'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Find key periods
        announce_idx = self.find_nearest_date_index(df, announcement_date)
        release_idx = self.find_nearest_date_index(df, release_date)
        
        print(f"{product_key} - Announcement date index: {announce_idx}")
        print(f"{product_key} - Release date index: {release_idx}")
        
        # Pre-announcement analysis (60 days before)
        pre_announce_start = max(0, announce_idx - 60)
        pre_announce_data = df.iloc[pre_announce_start:announce_idx]
        
        # Post-announcement to release
        announce_to_release = df.iloc[announce_idx:release_idx]
        
        # Post-release (30 days after)
        post_release_end = min(len(df), release_idx + 30)
        post_release_data = df.iloc[release_idx:post_release_end]
        
        # Calculate statistics
        stats = {
            'pre_announce_avg_return': pre_announce_data['Price_Change'].mean(),
            'pre_announce_avg_volume': pre_announce_data['Volume'].mean(),
            'announce_to_release_avg_return': announce_to_release['Price_Change'].mean(),
            'announce_to_release_avg_volume': announce_to_release['Volume'].mean(),
            'post_release_avg_return': post_release_data['Price_Change'].mean(),
            'post_release_avg_volume': post_release_data['Volume'].mean(),
        }
        
        # Price performance around key dates
        if announce_idx >= 5:
            pre_announce_price = df.iloc[announce_idx - 5]['Adj Close']
            announce_price = df.iloc[announce_idx]['Adj Close']
            announce_return = (announce_price - pre_announce_price) / pre_announce_price
            stats['announcement_5day_return'] = announce_return
        
        if release_idx >= 5:
            pre_release_price = df.iloc[release_idx - 5]['Adj Close']
            release_price = df.iloc[release_idx]['Adj Close']
            release_return = (release_price - pre_release_price) / pre_release_price
            stats['release_5day_return'] = release_return
        
        self.results[product_key] = stats
        return stats
    
    def _analyze_apple_data(self, df, announcement_date, release_date, product_key):
        """Common analysis function for Apple data"""
        # Calculate price movements
        df['Price_Change'] = df['Adj Close'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Find key periods
        announce_idx = self.find_nearest_date_index(df, announcement_date)
        release_idx = self.find_nearest_date_index(df, release_date)
        
        print(f"{product_key} - Announcement date index: {announce_idx}")
        print(f"{product_key} - Release date index: {release_idx}")
        
        # Pre-announcement analysis (60 days before)
        pre_announce_start = max(0, announce_idx - 60)
        pre_announce_data = df.iloc[pre_announce_start:announce_idx]
        
        # Post-announcement to release
        announce_to_release = df.iloc[announce_idx:release_idx]
        
        # Post-release (30 days after)
        post_release_end = min(len(df), release_idx + 30)
        post_release_data = df.iloc[release_idx:post_release_end]
        
        # Calculate statistics
        stats = {
            'pre_announce_avg_return': pre_announce_data['Price_Change'].mean(),
            'pre_announce_avg_volume': pre_announce_data['Volume'].mean(),
            'announce_to_release_avg_return': announce_to_release['Price_Change'].mean(),
            'announce_to_release_avg_volume': announce_to_release['Volume'].mean(),
            'post_release_avg_return': post_release_data['Price_Change'].mean(),
            'post_release_avg_volume': post_release_data['Volume'].mean(),
        }
        
        # Price performance around key dates
        if announce_idx >= 5:
            pre_announce_price = df.iloc[announce_idx - 5]['Adj Close']
            announce_price = df.iloc[announce_idx]['Adj Close']
            announce_return = (announce_price - pre_announce_price) / pre_announce_price
            stats['announcement_5day_return'] = announce_return
        
        if release_idx >= 5:
            pre_release_price = df.iloc[release_idx - 5]['Adj Close']
            release_price = df.iloc[release_idx]['Adj Close']
            release_return = (release_price - pre_release_price) / pre_release_price
            stats['release_5day_return'] = release_return
        
        self.results[product_key] = stats
        return stats
    
    def find_nearest_date_index(self, df, target_date):
        """Find the index of the row with the date closest to target_date"""
        df['date_diff'] = abs((df['Date'] - target_date).dt.days)
        nearest_idx = df['date_diff'].idxmin()
        df = df.drop('date_diff', axis=1)
        return nearest_idx
    
    def create_summary_report(self):
        """Create a summary report of all analyses"""
        if not self.results:
            print("No analysis results available. Please run analyses first.")
            return
        
        print("\n" + "="*50)
        print("PRELAUNCH OPTIONS SIGNALS - PHASE 1 ANALYSIS SUMMARY")
        print("="*50)
        
        for product, stats in self.results.items():
            print(f"\n{product.upper()} ANALYSIS:")
            print("-" * 30)
            print(f"Pre-announcement avg daily return: {stats['pre_announce_avg_return']:.4f} ({stats['pre_announce_avg_return']*100:.2f}%)")
            print(f"Announcement to release avg daily return: {stats['announce_to_release_avg_return']:.4f} ({stats['announce_to_release_avg_return']*100:.2f}%)")
            print(f"Post-release avg daily return: {stats['post_release_avg_return']:.4f} ({stats['post_release_avg_return']*100:.2f}%)")
            
            if 'announcement_5day_return' in stats:
                print(f"5-day return around announcement: {stats['announcement_5day_return']:.4f} ({stats['announcement_5day_return']*100:.2f}%)")
            if 'release_5day_return' in stats:
                print(f"5-day return around release: {stats['release_5day_return']:.4f} ({stats['release_5day_return']*100:.2f}%)")
            
            print(f"Pre-announcement avg volume: {stats['pre_announce_avg_volume']:,.0f}")
            print(f"Announcement to release avg volume: {stats['announce_to_release_avg_volume']:,.0f}")
            print(f"Post-release avg volume: {stats['post_release_avg_volume']:,.0f}")
    
    def save_results(self, filename_prefix: str = "phase1_summary") -> Tuple[str, str]:
        """
        Save analysis results to CSV and metadata to JSON for reproducibility.
        
        Returns:
            Tuple of (csv_filename, json_filename)
        """
        csv_filename = f"{filename_prefix}.csv"
        json_filename = f"{filename_prefix}_metadata.json"
        
        # Prepare results DataFrame
        results_data = []
        for product, stats in self.results.items():
            row = {
                'product': product,
                'timestamp': self.metadata['analysis_timestamp'],
                **stats
            }
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Add calculation clarity columns
        results_df['calculation_note'] = (
            f"announcement_return: {self.metadata['calculation_definitions']['announcement_5day_return']}; "
            f"volume_spike_pct: {self.metadata['calculation_definitions']['volume_spike_pct']}"
        )
        
        # Save CSV to timestamped results directory
        csv_path = self.run_dir / csv_filename
        results_df.to_csv(csv_path, index=False)
        
        # Save metadata JSON
        json_path = self.run_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Generate visualization artifacts
        self._create_visualizations(results_df)
        
        self.logger.info(f"Results saved to {csv_path}")
        self.logger.info(f"Metadata saved to {json_path}")
        self.logger.info(f"All artifacts saved to {self.run_dir}")
        
        return str(csv_filename), str(json_filename)
        
    def _create_visualizations(self, results_df: pd.DataFrame):
        """Create required visualization artifacts using pure functions."""
        if len(results_df) == 0:
            self.logger.warning("No results data to visualize")
            return
            
        from .visualizations import (
            create_volume_summary_plot,
            create_volume_analysis_plot, 
            create_returns_summary_plot
        )
        
        try:
            # Generate all required plots
            volume_summary_path = create_volume_summary_plot(
                results_df, self.run_dir / 'volume_summary.png'
            )
            volume_analysis_path = create_volume_analysis_plot(
                results_df, self.run_dir / 'volume_analysis.png'
            )
            returns_summary_path = create_returns_summary_plot(
                results_df, self.run_dir / 'returns_summary.png'
            )
            
            self.logger.info(f"Visualization artifacts created: {volume_summary_path}, {volume_analysis_path}, {returns_summary_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {e}")
            raise
    
    def calculate_baseline_volume(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate baseline volume using rolling mean for anomaly detection"""
        return df['Volume'].rolling(window=window, min_periods=window//2).mean()
    
    def detect_volume_anomalies(self, df: pd.DataFrame, z_threshold: float = 2.0) -> pd.Series:
        """Detect volume anomalies using z-score method"""
        baseline = self.calculate_baseline_volume(df)
        volume_ratio = df['Volume'] / baseline
        z_scores = (volume_ratio - volume_ratio.mean()) / volume_ratio.std()
        return z_scores.abs() > z_threshold
    
    def generate_volume_comparison_table(self) -> str:
        """Generate volume comparison table with proper baseline calculations"""
        if not self.results:
            logging.warning("No results available. Run analysis first.")
            return ""
        
        table_data = []
        
        product_names = {
            'xbox': 'Xbox Series X/S',
            'rtx30': 'RTX 30 Series', 
            'rtx40': 'RTX 40 Series',
            'rtx40_super': 'RTX 40 SUPER',
            'iphone12': 'iPhone 12',
            'iphone13': 'iPhone 13',
            'iphone14': 'iPhone 14',
            'iphone15': 'iPhone 15'
        }
        
        for key, name in product_names.items():
            if key in self.results:
                stats = self.results[key]
                pre_vol = stats['pre_announce_avg_volume'] / 1e6
                announce_vol = stats['announce_to_release_avg_volume'] / 1e6
                post_vol = stats['post_release_avg_volume'] / 1e6
                
                announce_change = ((announce_vol - pre_vol) / pre_vol) * 100
                post_change = ((post_vol - pre_vol) / pre_vol) * 100
                
                table_data.append({
                    'Product': name,
                    'Pre-Announcement Volume (M)': f"{pre_vol:.1f}",
                    'Announcement to Release (M)': f"{announce_vol:.1f} ({announce_change:+.0f}%)",
                    'Post-Release Volume (M)': f"{post_vol:.1f} ({post_change:+.0f}%)"
                })
        
        # Convert to markdown table
        df = pd.DataFrame(table_data)
        return df.to_markdown(index=False)

def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Prelaunch Options Signals Analysis")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single event command
    run_parser = subparsers.add_parser('run', help='Run analysis for a single event')
    run_parser.add_argument('--event-id', required=True, 
                           choices=['msft_xbox_series', 'nvda_rtx30', 'nvda_rtx40', 'nvda_rtx40_super',
                                   'aapl_iphone_12', 'aapl_iphone_13', 'aapl_iphone_14', 'aapl_iphone_15'],
                           help='Event ID to analyze')
    
    # Run all events command  
    run_all_parser = subparsers.add_parser('run-all', help='Run analysis for all events')
    
    # Common options for both commands
    for p in [run_parser, run_all_parser]:
        p.add_argument('--baseline-days', type=int, default=60,
                      help='Number of baseline days (default: 60)')
        p.add_argument('--z-thresholds', nargs='+', type=float, 
                      default=[1.645, 2.326, 2.576],
                      help='Z-score thresholds (default: 1.645 2.326 2.576)')
        p.add_argument('--windows', nargs='+', 
                      default=['-1:+1', '-2:+2', '-5:+5'],
                      help='Event windows (default: -1:+1 -2:+2 -5:+5)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create analyzer with custom parameters
    analyzer = PrelaunchAnalyzer()
    analyzer.params['baseline_days'] = args.baseline_days
    analyzer.params['z_thresholds'] = {
        'low': args.z_thresholds[0] if len(args.z_thresholds) > 0 else 1.645,
        'med': args.z_thresholds[1] if len(args.z_thresholds) > 1 else 2.326,  
        'high': args.z_thresholds[2] if len(args.z_thresholds) > 2 else 2.576,
        'extreme': args.z_thresholds[3] if len(args.z_thresholds) > 3 else 5.0
    }
    
    # Event ID to method mapping
    event_methods = {
        'msft_xbox_series': analyzer.analyze_xbox_data,
        'nvda_rtx30': analyzer.analyze_nvidia_rtx30_data,
        'nvda_rtx40': analyzer.analyze_nvidia_rtx40_data,
        'nvda_rtx40_super': analyzer.analyze_nvidia_rtx40_super_data,
        'aapl_iphone_12': analyzer.analyze_iphone_12_data,
        'aapl_iphone_13': analyzer.analyze_iphone_13_data,
        'aapl_iphone_14': analyzer.analyze_iphone_14_data,
        'aapl_iphone_15': analyzer.analyze_iphone_15_data,
    }
    
    if args.command == 'run':
        logging.info(f"Starting analysis for {args.event_id}...")
        event_methods[args.event_id]()
        analyzer.create_summary_report()
        csv_file, json_file = analyzer.save_results()
        logging.info(f"Analysis completed. Results: {csv_file}, {json_file}")
        
    elif args.command == 'run-all':
        logging.info("Starting Phase 1 analysis for all events...")
        for event_id, method in event_methods.items():
            logging.info(f"Analyzing {event_id}...")
            method()
        
        analyzer.create_summary_report()
        csv_file, json_file = analyzer.save_results()
        logging.info(f"Phase 1 analysis completed. Results: {csv_file}, {json_file}")


if __name__ == "__main__":
    main()