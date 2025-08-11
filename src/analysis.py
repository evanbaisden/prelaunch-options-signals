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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format=os.getenv('LOG_FORMAT', '%(asctime)s %(levelname)s %(message)s')
)

class PrelaunchAnalyzer:
    def __init__(self, data_dir=None, results_dir=None):
        self.data_dir = Path(data_dir or os.getenv('DATA_RAW_DIR', 'data/raw'))
        self.results_dir = Path(results_dir or os.getenv('RESULTS_DIR', 'results'))
        self.results_dir.mkdir(exist_ok=True)
        
        # Analysis parameters from environment or defaults
        self.params = {
            'baseline_days': int(os.getenv('BASELINE_DAYS', 60)),
            'signal_window_announce_before': int(os.getenv('SIGNAL_WINDOW_ANNOUNCE_BEFORE', 5)),
            'signal_window_announce_after': int(os.getenv('SIGNAL_WINDOW_ANNOUNCE_AFTER', 20)),
            'signal_window_release_before': int(os.getenv('SIGNAL_WINDOW_RELEASE_BEFORE', 5)),
            'signal_window_release_after': int(os.getenv('SIGNAL_WINDOW_RELEASE_AFTER', 10)),
            'z_thresholds': {
                'low': float(os.getenv('Z_THRESHOLD_LOW', 1.645)),
                'med': float(os.getenv('Z_THRESHOLD_MED', 2.326)),
                'high': float(os.getenv('Z_THRESHOLD_HIGH', 2.576)),
                'extreme': float(os.getenv('Z_THRESHOLD_EXTREME', 5.0))
            }
        }
        
        self.results = {}
        self.metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'parameters': self.params,
            'data_sources': [],
            'calculation_definitions': {
                'announcement_5day_return': 'Cumulative return from close[t-5] to close[t+0] on announcement day',
                'release_5day_return': 'Cumulative return from close[t-5] to close[t+0] on release day',
                'volume_spike_pct': '(volume[period] / baseline_mean) - 1, where baseline = 20-day MA',
                'pre_announce_return': 'Average daily return 60 trading days before announcement',
                'announce_to_release_return': 'Average daily return from announcement to release day',
                'post_release_return': 'Average daily return 30 trading days after release',
                'data_source': 'yfinance adjusted close prices (split/dividend adjusted), raw volumes'
            }
        }
        
        logging.info(f"PrelaunchAnalyzer initialized with parameters: {self.params}")
    
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        json_filename = f"{filename_prefix}_{timestamp}_metadata.json"
        
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
            f"announcement_return: {self.metadata['calculation_definitions']['announcement_return']}; "
            f"volume_spike_pct: {self.metadata['calculation_definitions']['volume_spike_pct']}"
        )
        
        # Save CSV
        csv_path = self.results_dir / csv_filename
        results_df.to_csv(csv_path, index=False)
        
        # Save metadata JSON
        json_path = self.results_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logging.info(f"Results saved to {csv_path}")
        logging.info(f"Metadata saved to {json_path}")
        
        return str(csv_filename), str(json_filename)
    
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

if __name__ == "__main__":
    # Run the analysis
    analyzer = PrelaunchAnalyzer()
    
    # Analyze each product launch
    logging.info("Starting Phase 1 analysis...")
    analyzer.analyze_xbox_data()
    analyzer.analyze_nvidia_rtx30_data()
    analyzer.analyze_nvidia_rtx40_data()
    analyzer.analyze_nvidia_rtx40_super_data()
    analyzer.analyze_iphone_12_data()
    analyzer.analyze_iphone_13_data()
    analyzer.analyze_iphone_14_data()
    analyzer.analyze_iphone_15_data()
    
    # Create summary report
    analyzer.create_summary_report()
    
    # Save reproducible results
    csv_file, json_file = analyzer.save_results()
    logging.info(f"Phase 1 analysis completed. Results: {csv_file}, {json_file}")