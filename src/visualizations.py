import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from analysis import PrelaunchAnalyzer

class PrelaunchVisualizer(PrelaunchAnalyzer):
    def __init__(self, data_dir="data/raw", output_dir="results"):
        super().__init__(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style for better plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_price_movement_comparison(self):
        """Create a comparison chart of price movements around announcements"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Stock Price Movements Around Product Launch Announcements', fontsize=16, fontweight='bold')
        
        products = [
            ('microsoft_xbox_series_x-s_raw.csv', datetime(2020, 9, 9), 'Microsoft Xbox Series X/S'),
            ('nvidia_rtx_30_series_raw.csv', datetime(2020, 9, 1), 'NVIDIA RTX 30 Series'),
            ('nvidia_rtx_40_series_raw.csv', datetime(2022, 9, 20), 'NVIDIA RTX 40 Series'),
            ('nvidia_rtx_40_super_raw.csv', datetime(2024, 1, 8), 'NVIDIA RTX 40 SUPER'),
            ('apple_iphone_12_raw.csv', datetime(2020, 10, 13), 'Apple iPhone 12'),
            ('apple_iphone_13_raw.csv', datetime(2021, 9, 14), 'Apple iPhone 13'),
            ('apple_iphone_14_raw.csv', datetime(2022, 9, 7), 'Apple iPhone 14'),
            ('apple_iphone_15_raw.csv', datetime(2023, 9, 12), 'Apple iPhone 15')
        ]
        
        for i, (filename, announce_date, title) in enumerate(products):
            ax = axes[i // 4, i % 4]
            
            # Load data
            df = self.load_stock_data(filename)
            announce_idx = self.find_nearest_date_index(df, announce_date)
            
            # Get data window around announcement (-30 to +30 days)
            start_idx = max(0, announce_idx - 30)
            end_idx = min(len(df), announce_idx + 30)
            
            window_df = df.iloc[start_idx:end_idx].copy()
            window_df['Days_From_Announcement'] = range(-len(window_df[:announce_idx-start_idx]), 
                                                      len(window_df[announce_idx-start_idx:]))
            
            # Normalize price to announcement day = 100
            announce_price = df.iloc[announce_idx]['Adj Close']
            window_df['Normalized_Price'] = (window_df['Adj Close'] / announce_price) * 100
            
            # Plot
            ax.plot(window_df['Days_From_Announcement'], window_df['Normalized_Price'], 
                   linewidth=2, alpha=0.8)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Announcement')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Days from Announcement')
            ax.set_ylabel('Normalized Price (Announcement = 100)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'price_movements_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_volume_analysis_chart(self):
        """Create volume analysis charts"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Trading Volume Analysis Around Product Announcements', fontsize=16, fontweight='bold')
        
        products = [
            ('microsoft_xbox_series_x-s_raw.csv', datetime(2020, 9, 9), 'Microsoft Xbox Series X/S'),
            ('nvidia_rtx_30_series_raw.csv', datetime(2020, 9, 1), 'NVIDIA RTX 30 Series'),
            ('nvidia_rtx_40_series_raw.csv', datetime(2022, 9, 20), 'NVIDIA RTX 40 Series'),
            ('nvidia_rtx_40_super_raw.csv', datetime(2024, 1, 8), 'NVIDIA RTX 40 SUPER'),
            ('apple_iphone_12_raw.csv', datetime(2020, 10, 13), 'Apple iPhone 12'),
            ('apple_iphone_13_raw.csv', datetime(2021, 9, 14), 'Apple iPhone 13'),
            ('apple_iphone_14_raw.csv', datetime(2022, 9, 7), 'Apple iPhone 14'),
            ('apple_iphone_15_raw.csv', datetime(2023, 9, 12), 'Apple iPhone 15')
        ]
        
        for i, (filename, announce_date, title) in enumerate(products):
            ax = axes[i // 4, i % 4]
            
            # Load data
            df = self.load_stock_data(filename)
            announce_idx = self.find_nearest_date_index(df, announce_date)
            
            # Calculate rolling volume average
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
            
            # Get data window around announcement (-30 to +30 days)
            start_idx = max(0, announce_idx - 30)
            end_idx = min(len(df), announce_idx + 30)
            
            window_df = df.iloc[start_idx:end_idx].copy()
            window_df['Days_From_Announcement'] = range(-len(window_df[:announce_idx-start_idx]), 
                                                      len(window_df[announce_idx-start_idx:]))
            
            # Plot volume ratio
            ax.bar(window_df['Days_From_Announcement'], window_df['Volume_Ratio'], 
                  alpha=0.7, width=0.8)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Announcement')
            ax.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='20-day Average')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Days from Announcement')
            ax.set_ylabel('Volume Ratio (vs 20-day MA)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'volume_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_returns_summary_chart(self):
        """Create a summary chart of returns across different periods"""
        # Get all results
        if not self.results:
            self.analyze_xbox_data()
            self.analyze_nvidia_rtx30_data()
            self.analyze_nvidia_rtx40_data()
            self.analyze_nvidia_rtx40_super_data()
            self.analyze_iphone_12_data()
            self.analyze_iphone_13_data()
            self.analyze_iphone_14_data()
            self.analyze_iphone_15_data()
        
        # Prepare data for visualization
        products = ['Xbox Series X/S', 'RTX 30 Series', 'RTX 40 Series', 'RTX 40 SUPER', 
                   'iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 15']
        keys = ['xbox', 'rtx30', 'rtx40', 'rtx40_super', 'iphone12', 'iphone13', 'iphone14', 'iphone15']
        
        pre_announce_returns = []
        announce_to_release_returns = []
        post_release_returns = []
        announce_5day_returns = []
        release_5day_returns = []
        
        for key in keys:
            pre_announce_returns.append(self.results[key]['pre_announce_avg_return'] * 100)
            announce_to_release_returns.append(self.results[key]['announce_to_release_avg_return'] * 100)
            post_release_returns.append(self.results[key]['post_release_avg_return'] * 100)
            announce_5day_returns.append(self.results[key].get('announcement_5day_return', 0) * 100)
            release_5day_returns.append(self.results[key].get('release_5day_return', 0) * 100)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Average daily returns chart
        x = np.arange(len(products))
        width = 0.25
        
        ax1.bar(x - width, pre_announce_returns, width, label='Pre-Announcement', alpha=0.8)
        ax1.bar(x, announce_to_release_returns, width, label='Announcement to Release', alpha=0.8)
        ax1.bar(x + width, post_release_returns, width, label='Post-Release', alpha=0.8)
        
        ax1.set_xlabel('Product')
        ax1.set_ylabel('Average Daily Return (%)')
        ax1.set_title('Average Daily Returns by Period', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(products, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 5-day event returns
        x = np.arange(len(products))
        width = 0.35
        
        ax2.bar(x - width/2, announce_5day_returns, width, label='5-day Announcement Return', alpha=0.8)
        ax2.bar(x + width/2, release_5day_returns, width, label='5-day Release Return', alpha=0.8)
        
        ax2.set_xlabel('Product')
        ax2.set_ylabel('5-Day Return (%)')
        ax2.set_title('5-Day Returns Around Key Events', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(products, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'returns_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_volume_summary_chart(self):
        """Create a summary chart of volume patterns"""
        if not self.results:
            self.analyze_xbox_data()
            self.analyze_nvidia_rtx30_data()
            self.analyze_nvidia_rtx40_data()
            self.analyze_nvidia_rtx40_super_data()
            self.analyze_iphone_12_data()
            self.analyze_iphone_13_data()
            self.analyze_iphone_14_data()
            self.analyze_iphone_15_data()
        
        products = ['Xbox Series X/S', 'RTX 30 Series', 'RTX 40 Series', 'RTX 40 SUPER',
                   'iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 15']
        keys = ['xbox', 'rtx30', 'rtx40', 'rtx40_super', 'iphone12', 'iphone13', 'iphone14', 'iphone15']
        
        pre_announce_volume = []
        announce_to_release_volume = []
        post_release_volume = []
        
        for key in keys:
            pre_announce_volume.append(self.results[key]['pre_announce_avg_volume'] / 1e6)  # Convert to millions
            announce_to_release_volume.append(self.results[key]['announce_to_release_avg_volume'] / 1e6)
            post_release_volume.append(self.results[key]['post_release_avg_volume'] / 1e6)
        
        # Create chart
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        x = np.arange(len(products))
        width = 0.25
        
        ax.bar(x - width, pre_announce_volume, width, label='Pre-Announcement', alpha=0.8)
        ax.bar(x, announce_to_release_volume, width, label='Announcement to Release', alpha=0.8)
        ax.bar(x + width, post_release_volume, width, label='Post-Release', alpha=0.8)
        
        ax.set_xlabel('Product')
        ax.set_ylabel('Average Daily Volume (Millions)')
        ax.set_title('Average Trading Volume by Period', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(products, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'volume_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Create visualizer and generate all charts
    visualizer = PrelaunchVisualizer()
    
    print("Creating price movement comparison chart...")
    visualizer.create_price_movement_comparison()
    
    print("Creating volume analysis chart...")  
    visualizer.create_volume_analysis_chart()
    
    print("Creating returns summary chart...")
    visualizer.create_returns_summary_chart()
    
    print("Creating volume summary chart...")
    visualizer.create_volume_summary_chart()
    
    print("All visualizations completed and saved to results/")