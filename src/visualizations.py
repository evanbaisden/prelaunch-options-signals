"""
Pure visualization functions for prelaunch options signals analysis.
All functions take DataFrames/Series and return saved plot paths.
Uses matplotlib only (no seaborn).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union


def create_volume_summary_plot(results_df: pd.DataFrame, output_path: Union[str, Path]) -> str:
    """
    Create volume spike summary plot from results DataFrame.
    
    Args:
        results_df: DataFrame with analysis results containing volume data
        output_path: Path to save the plot
        
    Returns:
        str: Path to saved plot file
    """
    if len(results_df) == 0:
        return ""
        
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    products = results_df['product']
    volume_spikes = results_df.get('volume_spike_pct', [0] * len(products))
    
    bars = ax.bar(products, volume_spikes, alpha=0.7)
    ax.set_title('Volume Spike Analysis by Product Launch')
    ax.set_ylabel('Volume Spike (%)')
    ax.set_xlabel('Product')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def create_volume_analysis_plot(results_df: pd.DataFrame, output_path: Union[str, Path]) -> str:
    """
    Create detailed volume analysis plot comparing announcement vs release spikes.
    
    Args:
        results_df: DataFrame with analysis results containing volume data
        output_path: Path to save the plot
        
    Returns:
        str: Path to saved plot file
    """
    if len(results_df) == 0:
        return ""
        
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    products = results_df['product']
    
    # Announcement volume spikes
    ann_spikes = results_df.get('announcement_volume_spike', [0] * len(products))
    ax1.bar(products, ann_spikes, alpha=0.7, color='blue')
    ax1.set_title('Volume Spikes at Announcement')
    ax1.set_ylabel('Volume Spike (%)')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Release volume spikes  
    rel_spikes = results_df.get('release_volume_spike', [0] * len(products))
    ax2.bar(products, rel_spikes, alpha=0.7, color='green')
    ax2.set_title('Volume Spikes at Release')
    ax2.set_ylabel('Volume Spike (%)')
    ax2.set_xlabel('Product')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def create_returns_summary_plot(results_df: pd.DataFrame, output_path: Union[str, Path]) -> str:
    """
    Create returns summary plot comparing announcement vs release returns.
    
    Args:
        results_df: DataFrame with analysis results containing returns data
        output_path: Path to save the plot
        
    Returns:
        str: Path to saved plot file
    """
    if len(results_df) == 0:
        return ""
        
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    products = results_df['product']
    
    # Plot announcement and release returns
    ann_returns = results_df.get('announcement_5day_return', [0] * len(products))
    rel_returns = results_df.get('release_5day_return', [0] * len(products))
    
    x = np.arange(len(products))
    width = 0.35
    
    ax.bar(x - width/2, ann_returns, width, label='Announcement Return', alpha=0.7)
    ax.bar(x + width/2, rel_returns, width, label='Release Return', alpha=0.7)
    
    ax.set_title('5-Day Returns Around Launch Events')
    ax.set_ylabel('Return (%)')
    ax.set_xlabel('Product')
    ax.set_xticks(x)
    ax.set_xticklabels(products, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def create_price_movement_plot(price_data: pd.DataFrame, event_date: pd.Timestamp, 
                              output_path: Union[str, Path], title: str = "Price Movement") -> str:
    """
    Create price movement plot around an event date.
    
    Args:
        price_data: DataFrame with Date and Adj Close columns
        event_date: Event date to center the plot around
        output_path: Path to save the plot
        title: Plot title
        
    Returns:
        str: Path to saved plot file
    """
    if len(price_data) == 0:
        return ""
        
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize prices to event date = 100
    event_price = price_data.loc[price_data['Date'] == event_date, 'Adj Close']
    if len(event_price) == 0:
        # Find nearest date
        event_price = price_data.iloc[(price_data['Date'] - event_date).abs().argsort()[:1]]['Adj Close']
    
    normalized_prices = (price_data['Adj Close'] / event_price.iloc[0]) * 100
    
    ax.plot(price_data['Date'], normalized_prices, linewidth=2, alpha=0.8)
    ax.axvline(x=event_date, color='red', linestyle='--', alpha=0.7, label='Event Date')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price (Event Date = 100)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def create_volume_timeseries_plot(volume_data: pd.DataFrame, event_date: pd.Timestamp,
                                 output_path: Union[str, Path], title: str = "Volume Analysis") -> str:
    """
    Create volume timeseries plot with baseline and event highlighting.
    
    Args:
        volume_data: DataFrame with Date, Volume, and optional Volume_MA20 columns
        event_date: Event date to highlight
        output_path: Path to save the plot
        title: Plot title
        
    Returns:
        str: Path to saved plot file
    """
    if len(volume_data) == 0:
        return ""
        
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate baseline if not provided
    if 'Volume_MA20' not in volume_data.columns:
        volume_data = volume_data.copy()
        volume_data['Volume_MA20'] = volume_data['Volume'].rolling(window=20).mean()
    
    ax.bar(volume_data['Date'], volume_data['Volume'], alpha=0.6, width=0.8)
    ax.plot(volume_data['Date'], volume_data['Volume_MA20'], 
           color='red', linewidth=2, label='20-day Moving Average')
    ax.axvline(x=event_date, color='green', linestyle='--', alpha=0.7, label='Event Date')
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Trading Volume')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)