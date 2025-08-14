"""
Earnings Data Collection Framework
Collects historical earnings data from AlphaVantage API for Phase II analysis
"""

import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple
import logging

class EarningsDataCollector:
    def __init__(self, api_key: str, rate_limit_delay: float = 12.0):
        """
        Initialize earnings data collector
        
        Args:
            api_key: AlphaVantage API key
            rate_limit_delay: Delay between API calls (12 seconds for 5 calls/minute)
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://www.alphavantage.co/query"
        self.call_count = 0
        self.max_calls_per_day = 25
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_earnings_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch earnings data for a symbol
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing earnings data or None if failed
        """
        if self.call_count >= self.max_calls_per_day:
            self.logger.warning(f"Daily API limit reached ({self.max_calls_per_day} calls)")
            return None
            
        params = {
            'function': 'EARNINGS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            self.logger.info(f"Fetching earnings data for {symbol}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                self.logger.error(f"API Error for {symbol}: {data['Error Message']}")
                return None
            elif 'Note' in data:
                self.logger.warning(f"API Note for {symbol}: {data['Note']}")
                return None
                
            self.call_count += 1
            self.logger.info(f"Successfully fetched earnings for {symbol} (Call {self.call_count}/{self.max_calls_per_day})")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching earnings for {symbol}: {str(e)}")
            return None
    
    def get_earnings_estimates(self, symbol: str) -> Optional[Dict]:
        """
        Fetch earnings estimates for a symbol
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing earnings estimates or None if failed
        """
        if self.call_count >= self.max_calls_per_day:
            self.logger.warning(f"Daily API limit reached ({self.max_calls_per_day} calls)")
            return None
            
        params = {
            'function': 'EARNINGS_ESTIMATES',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            self.logger.info(f"Fetching earnings estimates for {symbol}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                self.logger.error(f"API Error for {symbol}: {data['Error Message']}")
                return None
            elif 'Note' in data:
                self.logger.warning(f"API Note for {symbol}: {data['Note']}")
                return None
                
            self.call_count += 1
            self.logger.info(f"Successfully fetched estimates for {symbol} (Call {self.call_count}/{self.max_calls_per_day})")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching estimates for {symbol}: {str(e)}")
            return None
    
    def process_earnings_data(self, earnings_data: Dict) -> pd.DataFrame:
        """
        Process raw earnings data into DataFrame
        
        Args:
            earnings_data: Raw earnings data from API
            
        Returns:
            Processed DataFrame with earnings information
        """
        if not earnings_data or 'quarterlyEarnings' not in earnings_data:
            return pd.DataFrame()
            
        quarterly_earnings = earnings_data['quarterlyEarnings']
        
        df = pd.DataFrame(quarterly_earnings)
        
        # Convert date strings to datetime
        if 'fiscalDateEnding' in df.columns:
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            
        # Convert numeric columns
        numeric_columns = ['reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    
    def process_estimates_data(self, estimates_data: Dict) -> pd.DataFrame:
        """
        Process raw earnings estimates data into DataFrame
        
        Args:
            estimates_data: Raw estimates data from API
            
        Returns:
            Processed DataFrame with estimates information
        """
        if not estimates_data or 'quarterlyEstimates' not in estimates_data:
            return pd.DataFrame()
            
        quarterly_estimates = estimates_data['quarterlyEstimates']
        
        df = pd.DataFrame(quarterly_estimates)
        
        # Convert date strings to datetime
        if 'fiscalDateEnding' in df.columns:
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            
        # Convert numeric columns
        numeric_columns = ['estimatedEPS']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    
    def calculate_earnings_surprise(self, earnings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate earnings surprises from processed earnings data
        
        Args:
            earnings_df: Processed earnings DataFrame
            
        Returns:
            DataFrame with calculated surprises
        """
        if earnings_df.empty:
            return pd.DataFrame()
            
        # Calculate surprise if not already present
        if 'surprise' not in earnings_df.columns and 'reportedEPS' in earnings_df.columns and 'estimatedEPS' in earnings_df.columns:
            earnings_df['surprise'] = earnings_df['reportedEPS'] - earnings_df['estimatedEPS']
            
        # Calculate surprise percentage if not already present
        if 'surprisePercentage' not in earnings_df.columns and 'surprise' in earnings_df.columns and 'estimatedEPS' in earnings_df.columns:
            earnings_df['surprisePercentage'] = (earnings_df['surprise'] / earnings_df['estimatedEPS']) * 100
            
        return earnings_df
    
    def save_earnings_data(self, symbol: str, earnings_df: pd.DataFrame, estimates_df: pd.DataFrame, output_dir: str = "data/raw/earnings"):
        """
        Save earnings data to CSV files
        
        Args:
            symbol: Stock ticker symbol
            earnings_df: Processed earnings DataFrame
            estimates_df: Processed estimates DataFrame
            output_dir: Output directory for CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not earnings_df.empty:
            earnings_file = os.path.join(output_dir, f"{symbol}_earnings.csv")
            earnings_df.to_csv(earnings_file, index=False)
            self.logger.info(f"Saved earnings data to {earnings_file}")
            
        if not estimates_df.empty:
            estimates_file = os.path.join(output_dir, f"{symbol}_estimates.csv")
            estimates_df.to_csv(estimates_file, index=False)
            self.logger.info(f"Saved estimates data to {estimates_file}")
    
    def collect_company_earnings(self, symbol: str, output_dir: str = "data/raw/earnings") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect complete earnings data for a company
        
        Args:
            symbol: Stock ticker symbol
            output_dir: Output directory for CSV files
            
        Returns:
            Tuple of (earnings_df, estimates_df)
        """
        self.logger.info(f"Starting earnings collection for {symbol}")
        
        # Get earnings data
        earnings_data = self.get_earnings_data(symbol)
        earnings_df = self.process_earnings_data(earnings_data) if earnings_data else pd.DataFrame()
        
        # Get estimates data
        estimates_data = self.get_earnings_estimates(symbol)
        estimates_df = self.process_estimates_data(estimates_data) if estimates_data else pd.DataFrame()
        
        # Calculate surprises
        earnings_df = self.calculate_earnings_surprise(earnings_df)
        
        # Save data
        self.save_earnings_data(symbol, earnings_df, estimates_df, output_dir)
        
        return earnings_df, estimates_df
    
    def collect_all_earnings(self, symbols: List[str], output_dir: str = "data/raw/earnings") -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Collect earnings data for multiple companies
        
        Args:
            symbols: List of stock ticker symbols
            output_dir: Output directory for CSV files
            
        Returns:
            Dictionary mapping symbols to (earnings_df, estimates_df) tuples
        """
        results = {}
        
        self.logger.info(f"Starting earnings collection for {len(symbols)} symbols")
        
        for i, symbol in enumerate(symbols):
            if self.call_count >= self.max_calls_per_day:
                self.logger.warning(f"Stopping collection - daily API limit reached")
                break
                
            self.logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
            
            try:
                earnings_df, estimates_df = self.collect_company_earnings(symbol, output_dir)
                results[symbol] = (earnings_df, estimates_df)
                
            except Exception as e:
                self.logger.error(f"Failed to collect data for {symbol}: {str(e)}")
                results[symbol] = (pd.DataFrame(), pd.DataFrame())
        
        self.logger.info(f"Completed earnings collection. API calls used: {self.call_count}/{self.max_calls_per_day}")
        
        return results

def main():
    """
    Main function to test the earnings data collector
    """
    # Load API key from environment
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    if not api_key:
        print("Error: ALPHAVANTAGE_API_KEY not found in environment variables")
        return
    
    # Initialize collector
    collector = EarningsDataCollector(api_key)
    
    # Define companies from our dataset
    companies = ['AAPL', 'NVDA', 'MSFT', 'TSLA', 'AMD', 'SONY']
    
    # Collect earnings data
    results = collector.collect_all_earnings(companies)
    
    # Print summary
    print("\nEarnings Data Collection Summary:")
    print("-" * 50)
    for symbol, (earnings_df, estimates_df) in results.items():
        earnings_count = len(earnings_df) if not earnings_df.empty else 0
        estimates_count = len(estimates_df) if not estimates_df.empty else 0
        print(f"{symbol}: {earnings_count} earnings records, {estimates_count} estimate records")

if __name__ == "__main__":
    main()