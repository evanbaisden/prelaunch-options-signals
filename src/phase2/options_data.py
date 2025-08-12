"""
Options data collection and processing for Phase 2 analysis.
Provides unified interface for multiple options data sources.
"""
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import requests
import time
import math
from scipy.stats import norm
import os
from dotenv import load_dotenv

from ..config import get_config
from ..common.types import LaunchEvent

# Load environment variables
load_dotenv()


def calculate_black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
    """
    Calculate Black-Scholes Greeks for an option.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
    
    Returns:
        Dict with delta, gamma, theta, vega, rho
    """
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    try:
        # Calculate d1 and d2
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        # Standard normal cumulative distribution and probability density
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)  # probability density function
        
        if option_type.lower() == 'call':
            # Call option Greeks
            delta = N_d1
            rho = K * T * math.exp(-r * T) * N_d2 / 100  # Divide by 100 for per-percent change
        else:
            # Put option Greeks
            delta = N_d1 - 1
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
        
        # Common Greeks for both calls and puts
        gamma = n_d1 / (S * sigma * math.sqrt(T))
        theta = (-S * n_d1 * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * (N_d2 if option_type.lower() == 'call' else norm.cdf(-d2))) / 365  # Per day
        vega = S * n_d1 * math.sqrt(T) / 100  # Divide by 100 for per-percent change
        
        return {
            'delta': round(delta, 4),
            'gamma': round(gamma, 4),
            'theta': round(theta, 4),
            'vega': round(vega, 4),
            'rho': round(rho, 4)
        }
        
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        logging.warning(f"Error calculating Greeks: {e}")
        return {'delta': None, 'gamma': None, 'theta': None, 'vega': None, 'rho': None}


def days_to_expiration(expiration_date: date, current_date: date = None) -> float:
    """Calculate time to expiration in years."""
    if current_date is None:
        current_date = date.today()
    
    days_diff = (expiration_date - current_date).days
    return max(0, days_diff / 365.25)  # Account for leap years


@dataclass
class OptionContract:
    """Individual option contract data."""
    symbol: str
    underlying_ticker: str
    strike: float
    expiration: date
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class OptionsChain:
    """Complete options chain for an underlying at a point in time."""
    underlying_ticker: str
    underlying_price: float
    expiration: date
    options: List[OptionContract]
    timestamp: datetime
    
    @property
    def calls(self) -> List[OptionContract]:
        """Get all call options."""
        return [opt for opt in self.options if opt.option_type == 'call']
    
    @property  
    def puts(self) -> List[OptionContract]:
        """Get all put options."""
        return [opt for opt in self.options if opt.option_type == 'put']
    
    @property
    def strikes(self) -> List[float]:
        """Get all unique strike prices."""
        return sorted(list(set(opt.strike for opt in self.options)))


class OptionsDataProvider:
    """Base class for options data providers."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(self.config.raw_data_dir) / "options"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_options_chain(self, ticker: str, expiration: date, date: Optional[date] = None) -> Optional[OptionsChain]:
        """Get options chain for ticker and expiration."""
        raise NotImplementedError("Subclasses must implement get_options_chain")
    
    def get_historical_options_data(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Get historical options data for date range."""
        raise NotImplementedError("Subclasses must implement get_historical_options_data")
    
    def validate_options_data(self, options_df: pd.DataFrame) -> Dict[str, bool]:
        """Validate options data quality."""
        checks = {
            'has_data': len(options_df) > 0,
            'has_required_columns': all(col in options_df.columns for col in [
                'symbol', 'strike', 'expiration', 'option_type', 'volume'
            ]),
            'no_negative_prices': (options_df['bid'] >= 0).all() if 'bid' in options_df.columns else True,
            'reasonable_strikes': True,  # TODO: implement strike validation
            'valid_expirations': True,  # TODO: implement expiration validation
        }
        
        if checks['has_data'] and checks['has_required_columns']:
            checks['reasonable_strikes'] = (
                (options_df['strike'] > 0) & 
                (options_df['strike'] < options_df['strike'].quantile(0.99) * 2)
            ).all()
        
        return checks


class AlphaVantageOptionsProvider(OptionsDataProvider):
    """Alpha Vantage options data provider with historical data support."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is required")
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Alpha Vantage free tier: 25 requests per day, so space them out
    
    def get_historical_options_data(self, ticker: str, date: date, risk_free_rate: float = 0.02) -> pd.DataFrame:
        """Get historical options data for a specific date from Alpha Vantage."""
        try:
            url = f"{self.base_url}?function=HISTORICAL_OPTIONS&symbol={ticker}&date={date.strftime('%Y-%m-%d')}&apikey={self.api_key}"
            
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data or not data['data']:
                self.logger.warning(f"No options data found for {ticker} on {date}")
                return pd.DataFrame()
            
            options_data = data['data']
            
            # Convert to DataFrame
            df = pd.DataFrame(options_data)
            
            # Standardize column names
            column_mapping = {
                'contractID': 'contract_id',
                'symbol': 'underlying_ticker', 
                'expiration': 'expiration',
                'strike': 'strike',
                'type': 'option_type',
                'last': 'last_price',
                'mark': 'mark_price',
                'bid': 'bid',
                'ask': 'ask',
                'bid_size': 'bid_size',
                'ask_size': 'ask_size',
                'volume': 'volume',
                'open_interest': 'open_interest',
                'implied_volatility': 'implied_volatility',
                'delta': 'delta',
                'gamma': 'gamma',
                'theta': 'theta',
                'vega': 'vega',
                'rho': 'rho',
                'date': 'date'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Convert data types
            df['expiration'] = pd.to_datetime(df['expiration']).dt.date
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
            df['last_price'] = pd.to_numeric(df['last_price'], errors='coerce')
            df['bid'] = pd.to_numeric(df['bid'], errors='coerce')
            df['ask'] = pd.to_numeric(df['ask'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
            df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce').fillna(0).astype(int)
            df['implied_volatility'] = pd.to_numeric(df['implied_volatility'], errors='coerce')
            
            # Convert Greeks to numeric, filling missing values
            greek_cols = ['delta', 'gamma', 'theta', 'vega', 'rho']
            for col in greek_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate missing Greeks using Black-Scholes if we have the required data
            if 'implied_volatility' in df.columns:
                # Get underlying price (approximate from options data)
                underlying_price = self._estimate_underlying_price(df)
                
                for idx, row in df.iterrows():
                    if pd.isna(row['implied_volatility']) or row['implied_volatility'] <= 0:
                        continue
                        
                    time_to_exp = days_to_expiration(row['expiration'], row['date'])
                    if time_to_exp <= 0:
                        continue
                    
                    # Calculate Greeks if missing
                    calculated_greeks = calculate_black_scholes_greeks(
                        S=underlying_price,
                        K=row['strike'],
                        T=time_to_exp,
                        r=risk_free_rate,
                        sigma=row['implied_volatility'],
                        option_type=row['option_type']
                    )
                    
                    # Fill missing Greeks
                    for greek, value in calculated_greeks.items():
                        if greek in df.columns and (pd.isna(row[greek]) and value is not None):
                            df.at[idx, greek] = value
            
            df['data_source'] = 'alpha_vantage'
            
            self.logger.info(f"Retrieved {len(df)} options contracts for {ticker} on {date} from Alpha Vantage")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed for {ticker} on {date}: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error getting Alpha Vantage options data for {ticker} on {date}: {e}")
            return pd.DataFrame()
    
    def _estimate_underlying_price(self, options_df: pd.DataFrame) -> float:
        """Estimate underlying price from options data using put-call parity approximation."""
        try:
            # Find ATM options (closest to strike where call and put have similar volumes)
            if len(options_df) == 0:
                return 100.0  # Default fallback
            
            # Simple approach: use the middle strike as approximation
            strikes = sorted(options_df['strike'].dropna().unique())
            if len(strikes) > 0:
                mid_strike = strikes[len(strikes) // 2]
                return mid_strike
            
            return 100.0
            
        except Exception as e:
            self.logger.warning(f"Could not estimate underlying price: {e}")
            return 100.0

    def get_options_chain(self, ticker: str, expiration: date, date: Optional[date] = None) -> Optional[OptionsChain]:
        """Get options chain for specific expiration date."""
        if date is None:
            date = datetime.now().date()
            
        df = self.get_historical_options_data(ticker, date)
        if df.empty:
            return None
        
        # Filter for specific expiration
        df_exp = df[df['expiration'] == expiration]
        if df_exp.empty:
            return None
        
        # Convert to OptionContract objects
        contracts = []
        for _, row in df_exp.iterrows():
            contract = OptionContract(
                symbol=row.get('contract_id', ''),
                underlying_ticker=ticker,
                strike=row['strike'],
                expiration=expiration,
                option_type=row['option_type'],
                bid=row['bid'],
                ask=row['ask'], 
                last_price=row['last_price'],
                volume=row['volume'],
                open_interest=row['open_interest'],
                implied_volatility=row['implied_volatility'],
                delta=row.get('delta'),
                gamma=row.get('gamma'),
                theta=row.get('theta'),
                vega=row.get('vega'),
                timestamp=datetime.combine(date, datetime.min.time())
            )
            contracts.append(contract)
        
        # Estimate underlying price
        underlying_price = self._estimate_underlying_price(df_exp)
        
        return OptionsChain(
            underlying_ticker=ticker,
            underlying_price=underlying_price,
            expiration=expiration,
            options=contracts,
            timestamp=datetime.combine(date, datetime.min.time())
        )


class YahooOptionsProvider(OptionsDataProvider):
    """Yahoo Finance options data provider."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.base_url = "https://query1.finance.yahoo.com/v7/finance/options"
    
    def get_options_chain(self, ticker: str, expiration: date, date: Optional[date] = None) -> Optional[OptionsChain]:
        """Get options chain from Yahoo Finance."""
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            
            # Get options chain
            try:
                options = stock.option_chain(expiration.strftime('%Y-%m-%d'))
            except Exception as e:
                self.logger.warning(f"Could not get options for {ticker} on {expiration}: {e}")
                return None
            
            # Get current stock price
            info = stock.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            contracts = []
            
            # Process calls
            if hasattr(options, 'calls') and not options.calls.empty:
                for _, row in options.calls.iterrows():
                    contract = OptionContract(
                        symbol=row.get('contractSymbol', ''),
                        underlying_ticker=ticker,
                        strike=float(row.get('strike', 0)),
                        expiration=expiration,
                        option_type='call',
                        bid=float(row.get('bid', 0)),
                        ask=float(row.get('ask', 0)),
                        last_price=float(row.get('lastPrice', 0)),
                        volume=int(row.get('volume', 0)) if row.get('volume') else 0,
                        open_interest=int(row.get('openInterest', 0)) if row.get('openInterest') else 0,
                        implied_volatility=float(row.get('impliedVolatility', 0)) if row.get('impliedVolatility') else 0,
                        timestamp=datetime.now()
                    )
                    contracts.append(contract)
            
            # Process puts  
            if hasattr(options, 'puts') and not options.puts.empty:
                for _, row in options.puts.iterrows():
                    contract = OptionContract(
                        symbol=row.get('contractSymbol', ''),
                        underlying_ticker=ticker,
                        strike=float(row.get('strike', 0)),
                        expiration=expiration,
                        option_type='put',
                        bid=float(row.get('bid', 0)),
                        ask=float(row.get('ask', 0)),
                        last_price=float(row.get('lastPrice', 0)),
                        volume=int(row.get('volume', 0)) if row.get('volume') else 0,
                        open_interest=int(row.get('openInterest', 0)) if row.get('openInterest') else 0,
                        implied_volatility=float(row.get('impliedVolatility', 0)) if row.get('impliedVolatility') else 0,
                        timestamp=datetime.now()
                    )
                    contracts.append(contract)
            
            if not contracts:
                self.logger.warning(f"No options contracts found for {ticker} on {expiration}")
                return None
                
            return OptionsChain(
                underlying_ticker=ticker,
                underlying_price=current_price,
                expiration=expiration,
                options=contracts,
                timestamp=datetime.now()
            )
            
        except ImportError:
            self.logger.error("yfinance package required for Yahoo options data")
            return None
        except Exception as e:
            self.logger.error(f"Error getting options chain for {ticker}: {e}")
            return None
    
    def get_historical_options_data(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Get historical options data (placeholder - Yahoo doesn't provide historical options)."""
        self.logger.warning("Yahoo Finance doesn't provide historical options data")
        return pd.DataFrame()


class QuantConnectOptionsProvider(OptionsDataProvider):
    """QuantConnect options data provider."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.qc_api_available = False
        
        # Check if running in QuantConnect environment
        try:
            # This would be available in QuantConnect environment
            import AlgorithmImports
            self.qc_api_available = True
            self.logger.info("QuantConnect API available")
        except ImportError:
            self.logger.warning("QuantConnect API not available - running in simulation mode")
    
    def get_options_chain(self, ticker: str, expiration: date, date: Optional[date] = None) -> Optional[OptionsChain]:
        """Get options chain from QuantConnect."""
        if not self.qc_api_available:
            return self._simulate_options_chain(ticker, expiration)
        
        # TODO: Implement actual QuantConnect options chain retrieval
        # This would be called within a QuantConnect algorithm
        self.logger.warning("QuantConnect options chain retrieval not yet implemented for standalone use")
        return self._simulate_options_chain(ticker, expiration)
    
    def get_historical_options_data(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Get historical options data from QuantConnect."""
        if not self.qc_api_available:
            return self._simulate_historical_options_data(ticker, start_date, end_date)
        
        # TODO: Implement QuantConnect historical data retrieval
        self.logger.warning("QuantConnect historical options not yet implemented for standalone use")
        return self._simulate_historical_options_data(ticker, start_date, end_date)
    
    def _simulate_options_chain(self, ticker: str, expiration: date) -> Optional[OptionsChain]:
        """Simulate options chain data for testing purposes."""
        import random
        
        # Get approximate stock price (placeholder - would use real data)
        stock_prices = {'AAPL': 150, 'NVDA': 400, 'MSFT': 300, 'GOOG': 2500}
        stock_price = stock_prices.get(ticker, 100)
        
        contracts = []
        
        # Generate simulated option contracts around current stock price
        strikes = np.arange(stock_price * 0.8, stock_price * 1.2, stock_price * 0.02)
        
        for strike in strikes:
            # Simulate call option
            call_contract = OptionContract(
                symbol=f"{ticker}_{expiration.strftime('%Y%m%d')}_C_{strike:.0f}",
                underlying_ticker=ticker,
                strike=float(strike),
                expiration=expiration,
                option_type='call',
                bid=max(0.01, stock_price - strike + random.uniform(-5, 5)),
                ask=max(0.02, stock_price - strike + random.uniform(-3, 7)),
                last_price=max(0.01, stock_price - strike + random.uniform(-4, 6)),
                volume=random.randint(0, 1000),
                open_interest=random.randint(0, 5000),
                implied_volatility=random.uniform(0.15, 0.35),
                timestamp=datetime.now()
            )
            
            # Simulate put option
            put_contract = OptionContract(
                symbol=f"{ticker}_{expiration.strftime('%Y%m%d')}_P_{strike:.0f}",
                underlying_ticker=ticker,
                strike=float(strike),
                expiration=expiration,
                option_type='put',
                bid=max(0.01, strike - stock_price + random.uniform(-5, 5)),
                ask=max(0.02, strike - stock_price + random.uniform(-3, 7)),
                last_price=max(0.01, strike - stock_price + random.uniform(-4, 6)),
                volume=random.randint(0, 1000),
                open_interest=random.randint(0, 5000),
                implied_volatility=random.uniform(0.18, 0.40),
                timestamp=datetime.now()
            )
            
            contracts.extend([call_contract, put_contract])
        
        return OptionsChain(
            underlying_ticker=ticker,
            underlying_price=stock_price,
            expiration=expiration,
            options=contracts,
            timestamp=datetime.now()
        )
    
    def _simulate_historical_options_data(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Simulate historical options data for testing."""
        # Generate placeholder historical options data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        records = []
        for date in date_range:
            if date.weekday() < 5:  # Weekdays only
                # Simulate a few option contracts for each day
                for i in range(10):
                    records.append({
                        'date': date.date(),
                        'symbol': f"{ticker}_option_{i}",
                        'strike': 100 + i * 5,
                        'option_type': 'call' if i % 2 == 0 else 'put',
                        'volume': np.random.randint(0, 500),
                        'open_interest': np.random.randint(0, 2000),
                        'implied_volatility': np.random.uniform(0.15, 0.35)
                    })
        
        return pd.DataFrame(records)


class PolygonOptionsProvider(OptionsDataProvider):
    """Polygon.io options data provider."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = getattr(config, 'polygon_api_key', None)
        self.base_url = "https://api.polygon.io"
        
        if not self.api_key:
            self.logger.warning("Polygon API key not configured")
    
    def get_options_chain(self, ticker: str, expiration: date, date: Optional[date] = None) -> Optional[OptionsChain]:
        """Get options chain from Polygon.io."""
        if not self.api_key:
            self.logger.error("Polygon API key required")
            return None
        
        # TODO: Implement Polygon options chain API
        self.logger.warning("Polygon options chain not yet implemented")
        return None
    
    def get_historical_options_data(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Get historical options data from Polygon.io."""
        if not self.api_key:
            self.logger.error("Polygon API key required")
            return pd.DataFrame()
        
        # TODO: Implement Polygon historical options API
        self.logger.warning("Polygon historical options not yet implemented")  
        return pd.DataFrame()


class OptionsDataManager:
    """Centralized options data management."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers (only include available ones)
        self.providers = {}
        
        # Try to initialize Alpha Vantage (primary for historical data)
        try:
            self.providers['alpha_vantage'] = AlphaVantageOptionsProvider(config)
            self.logger.info("Alpha Vantage options provider initialized")
        except Exception as e:
            self.logger.warning(f"Alpha Vantage options provider not available: {e}")
        
        # Try to initialize Polygon (secondary, good for recent data)
        try:
            self.providers['polygon'] = PolygonOptionsProvider(config)
            self.logger.info("Polygon options provider initialized")
        except Exception as e:
            self.logger.warning(f"Polygon options provider not available: {e}")
        
        # Always available Yahoo Finance (backup)
        try:
            self.providers['yahoo'] = YahooOptionsProvider(config)
            self.logger.info("Yahoo Finance options provider initialized")
        except Exception as e:
            self.logger.warning(f"Yahoo Finance options provider not available: {e}")
        
        # Try to initialize QuantConnect (premium)
        try:
            self.providers['quantconnect'] = QuantConnectOptionsProvider(config)
            self.logger.info("QuantConnect options provider initialized")
        except Exception as e:
            self.logger.warning(f"QuantConnect options provider not available: {e}")
        
        # Provider priority: Alpha Vantage (historical), Polygon (recent), Yahoo (backup), QuantConnect (premium)
        self.provider_order = [name for name in ['alpha_vantage', 'polygon', 'yahoo', 'quantconnect'] 
                               if name in self.providers]
        
        self.logger.info(f"Initialized options providers: {list(self.providers.keys())}")
        self.logger.info(f"Provider priority order: {self.provider_order}")
    
    def get_options_for_event(self, event: LaunchEvent, days_before: int = 60) -> Dict[date, OptionsChain]:
        """Get options data for dates leading up to a launch event."""
        options_data = {}
        
        # Generate dates for data collection
        event_date = event.announcement if event.announcement else event.release
        start_date = event_date - timedelta(days=days_before)
        
        # Get available expiration dates around the event
        expirations = self._get_event_expirations(event)
        
        for exp_date in expirations:
            # Try each provider until we get data
            for provider_name in self.provider_order:
                provider = self.providers[provider_name]
                
                try:
                    chain = provider.get_options_chain(event.ticker, exp_date)
                    if chain and chain.options:
                        options_data[exp_date] = chain
                        self.logger.info(f"Got {len(chain.options)} options for {event.ticker} exp {exp_date} from {provider_name}")
                        break
                except Exception as e:
                    self.logger.warning(f"Provider {provider_name} failed for {event.ticker}: {e}")
                    continue
        
        return options_data
    
    def cross_validate_options_data(self, ticker: str, date: date, max_providers: int = 2) -> Dict[str, pd.DataFrame]:
        """Cross-validate options data across multiple providers."""
        validation_results = {}
        
        # Get data from multiple providers
        for i, provider_name in enumerate(self.provider_order):
            if i >= max_providers:
                break
                
            provider = self.providers[provider_name]
            
            try:
                if provider_name == 'alpha_vantage':
                    # Alpha Vantage has direct historical data method
                    df = provider.get_historical_options_data(ticker, date)
                else:
                    # Other providers might need different approaches
                    # For now, skip or implement different logic
                    continue
                    
                if not df.empty:
                    validation_results[provider_name] = df
                    self.logger.info(f"Cross-validation: {provider_name} returned {len(df)} contracts")
                    
            except Exception as e:
                self.logger.warning(f"Cross-validation failed for {provider_name}: {e}")
                continue
        
        # Compare results if we have multiple providers
        if len(validation_results) > 1:
            self._compare_provider_data(validation_results, ticker, date)
        
        return validation_results
    
    def _compare_provider_data(self, provider_data: Dict[str, pd.DataFrame], ticker: str, date: date):
        """Compare data quality across providers."""
        provider_names = list(provider_data.keys())
        
        for i, provider1 in enumerate(provider_names):
            for provider2 in provider_names[i+1:]:
                df1 = provider_data[provider1]
                df2 = provider_data[provider2]
                
                # Compare contract counts
                self.logger.info(f"Cross-validation {provider1} vs {provider2}: {len(df1)} vs {len(df2)} contracts")
                
                # Compare overlapping strikes for same expiration
                if 'strike' in df1.columns and 'strike' in df2.columns:
                    common_strikes = set(df1['strike']) & set(df2['strike'])
                    if common_strikes:
                        self.logger.info(f"Common strikes between {provider1} and {provider2}: {len(common_strikes)}")
                        
                        # Compare prices for common strikes (if available)
                        for strike in list(common_strikes)[:3]:  # Check first 3
                            records1 = df1[df1['strike'] == strike]
                            records2 = df2[df2['strike'] == strike]
                            
                            if len(records1) > 0 and len(records2) > 0:
                                if 'last_price' in df1.columns and 'last_price' in df2.columns:
                                    price1 = records1.iloc[0]['last_price']
                                    price2 = records2.iloc[0]['last_price']
                                    if price1 > 0 and price2 > 0:
                                        price_diff = abs(price1 - price2) / max(price1, price2)
                                        self.logger.info(f"Price comparison for strike {strike}: {price1:.2f} vs {price2:.2f} (diff: {price_diff:.1%})")
    
    def _get_event_expirations(self, event: LaunchEvent) -> List[date]:
        """Get relevant option expiration dates for an event."""
        # For now, return some standard monthly expirations
        # TODO: Make this more sophisticated based on actual available expirations
        
        event_date = event.announcement if event.announcement else event.release
        expirations = []
        
        # Get next few monthly expirations after event
        current = event_date.replace(day=1)  # First of month
        for i in range(3):  # Next 3 months
            # Third Friday of month (standard option expiration)
            third_friday = current + timedelta(days=14)  # Approximate
            while third_friday.weekday() != 4:  # Friday
                third_friday += timedelta(days=1)
            
            if third_friday > event_date:
                expirations.append(third_friday.date())
            
            # Next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return expirations
    
    def save_options_data(self, options_data: Dict[date, OptionsChain], event_name: str):
        """Save options data to CSV files."""
        for exp_date, chain in options_data.items():
            filename = f"{event_name}_{exp_date.strftime('%Y%m%d')}_options.csv"
            filepath = self.config.raw_data_dir / "options" / filename
            
            # Convert to DataFrame
            records = []
            for option in chain.options:
                records.append({
                    'symbol': option.symbol,
                    'underlying_ticker': option.underlying_ticker,
                    'strike': option.strike,
                    'expiration': option.expiration,
                    'option_type': option.option_type,
                    'bid': option.bid,
                    'ask': option.ask,
                    'last_price': option.last_price,
                    'volume': option.volume,
                    'open_interest': option.open_interest,
                    'implied_volatility': option.implied_volatility,
                    'underlying_price': chain.underlying_price,
                    'timestamp': option.timestamp
                })
            
            df = pd.DataFrame(records)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved {len(df)} options to {filepath}")
    
    def load_options_data(self, event_name: str, exp_date: date) -> Optional[OptionsChain]:
        """Load options data from CSV file."""
        filename = f"{event_name}_{exp_date.strftime('%Y%m%d')}_options.csv"
        filepath = Path(self.config.raw_data_dir) / "options" / filename
        
        if not filepath.exists():
            return None
        
        try:
            df = pd.read_csv(filepath)
            df['expiration'] = pd.to_datetime(df['expiration']).dt.date
            
            contracts = []
            for _, row in df.iterrows():
                contract = OptionContract(
                    symbol=row['symbol'],
                    underlying_ticker=row['underlying_ticker'],
                    strike=row['strike'],
                    expiration=row['expiration'],
                    option_type=row['option_type'],
                    bid=row['bid'],
                    ask=row['ask'],
                    last_price=row['last_price'],
                    volume=row['volume'],
                    open_interest=row['open_interest'],
                    implied_volatility=row['implied_volatility'],
                    timestamp=pd.to_datetime(row['timestamp']) if 'timestamp' in row else None
                )
                contracts.append(contract)
            
            return OptionsChain(
                underlying_ticker=df.iloc[0]['underlying_ticker'],
                underlying_price=df.iloc[0]['underlying_price'],
                expiration=exp_date,
                options=contracts,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error loading options data from {filepath}: {e}")
            return None