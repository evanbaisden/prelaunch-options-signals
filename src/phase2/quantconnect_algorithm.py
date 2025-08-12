"""
QuantConnect algorithm template for collecting options data.
This algorithm can be deployed to QuantConnect to collect real options data.
"""
from AlgorithmImports import *
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List


class OptionsDataCollectionAlgorithm(QCAlgorithm):
    """
    QuantConnect algorithm to collect options data for product launch analysis.
    """
    
    def initialize(self):
        """Initialize the algorithm with target stocks and date ranges."""
        
        # Set algorithm parameters
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100000)
        self.universe_settings.asynchronous = True
        
        # Target tickers for analysis
        self.tickers = ['AAPL', 'NVDA', 'MSFT', 'GOOG']
        
        # Options symbols and data storage
        self.option_symbols = {}
        self.options_data = {}
        self.stock_data = {}
        
        # Product launch event dates (approximate)
        self.launch_events = {
            'AAPL': [
                datetime(2020, 10, 13),  # iPhone 12
                datetime(2021, 9, 14),   # iPhone 13
                datetime(2022, 9, 7),    # iPhone 14
                datetime(2023, 9, 12)    # iPhone 15
            ],
            'NVDA': [
                datetime(2020, 9, 1),    # RTX 30 Series
                datetime(2022, 9, 20),   # RTX 40 Series
                datetime(2024, 1, 8)     # RTX 40 SUPER
            ],
            'MSFT': [
                datetime(2020, 9, 9)     # Xbox Series X/S
            ]
        }
        
        # Add stock and options subscriptions
        for ticker in self.tickers:
            # Add stock subscription
            stock = self.add_equity(ticker, Resolution.DAILY)
            
            # Add options subscription
            option = self.add_option(ticker)
            self.option_symbols[ticker] = option.symbol
            
            # Set options filter: strikes from -20% to +20%, expirations up to 6 months
            option.set_filter(-20, +20, 0, 180)
            
            # Initialize data storage
            self.options_data[ticker] = []
            self.stock_data[ticker] = []
        
        # Schedule data collection around launch events
        self.schedule.on(
            self.date_rules.every_day(), 
            self.time_rules.at(10, 0), 
            self.collect_options_data
        )
        
        # Schedule end-of-day summary
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.at(15, 50),
            self.end_of_day_summary
        )
    
    def on_data(self, slice: Slice):
        """Process incoming data."""
        
        # Store stock price data
        for ticker in self.tickers:
            if slice.bars.contains_key(ticker):
                bar = slice.bars[ticker]
                self.stock_data[ticker].append({
                    'time': slice.time,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
        
        # Process options data
        self.process_options_data(slice)
    
    def process_options_data(self, slice: Slice):
        """Process and store options chain data."""
        
        for ticker in self.tickers:
            option_symbol = self.option_symbols.get(ticker)
            if not option_symbol:
                continue
            
            # Get options chain for this ticker
            chain = slice.option_chains.get(option_symbol)
            if not chain:
                continue
            
            # Check if we're near a launch event (within 60 days)
            if not self._is_near_launch_event(ticker, slice.time):
                continue
            
            # Store options data
            options_record = {
                'time': slice.time,
                'underlying_price': self.securities[ticker].price,
                'contracts': []
            }
            
            # Process each contract in the chain
            for contract in chain:
                contract_data = {
                    'symbol': str(contract.symbol),
                    'strike': contract.strike,
                    'expiry': contract.expiry,
                    'option_type': 'call' if contract.right == OptionRight.CALL else 'put',
                    'bid': contract.bid_price,
                    'ask': contract.ask_price,
                    'last_price': contract.last_price,
                    'volume': contract.volume,
                    'open_interest': contract.open_interest,
                    'implied_volatility': contract.implied_volatility,
                    # Greeks (if available)
                    'delta': getattr(contract, 'delta', None),
                    'gamma': getattr(contract, 'gamma', None),
                    'theta': getattr(contract, 'theta', None),
                    'vega': getattr(contract, 'vega', None),
                    'rho': getattr(contract, 'rho', None)
                }
                options_record['contracts'].append(contract_data)
            
            # Store the record
            self.options_data[ticker].append(options_record)
            
            # Log activity
            self.log(f"Collected {len(options_record['contracts'])} option contracts for {ticker} at {slice.time}")
    
    def collect_options_data(self):
        """Scheduled data collection function."""
        current_time = self.time
        
        for ticker in self.tickers:
            if self._is_near_launch_event(ticker, current_time):
                # Request additional historical data if needed
                # This is where you might call self.history() for more data
                pass
    
    def _is_near_launch_event(self, ticker: str, current_time: datetime) -> bool:
        """Check if current time is within 60 days of any launch event."""
        events = self.launch_events.get(ticker, [])
        
        for event_date in events:
            days_to_event = (event_date - current_time).days
            if -30 <= days_to_event <= 60:  # 30 days after to 60 days before
                return True
        
        return False
    
    def end_of_day_summary(self):
        """End of day data summary and export."""
        current_date = self.time.date()
        
        # Log daily summary
        total_contracts = sum(
            len(records[-1]['contracts']) if records else 0
            for records in self.options_data.values()
        )
        
        self.log(f"End of day {current_date}: {total_contracts} total option contracts collected")
        
        # Export data periodically (e.g., weekly)
        if self.time.weekday() == 4:  # Friday
            self.export_collected_data()
    
    def export_collected_data(self):
        """Export collected data to object store or files."""
        try:
            # Export options data
            for ticker, records in self.options_data.items():
                if not records:
                    continue
                
                # Convert to DataFrame format
                export_records = []
                for record in records:
                    for contract in record['contracts']:
                        export_record = {
                            'timestamp': record['time'],
                            'underlying_ticker': ticker,
                            'underlying_price': record['underlying_price'],
                            **contract
                        }
                        export_records.append(export_record)
                
                if export_records:
                    df = pd.DataFrame(export_records)
                    
                    # Save to object store (QuantConnect's cloud storage)
                    filename = f"{ticker}_options_data_{self.time.strftime('%Y%m%d')}.csv"
                    csv_content = df.to_csv(index=False)
                    self.object_store.save(filename, csv_content)
                    
                    self.log(f"Exported {len(export_records)} option records for {ticker}")
            
            # Export stock data
            for ticker, records in self.stock_data.items():
                if not records:
                    continue
                
                df = pd.DataFrame(records)
                filename = f"{ticker}_stock_data_{self.time.strftime('%Y%m%d')}.csv"
                csv_content = df.to_csv(index=False)
                self.object_store.save(filename, csv_content)
                
                self.log(f"Exported {len(records)} stock records for {ticker}")
        
        except Exception as e:
            self.log(f"Error exporting data: {e}")
    
    def on_end_of_algorithm(self):
        """Final export when algorithm ends."""
        self.log("Algorithm ending - performing final data export")
        self.export_collected_data()
        
        # Log final statistics
        total_options_records = sum(len(records) for records in self.options_data.values())
        total_stock_records = sum(len(records) for records in self.stock_data.values())
        
        self.log(f"Final statistics:")
        self.log(f"Total options records: {total_options_records}")
        self.log(f"Total stock records: {total_stock_records}")
        
        for ticker in self.tickers:
            options_count = len(self.options_data.get(ticker, []))
            stock_count = len(self.stock_data.get(ticker, []))
            self.log(f"{ticker}: {options_count} options records, {stock_count} stock records")


class OptionsSignalDetectionAlgorithm(QCAlgorithm):
    """
    QuantConnect algorithm for real-time options signal detection and trading.
    """
    
    def initialize(self):
        """Initialize the signal detection algorithm."""
        
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(1000000)
        
        # Target stocks
        self.tickers = ['AAPL', 'NVDA', 'MSFT']
        self.option_symbols = {}
        
        # Signal detection parameters
        self.put_call_threshold = 0.7
        self.volume_spike_threshold = 2.0
        self.iv_spike_threshold = 1.5
        
        # Historical data for baseline calculations
        self.historical_put_call_ratios = {}
        self.historical_volumes = {}
        
        # Add subscriptions
        for ticker in self.tickers:
            stock = self.add_equity(ticker, Resolution.MINUTE)
            option = self.add_option(ticker)
            self.option_symbols[ticker] = option.symbol
            option.set_filter(-10, +10, 0, 45)  # Shorter-term options
            
            # Initialize historical data
            self.historical_put_call_ratios[ticker] = RollingWindow[float](20)
            self.historical_volumes[ticker] = RollingWindow[float](20)
        
        # Schedule signal detection
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(minutes=15)),
            self.detect_options_signals
        )
    
    def on_data(self, slice: Slice):
        """Process real-time data for signal detection."""
        self.update_historical_data(slice)
    
    def update_historical_data(self, slice: Slice):
        """Update historical data for baseline calculations."""
        
        for ticker in self.tickers:
            option_symbol = self.option_symbols.get(ticker)
            if not option_symbol:
                continue
            
            chain = slice.option_chains.get(option_symbol)
            if not chain:
                continue
            
            # Calculate current put/call ratio
            call_volume = sum(c.volume for c in chain if c.right == OptionRight.CALL)
            put_volume = sum(c.volume for c in chain if c.right == OptionRight.PUT)
            
            if call_volume > 0:
                put_call_ratio = put_volume / call_volume
                self.historical_put_call_ratios[ticker].add(put_call_ratio)
            
            # Calculate total volume
            total_volume = call_volume + put_volume
            if total_volume > 0:
                self.historical_volumes[ticker].add(total_volume)
    
    def detect_options_signals(self):
        """Detect unusual options activity signals."""
        
        for ticker in self.tickers:
            try:
                signals = self._analyze_ticker_signals(ticker)
                
                if signals:
                    self.process_signals(ticker, signals)
            
            except Exception as e:
                self.log(f"Error detecting signals for {ticker}: {e}")
    
    def _analyze_ticker_signals(self, ticker: str) -> List[Dict]:
        """Analyze signals for a specific ticker."""
        signals = []
        
        option_symbol = self.option_symbols.get(ticker)
        if not option_symbol:
            return signals
        
        # Get current options chain
        chain = self.securities[option_symbol].holdings
        # Note: In practice, you'd get this from current slice data
        
        # Placeholder signal detection logic
        # This would be replaced with actual QuantConnect data access
        
        return signals
    
    def process_signals(self, ticker: str, signals: List[Dict]):
        """Process detected signals and potentially place trades."""
        
        for signal in signals:
            signal_strength = signal.get('strength', 0)
            signal_type = signal.get('type', '')
            
            self.log(f"Signal detected for {ticker}: {signal_type} (strength: {signal_strength})")
            
            # Trading logic would go here
            # For now, just log the signal
            
            if abs(signal_strength) > 0.5:
                self.log(f"Strong signal for {ticker} - consider trading action")


# Export classes for use in QuantConnect
__all__ = ['OptionsDataCollectionAlgorithm', 'OptionsSignalDetectionAlgorithm']