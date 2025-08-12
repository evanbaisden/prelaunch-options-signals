"""
Earnings data collection and analysis for correlation testing.
Integrates earnings surprises with options signals analysis.
"""
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import requests
from pathlib import Path

from ..config import get_config
from ..common.types import LaunchEvent


@dataclass
class EarningsEvent:
    """Individual earnings announcement data."""
    ticker: str
    earnings_date: date
    report_date: date  # When earnings were actually reported
    fiscal_period: str  # Q1, Q2, Q3, Q4
    fiscal_year: int
    
    # Consensus estimates
    eps_estimate: float
    revenue_estimate: float
    
    # Actual results
    eps_actual: float
    revenue_actual: float
    
    # Surprises (actual - estimate)
    eps_surprise: float
    revenue_surprise: float
    
    # Surprise percentages
    eps_surprise_pct: float
    revenue_surprise_pct: float
    
    # Market reaction
    price_reaction_1day: Optional[float] = None
    price_reaction_3day: Optional[float] = None
    volume_reaction: Optional[float] = None


@dataclass
class EarningsAnalysis:
    """Earnings analysis results for a product launch event."""
    event: LaunchEvent
    earnings_before_event: List[EarningsEvent]
    earnings_after_event: List[EarningsEvent]
    
    # Analysis metrics
    avg_eps_surprise_before: float
    avg_eps_surprise_after: float
    avg_revenue_surprise_before: float
    avg_revenue_surprise_after: float
    
    # Correlation metrics
    surprise_improvement: bool
    surprise_significance: float  # Statistical significance of change


class EarningsDataProvider:
    """Base class for earnings data providers."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
    
    def get_earnings_calendar(self, ticker: str, start_date: date, end_date: date) -> List[EarningsEvent]:
        """Get earnings calendar for ticker and date range."""
        raise NotImplementedError("Subclasses must implement get_earnings_calendar")
    
    def get_earnings_estimates(self, ticker: str, fiscal_period: str, fiscal_year: int) -> Dict:
        """Get analyst estimates for specific earnings period."""
        raise NotImplementedError("Subclasses must implement get_earnings_estimates")


class AlphaVantageEarningsProvider(EarningsDataProvider):
    """Alpha Vantage earnings data provider."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = getattr(config, 'alpha_vantage_api_key', None)
        self.base_url = "https://www.alphavantage.co/query"
        
        if not self.api_key:
            self.logger.warning("Alpha Vantage API key not configured")
    
    def get_earnings_calendar(self, ticker: str, start_date: date, end_date: date) -> List[EarningsEvent]:
        """Get earnings calendar from Alpha Vantage."""
        if not self.api_key:
            self.logger.error("Alpha Vantage API key required")
            return []
        
        try:
            # Get earnings data
            params = {
                'function': 'EARNINGS',
                'symbol': ticker,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                self.logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return []
            
            earnings_events = []
            
            # Process quarterly earnings
            if 'quarterlyEarnings' in data:
                for earnings in data['quarterlyEarnings']:
                    try:
                        report_date = datetime.strptime(earnings['reportedDate'], '%Y-%m-%d').date()
                        
                        # Filter by date range
                        if start_date <= report_date <= end_date:
                            event = EarningsEvent(
                                ticker=ticker,
                                earnings_date=report_date,  # Alpha Vantage doesn't separate announcement vs report
                                report_date=report_date,
                                fiscal_period=earnings.get('fiscalDateEnding', '')[-2:],  # Extract quarter
                                fiscal_year=int(earnings.get('fiscalDateEnding', '2000')[:4]),
                                eps_estimate=float(earnings.get('estimatedEPS', 0)),
                                revenue_estimate=0,  # Alpha Vantage doesn't provide revenue estimates
                                eps_actual=float(earnings.get('reportedEPS', 0)),
                                revenue_actual=0,  # Alpha Vantage doesn't provide revenue actuals
                                eps_surprise=float(earnings.get('surprise', 0)),
                                revenue_surprise=0,
                                eps_surprise_pct=float(earnings.get('surprisePercentage', 0)),
                                revenue_surprise_pct=0
                            )
                            earnings_events.append(event)
                    
                    except (ValueError, KeyError) as e:
                        self.logger.warning(f"Error processing earnings data for {ticker}: {e}")
                        continue
            
            return earnings_events
            
        except Exception as e:
            self.logger.error(f"Error fetching earnings data for {ticker}: {e}")
            return []


class YahooEarningsProvider(EarningsDataProvider):
    """Yahoo Finance earnings data provider."""
    
    def get_earnings_calendar(self, ticker: str, start_date: date, end_date: date) -> List[EarningsEvent]:
        """Get earnings calendar from Yahoo Finance."""
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            
            # Get earnings data
            earnings = stock.earnings_dates
            
            if earnings is None or earnings.empty:
                self.logger.warning(f"No earnings data found for {ticker}")
                return []
            
            earnings_events = []
            
            for earnings_date, row in earnings.iterrows():
                try:
                    if start_date <= earnings_date.date() <= end_date:
                        # Yahoo provides limited earnings data
                        event = EarningsEvent(
                            ticker=ticker,
                            earnings_date=earnings_date.date(),
                            report_date=earnings_date.date(),
                            fiscal_period='',  # Yahoo doesn't provide this
                            fiscal_year=earnings_date.year,
                            eps_estimate=float(row.get('EPS Estimate', 0)) if pd.notna(row.get('EPS Estimate')) else 0,
                            revenue_estimate=float(row.get('Revenue Estimate', 0)) if pd.notna(row.get('Revenue Estimate')) else 0,
                            eps_actual=float(row.get('Reported EPS', 0)) if pd.notna(row.get('Reported EPS')) else 0,
                            revenue_actual=float(row.get('Reported Revenue', 0)) if pd.notna(row.get('Reported Revenue')) else 0,
                            eps_surprise=0,  # Calculate below
                            revenue_surprise=0,  # Calculate below
                            eps_surprise_pct=0,  # Calculate below
                            revenue_surprise_pct=0  # Calculate below
                        )
                        
                        # Calculate surprises
                        if event.eps_estimate != 0:
                            event.eps_surprise = event.eps_actual - event.eps_estimate
                            event.eps_surprise_pct = (event.eps_surprise / abs(event.eps_estimate)) * 100
                        
                        if event.revenue_estimate != 0:
                            event.revenue_surprise = event.revenue_actual - event.revenue_estimate
                            event.revenue_surprise_pct = (event.revenue_surprise / abs(event.revenue_estimate)) * 100
                        
                        earnings_events.append(event)
                
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Error processing earnings data for {ticker}: {e}")
                    continue
            
            return earnings_events
            
        except ImportError:
            self.logger.error("yfinance package required for Yahoo earnings data")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching earnings data for {ticker}: {e}")
            return []


class EarningsAnalyzer:
    """Analyzes earnings data for correlation with product launch events."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize earnings data providers
        self.providers = {
            'alpha_vantage': AlphaVantageEarningsProvider(config),
            'yahoo': YahooEarningsProvider(config)
        }
        
        self.provider_order = ['alpha_vantage', 'yahoo']
    
    def analyze_earnings_around_event(self, event: LaunchEvent, lookback_quarters: int = 4) -> EarningsAnalysis:
        """Analyze earnings patterns around a product launch event."""
        
        # Define analysis windows
        event_date = event.announcement if event.announcement else event.release
        analysis_start = event_date - timedelta(days=365 * 2)  # 2 years before
        analysis_end = event_date + timedelta(days=365)  # 1 year after
        
        # Get earnings data
        earnings_data = self._get_earnings_data(event.ticker, analysis_start, analysis_end)
        
        if not earnings_data:
            self.logger.warning(f"No earnings data found for {event.ticker}")
            return EarningsAnalysis(
                event=event,
                earnings_before_event=[],
                earnings_after_event=[],
                avg_eps_surprise_before=0,
                avg_eps_surprise_after=0,
                avg_revenue_surprise_before=0,
                avg_revenue_surprise_after=0,
                surprise_improvement=False,
                surprise_significance=0
            )
        
        # Split earnings into before/after event
        earnings_before = [e for e in earnings_data if e.earnings_date < event_date]
        earnings_after = [e for e in earnings_data if e.earnings_date >= event_date]
        
        # Calculate metrics
        avg_eps_surprise_before = np.mean([e.eps_surprise_pct for e in earnings_before[-lookback_quarters:]]) if earnings_before else 0
        avg_eps_surprise_after = np.mean([e.eps_surprise_pct for e in earnings_after[:lookback_quarters]]) if earnings_after else 0
        
        avg_revenue_surprise_before = np.mean([e.revenue_surprise_pct for e in earnings_before[-lookback_quarters:]]) if earnings_before else 0
        avg_revenue_surprise_after = np.mean([e.revenue_surprise_pct for e in earnings_after[:lookback_quarters]]) if earnings_after else 0
        
        # Test for improvement
        surprise_improvement = (avg_eps_surprise_after > avg_eps_surprise_before) or (avg_revenue_surprise_after > avg_revenue_surprise_before)
        
        # Statistical significance test (simple t-test approximation)
        surprise_significance = self._calculate_surprise_significance(earnings_before, earnings_after, lookback_quarters)
        
        return EarningsAnalysis(
            event=event,
            earnings_before_event=earnings_before[-lookback_quarters:],
            earnings_after_event=earnings_after[:lookback_quarters],
            avg_eps_surprise_before=avg_eps_surprise_before,
            avg_eps_surprise_after=avg_eps_surprise_after,
            avg_revenue_surprise_before=avg_revenue_surprise_before,
            avg_revenue_surprise_after=avg_revenue_surprise_after,
            surprise_improvement=surprise_improvement,
            surprise_significance=surprise_significance
        )
    
    def correlate_options_with_earnings(
        self, 
        earnings_analysis: EarningsAnalysis,
        options_signals: List[Dict]  # Options signals from Phase 2 analysis
    ) -> Dict:
        """Correlate options signals with earnings surprise predictions."""
        
        correlation_results = {
            'earnings_predictive_power': 0,
            'options_signals_correlation': 0,
            'combined_signal_strength': 0,
            'statistical_significance': 0,
            'recommendations': []
        }
        
        if not options_signals or not earnings_analysis.earnings_after_event:
            return correlation_results
        
        # TODO: Implement sophisticated correlation analysis
        # For now, return placeholder results
        
        # Simple correlation based on directional agreement
        earnings_direction = 'positive' if earnings_analysis.avg_eps_surprise_after > 0 else 'negative'
        
        # Count agreeing options signals
        agreeing_signals = 0
        total_signals = len(options_signals)
        
        for signal in options_signals:
            signal_direction = signal.get('direction', 'neutral')
            if (earnings_direction == 'positive' and signal_direction == 'bullish') or \
               (earnings_direction == 'negative' and signal_direction == 'bearish'):
                agreeing_signals += 1
        
        if total_signals > 0:
            correlation_results['options_signals_correlation'] = agreeing_signals / total_signals
        
        # Generate recommendations
        if correlation_results['options_signals_correlation'] > 0.6:
            correlation_results['recommendations'].append("Strong correlation between options signals and earnings outcomes")
        
        if earnings_analysis.surprise_improvement:
            correlation_results['recommendations'].append("Product launch appears to correlate with improved earnings performance")
        
        return correlation_results
    
    def _get_earnings_data(self, ticker: str, start_date: date, end_date: date) -> List[EarningsEvent]:
        """Get earnings data from available providers."""
        
        for provider_name in self.provider_order:
            provider = self.providers[provider_name]
            
            try:
                earnings_data = provider.get_earnings_calendar(ticker, start_date, end_date)
                if earnings_data:
                    self.logger.info(f"Got {len(earnings_data)} earnings events for {ticker} from {provider_name}")
                    return earnings_data
            except Exception as e:
                self.logger.warning(f"Provider {provider_name} failed for {ticker}: {e}")
                continue
        
        return []
    
    def _calculate_surprise_significance(
        self, 
        earnings_before: List[EarningsEvent],
        earnings_after: List[EarningsEvent],
        lookback_quarters: int
    ) -> float:
        """Calculate statistical significance of earnings surprise changes."""
        
        if len(earnings_before) < 2 or len(earnings_after) < 2:
            return 0
        
        # Get recent earnings before and after
        before_surprises = [e.eps_surprise_pct for e in earnings_before[-lookback_quarters:]]
        after_surprises = [e.eps_surprise_pct for e in earnings_after[:lookback_quarters]]
        
        if len(before_surprises) < 2 or len(after_surprises) < 2:
            return 0
        
        # Simple t-test approximation
        mean_before = np.mean(before_surprises)
        mean_after = np.mean(after_surprises)
        
        std_before = np.std(before_surprises, ddof=1)
        std_after = np.std(after_surprises, ddof=1)
        
        n_before = len(before_surprises)
        n_after = len(after_surprises)
        
        # Pooled standard error
        pooled_se = np.sqrt((std_before**2 / n_before) + (std_after**2 / n_after))
        
        if pooled_se == 0:
            return 0
        
        # T-statistic
        t_stat = (mean_after - mean_before) / pooled_se
        
        # Return absolute t-statistic as significance measure
        return abs(t_stat)
    
    def save_earnings_data(self, earnings_events: List[EarningsEvent], ticker: str):
        """Save earnings data to CSV file."""
        if not earnings_events:
            return
        
        filename = f"{ticker}_earnings_data.csv"
        filepath = Path(self.config.raw_data_dir) / "earnings" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        records = []
        for event in earnings_events:
            records.append({
                'ticker': event.ticker,
                'earnings_date': event.earnings_date,
                'report_date': event.report_date,
                'fiscal_period': event.fiscal_period,
                'fiscal_year': event.fiscal_year,
                'eps_estimate': event.eps_estimate,
                'revenue_estimate': event.revenue_estimate,
                'eps_actual': event.eps_actual,
                'revenue_actual': event.revenue_actual,
                'eps_surprise': event.eps_surprise,
                'revenue_surprise': event.revenue_surprise,
                'eps_surprise_pct': event.eps_surprise_pct,
                'revenue_surprise_pct': event.revenue_surprise_pct
            })
        
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved {len(df)} earnings events to {filepath}")
    
    def load_earnings_data(self, ticker: str) -> List[EarningsEvent]:
        """Load earnings data from CSV file."""
        filename = f"{ticker}_earnings_data.csv"
        filepath = Path(self.config.raw_data_dir) / "earnings" / filename
        
        if not filepath.exists():
            return []
        
        try:
            df = pd.read_csv(filepath)
            df['earnings_date'] = pd.to_datetime(df['earnings_date']).dt.date
            df['report_date'] = pd.to_datetime(df['report_date']).dt.date
            
            earnings_events = []
            for _, row in df.iterrows():
                event = EarningsEvent(
                    ticker=row['ticker'],
                    earnings_date=row['earnings_date'],
                    report_date=row['report_date'],
                    fiscal_period=row['fiscal_period'],
                    fiscal_year=int(row['fiscal_year']),
                    eps_estimate=float(row['eps_estimate']),
                    revenue_estimate=float(row['revenue_estimate']),
                    eps_actual=float(row['eps_actual']),
                    revenue_actual=float(row['revenue_actual']),
                    eps_surprise=float(row['eps_surprise']),
                    revenue_surprise=float(row['revenue_surprise']),
                    eps_surprise_pct=float(row['eps_surprise_pct']),
                    revenue_surprise_pct=float(row['revenue_surprise_pct'])
                )
                earnings_events.append(event)
            
            return earnings_events
            
        except Exception as e:
            self.logger.error(f"Error loading earnings data from {filepath}: {e}")
            return []