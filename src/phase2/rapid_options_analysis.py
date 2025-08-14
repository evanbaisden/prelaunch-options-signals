"""
Rapid Options Analysis Implementation
Simplified options flow analysis to complement the equity event study.
"""
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import yfinance as yf
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OptionsFlowMetrics:
    """Options flow metrics for an event."""
    ticker: str
    event_date: date
    days_to_event: int
    
    # Volume metrics
    total_call_volume: float
    total_put_volume: float
    put_call_volume_ratio: float
    
    # Open interest metrics
    total_call_oi: float
    total_put_oi: float
    put_call_oi_ratio: float
    
    # Unusual activity indicators
    volume_spike: bool
    oi_spike: bool
    unusual_activity_score: float
    
    # Implied volatility
    avg_call_iv: Optional[float] = None
    avg_put_iv: Optional[float] = None
    iv_skew: Optional[float] = None
    
    # Price metrics
    underlying_price: Optional[float] = None
    timestamp: Optional[datetime] = None


class RapidOptionsAnalyzer:
    """Simplified options analysis using yfinance options data."""
    
    def __init__(self):
        self.cache_dir = Path("data/raw/options")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_options_data(self, ticker: str, event_date: date, days_before: int = 30) -> List[OptionsFlowMetrics]:
        """
        Get options flow metrics for a ticker around an event date.
        Uses yfinance for recent data availability.
        """
        logger.info(f"Analyzing options for {ticker} around {event_date}")
        
        try:
            stock = yf.Ticker(ticker)
            
            # Get options expirations
            expirations = stock.options
            if not expirations:
                logger.warning(f"No options data available for {ticker}")
                return []
            
            # Find expiration closest to but after event date
            event_datetime = pd.to_datetime(event_date)
            exp_dates = [pd.to_datetime(exp) for exp in expirations]
            future_exps = [exp for exp in exp_dates if exp >= event_datetime]
            
            if not future_exps:
                logger.warning(f"No future expirations found for {ticker} after {event_date}")
                return []
            
            closest_exp = min(future_exps, key=lambda x: abs((x - event_datetime).days))
            exp_str = closest_exp.strftime('%Y-%m-%d')
            
            # Get options chain for the closest expiration
            options_chain = stock.option_chain(exp_str)
            calls = options_chain.calls
            puts = options_chain.puts
            
            if calls.empty and puts.empty:
                logger.warning(f"Empty options chain for {ticker} expiration {exp_str}")
                return []
            
            # Calculate metrics (simplified since we can't get historical data easily)
            metrics = self._calculate_flow_metrics(
                ticker, event_date, calls, puts, exp_str
            )
            
            return [metrics] if metrics else []
            
        except Exception as e:
            logger.error(f"Error getting options data for {ticker}: {e}")
            return []
    
    def _calculate_flow_metrics(self, ticker: str, event_date: date, 
                              calls: pd.DataFrame, puts: pd.DataFrame,
                              expiration: str) -> Optional[OptionsFlowMetrics]:
        """Calculate options flow metrics from options chain data."""
        
        try:
            # Basic volume and open interest metrics
            total_call_volume = calls['volume'].fillna(0).sum() if not calls.empty else 0
            total_put_volume = puts['volume'].fillna(0).sum() if not puts.empty else 0
            
            total_call_oi = calls['openInterest'].fillna(0).sum() if not calls.empty else 0
            total_put_oi = puts['openInterest'].fillna(0).sum() if not puts.empty else 0
            
            # Put/call ratios
            pcr_volume = total_put_volume / max(total_call_volume, 1)
            pcr_oi = total_put_oi / max(total_call_oi, 1)
            
            # Implied volatility metrics
            avg_call_iv = calls['impliedVolatility'].mean() if not calls.empty else None
            avg_put_iv = puts['impliedVolatility'].mean() if not puts.empty else None
            iv_skew = (avg_put_iv - avg_call_iv) if (avg_call_iv and avg_put_iv) else None
            
            # Unusual activity detection (simplified)
            total_volume = total_call_volume + total_put_volume
            total_oi = total_call_oi + total_put_oi
            
            # Simple thresholds for unusual activity
            volume_spike = total_volume > 1000  # Simplified threshold
            oi_spike = total_oi > 5000  # Simplified threshold
            
            # Unusual activity score (0-1)
            volume_score = min(total_volume / 10000, 1.0)  # Normalize
            pcr_score = min(abs(pcr_volume - 1.0), 1.0)  # Deviation from neutral
            unusual_activity_score = (volume_score + pcr_score) / 2
            
            # Days to event (estimate)
            days_to_event = max((event_date - date.today()).days, 0)
            
            return OptionsFlowMetrics(
                ticker=ticker,
                event_date=event_date,
                days_to_event=days_to_event,
                total_call_volume=total_call_volume,
                total_put_volume=total_put_volume,
                put_call_volume_ratio=pcr_volume,
                total_call_oi=total_call_oi,
                total_put_oi=total_put_oi,
                put_call_oi_ratio=pcr_oi,
                volume_spike=volume_spike,
                oi_spike=oi_spike,
                unusual_activity_score=unusual_activity_score,
                avg_call_iv=avg_call_iv,
                avg_put_iv=avg_put_iv,
                iv_skew=iv_skew,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating flow metrics for {ticker}: {e}")
            return None
    
    def analyze_events_options_flow(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze options flow for all events in the dataset."""
        logger.info("Starting options flow analysis for all events...")
        
        options_results = []
        
        for _, event in events_df.iterrows():
            ticker = event['ticker']
            event_date = event['announcement']
            
            # Skip very old events where options data won't be available
            if isinstance(event_date, str):
                event_date = pd.to_datetime(event_date).date()
            
            # Only analyze recent events (yfinance limitations)
            days_ago = (date.today() - event_date).days
            if days_ago > 365:  # Skip events older than 1 year
                logger.info(f"Skipping {event['name']} ({event_date}) - too old for current options data")
                continue
            
            flow_metrics = self.get_options_data(ticker, event_date)
            
            for metrics in flow_metrics:
                options_results.append({
                    'event_name': event['name'],
                    'company': event['company'], 
                    'ticker': metrics.ticker,
                    'event_date': metrics.event_date,
                    'days_to_event': metrics.days_to_event,
                    'total_call_volume': metrics.total_call_volume,
                    'total_put_volume': metrics.total_put_volume,
                    'put_call_volume_ratio': metrics.put_call_volume_ratio,
                    'total_call_oi': metrics.total_call_oi,
                    'total_put_oi': metrics.total_put_oi,
                    'put_call_oi_ratio': metrics.put_call_oi_ratio,
                    'volume_spike': metrics.volume_spike,
                    'oi_spike': metrics.oi_spike,
                    'unusual_activity_score': metrics.unusual_activity_score,
                    'avg_call_iv': metrics.avg_call_iv,
                    'avg_put_iv': metrics.avg_put_iv,
                    'iv_skew': metrics.iv_skew,
                    'underlying_price': metrics.underlying_price,
                    'analysis_timestamp': metrics.timestamp
                })
        
        return pd.DataFrame(options_results)
    
    def generate_options_summary(self, options_df: pd.DataFrame) -> Dict:
        """Generate summary statistics for options analysis."""
        if options_df.empty:
            return {'error': 'No options data available for analysis'}
        
        summary = {
            'sample_size': {
                'total_events_analyzed': len(options_df),
                'events_with_volume_spikes': options_df['volume_spike'].sum(),
                'events_with_oi_spikes': options_df['oi_spike'].sum(),
                'unique_tickers': options_df['ticker'].nunique()
            },
            'volume_metrics': {
                'avg_call_volume': options_df['total_call_volume'].mean(),
                'avg_put_volume': options_df['total_put_volume'].mean(),
                'avg_pcr_volume': options_df['put_call_volume_ratio'].mean(),
                'median_pcr_volume': options_df['put_call_volume_ratio'].median()
            },
            'open_interest_metrics': {
                'avg_call_oi': options_df['total_call_oi'].mean(),
                'avg_put_oi': options_df['total_put_oi'].mean(),
                'avg_pcr_oi': options_df['put_call_oi_ratio'].mean(),
                'median_pcr_oi': options_df['put_call_oi_ratio'].median()
            },
            'unusual_activity': {
                'avg_activity_score': options_df['unusual_activity_score'].mean(),
                'high_activity_events': (options_df['unusual_activity_score'] > 0.5).sum(),
                'volume_spike_rate': options_df['volume_spike'].mean(),
                'oi_spike_rate': options_df['oi_spike'].mean()
            }
        }
        
        # Implied volatility metrics (if available)
        iv_data = options_df.dropna(subset=['avg_call_iv', 'avg_put_iv'])
        if not iv_data.empty:
            summary['implied_volatility'] = {
                'avg_call_iv': iv_data['avg_call_iv'].mean(),
                'avg_put_iv': iv_data['avg_put_iv'].mean(),
                'avg_iv_skew': iv_data['iv_skew'].mean(),
                'events_with_iv_data': len(iv_data)
            }
        
        return summary
    
    def correlate_with_equity_results(self, options_df: pd.DataFrame, 
                                    equity_results_df: pd.DataFrame) -> Dict:
        """Correlate options metrics with equity abnormal returns."""
        if options_df.empty or equity_results_df.empty:
            return {'error': 'Insufficient data for correlation analysis'}
        
        # Merge datasets on ticker and event
        merged = pd.merge(
            options_df, 
            equity_results_df,
            left_on=['ticker', 'event_date'],
            right_on=['ticker', 'announcement_date'],
            how='inner'
        )
        
        if merged.empty:
            return {'error': 'No matching events between options and equity data'}
        
        # Calculate correlations
        correlations = {}
        
        # Get main abnormal return column (adjust name as needed)
        ar_col = 'abnormal_return_minus5_0' if 'abnormal_return_minus5_0' in merged.columns else None
        if not ar_col:
            # Try to find any abnormal return column
            ar_cols = [col for col in merged.columns if 'abnormal_return' in col.lower()]
            ar_col = ar_cols[0] if ar_cols else None
        
        if ar_col:
            correlations['options_vs_abnormal_returns'] = {
                'pcr_volume_correlation': merged['put_call_volume_ratio'].corr(merged[ar_col]),
                'call_volume_correlation': merged['total_call_volume'].corr(merged[ar_col]),
                'put_volume_correlation': merged['total_put_volume'].corr(merged[ar_col]),
                'unusual_activity_correlation': merged['unusual_activity_score'].corr(merged[ar_col]),
                'sample_size': len(merged)
            }
        
        return {
            'correlations': correlations,
            'merged_sample_size': len(merged),
            'events_analyzed': merged['event_name'].tolist()
        }


def main():
    """Run rapid options analysis."""
    print("="*80)
    print("RAPID OPTIONS FLOW ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = RapidOptionsAnalyzer()
    
    # Load events
    events_file = Path("data/processed/events_master.csv")
    if not events_file.exists():
        print("ERROR: Events master file not found")
        return
    
    events_df = pd.read_csv(events_file)
    events_df['announcement'] = pd.to_datetime(events_df['announcement']).dt.date
    
    print(f"Loaded {len(events_df)} events for options analysis")
    
    # Analyze options flow
    print("\n[COLLECTING] Options flow data...")
    options_results = analyzer.analyze_events_options_flow(events_df)
    
    if options_results.empty:
        print("No options data could be collected for recent events")
        return
    
    # Generate summary
    print("\n[ANALYZING] Generating options flow summary...")
    summary = analyzer.generate_options_summary(options_results)
    
    # Display results
    print("\n" + "="*60)
    print("OPTIONS FLOW SUMMARY")
    print("="*60)
    
    if 'sample_size' in summary:
        ss = summary['sample_size']
        print(f"\n[SAMPLE] Options Analysis Coverage:")
        print(f"  Events Analyzed: {ss['total_events_analyzed']}")
        print(f"  Volume Spikes: {ss['events_with_volume_spikes']}")
        print(f"  OI Spikes: {ss['events_with_oi_spikes']}")
        print(f"  Unique Tickers: {ss['unique_tickers']}")
    
    if 'volume_metrics' in summary:
        vm = summary['volume_metrics']
        print(f"\n[VOLUME] Average Options Volume:")
        print(f"  Call Volume: {vm['avg_call_volume']:,.0f}")
        print(f"  Put Volume: {vm['avg_put_volume']:,.0f}")
        print(f"  P/C Ratio: {vm['avg_pcr_volume']:.3f}")
    
    if 'unusual_activity' in summary:
        ua = summary['unusual_activity']
        print(f"\n[ACTIVITY] Unusual Activity Metrics:")
        print(f"  Avg Activity Score: {ua['avg_activity_score']:.3f}")
        print(f"  High Activity Events: {ua['high_activity_events']}")
        print(f"  Volume Spike Rate: {ua['volume_spike_rate']:.1%}")
    
    # Save results
    output_file = Path("results/options_flow_analysis.csv")
    options_results.to_csv(output_file, index=False)
    print(f"\n[SAVED] Results saved to {output_file}")
    
    # Try to correlate with equity results
    equity_file = Path("results/final_analysis_results.csv")
    if equity_file.exists():
        print("\n[CORRELATING] With equity abnormal returns...")
        equity_df = pd.read_csv(equity_file)
        correlation_results = analyzer.correlate_with_equity_results(options_results, equity_df)
        
        if 'correlations' in correlation_results:
            corr = correlation_results['correlations']['options_vs_abnormal_returns']
            print(f"  P/C Ratio vs AR: {corr['pcr_volume_correlation']:.3f}")
            print(f"  Unusual Activity vs AR: {corr['unusual_activity_correlation']:.3f}")
            print(f"  Sample Size: {corr['sample_size']}")
    
    print("\n[COMPLETE] Options flow analysis finished!")


if __name__ == "__main__":
    main()