"""
Comprehensive Options Analysis
Historical options flow analysis around product launch events.
Integrates with the equity event study analysis.
"""
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
from scipy import stats
import warnings

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from phase2.options_data import AlphaVantageOptionsProvider

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveOptionsAnalyzer:
    """Comprehensive options analysis for product launch events."""
    
    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.cache_dir = Path("data/raw/options")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize options data provider
        try:
            self.options_provider = AlphaVantageOptionsProvider()
            logger.info("Alpha Vantage options provider initialized")
        except Exception as e:
            logger.error(f"Failed to initialize options provider: {e}")
            self.options_provider = None
    
    def load_events(self) -> pd.DataFrame:
        """Load events dataset."""
        events_file = Path("data/processed/events_master.csv")
        if not events_file.exists():
            raise FileNotFoundError(f"Events file not found: {events_file}")
        
        df = pd.read_csv(events_file)
        df['announcement'] = pd.to_datetime(df['announcement']).dt.date
        df['release'] = pd.to_datetime(df['release']).dt.date
        
        logger.info(f"Loaded {len(df)} events from master dataset")
        return df
    
    def collect_options_data(self, events_df: pd.DataFrame, sample_events: int = 5) -> pd.DataFrame:
        """
        Collect historical options data for a sample of events.
        Limited sample due to API rate limits.
        """
        if not self.options_provider:
            logger.error("Options provider not available")
            return pd.DataFrame()
        
        logger.info(f"Collecting options data for {sample_events} sample events...")
        
        # Select recent events for better data availability
        recent_events = events_df[events_df['announcement'] >= date(2024, 1, 1)].head(sample_events)
        if len(recent_events) < sample_events:
            # Fall back to most recent available
            recent_events = events_df.sort_values('announcement', ascending=False).head(sample_events)
        
        all_options_data = []
        
        for i, (_, event) in enumerate(recent_events.iterrows()):
            logger.info(f"Collecting options data for {event['name']} ({i+1}/{len(recent_events)})")
            
            try:
                # Get options data for event date
                options_data = self.options_provider.get_historical_options_data(
                    event['ticker'], 
                    event['announcement']
                )
                
                if not options_data.empty:
                    # Add event metadata
                    options_data['event_name'] = event['name']
                    options_data['event_company'] = event['company']
                    options_data['event_date'] = event['announcement']
                    options_data['event_category'] = event['category']
                    
                    all_options_data.append(options_data)
                    logger.info(f"Collected {len(options_data)} options contracts for {event['name']}")
                else:
                    logger.warning(f"No options data found for {event['name']}")
                
                # Rate limiting for Alpha Vantage
                import time
                time.sleep(12)  # Alpha Vantage free tier rate limit
                
            except Exception as e:
                logger.error(f"Error collecting options data for {event['name']}: {e}")
                continue
        
        if all_options_data:
            combined_df = pd.concat(all_options_data, ignore_index=True)
            logger.info(f"Total options contracts collected: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No options data collected")
            return pd.DataFrame()
    
    def analyze_options_flow(self, options_df: pd.DataFrame) -> Dict:
        """Analyze options flow patterns around events."""
        if options_df.empty:
            return {'error': 'No options data available'}
        
        logger.info("Analyzing options flow patterns...")
        
        # Calculate event-level metrics
        event_metrics = []
        
        for event_name in options_df['event_name'].unique():
            event_data = options_df[options_df['event_name'] == event_name]
            
            # Separate calls and puts
            calls = event_data[event_data['option_type'] == 'call']
            puts = event_data[event_data['option_type'] == 'put']
            
            # Volume metrics
            total_call_volume = calls['volume'].sum()
            total_put_volume = puts['volume'].sum()
            pcr_volume = total_put_volume / max(total_call_volume, 1)
            
            # Open Interest metrics
            total_call_oi = calls['open_interest'].sum()
            total_put_oi = puts['open_interest'].sum()
            pcr_oi = total_put_oi / max(total_call_oi, 1)
            
            # Implied Volatility metrics
            avg_call_iv = calls['implied_volatility'].mean()
            avg_put_iv = puts['implied_volatility'].mean()
            iv_skew = avg_put_iv - avg_call_iv
            
            # Volume-weighted metrics
            calls_vwap_iv = (calls['implied_volatility'] * calls['volume']).sum() / max(calls['volume'].sum(), 1)
            puts_vwap_iv = (puts['implied_volatility'] * puts['volume']).sum() / max(puts['volume'].sum(), 1)
            
            # Unusual activity detection
            total_volume = total_call_volume + total_put_volume
            unusual_volume = total_volume > options_df['volume'].quantile(0.8)
            
            event_metrics.append({
                'event_name': event_name,
                'company': event_data['event_company'].iloc[0],
                'ticker': event_data['underlying_ticker'].iloc[0],
                'event_date': event_data['event_date'].iloc[0],
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'put_call_ratio_volume': pcr_volume,
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'put_call_ratio_oi': pcr_oi,
                'avg_call_iv': avg_call_iv,
                'avg_put_iv': avg_put_iv,
                'iv_skew': iv_skew,
                'vwap_call_iv': calls_vwap_iv,
                'vwap_put_iv': puts_vwap_iv,
                'total_volume': total_volume,
                'unusual_volume': unusual_volume,
                'num_call_contracts': len(calls),
                'num_put_contracts': len(puts)
            })
        
        event_metrics_df = pd.DataFrame(event_metrics)
        
        # Generate aggregate statistics
        aggregate_stats = {
            'sample_size': len(event_metrics_df),
            'volume_statistics': {
                'avg_call_volume': event_metrics_df['total_call_volume'].mean(),
                'avg_put_volume': event_metrics_df['total_put_volume'].mean(),
                'avg_pcr_volume': event_metrics_df['put_call_ratio_volume'].mean(),
                'median_pcr_volume': event_metrics_df['put_call_ratio_volume'].median(),
                'std_pcr_volume': event_metrics_df['put_call_ratio_volume'].std()
            },
            'open_interest_statistics': {
                'avg_call_oi': event_metrics_df['total_call_oi'].mean(),
                'avg_put_oi': event_metrics_df['total_put_oi'].mean(),
                'avg_pcr_oi': event_metrics_df['put_call_ratio_oi'].mean(),
                'median_pcr_oi': event_metrics_df['put_call_ratio_oi'].median(),
                'std_pcr_oi': event_metrics_df['put_call_ratio_oi'].std()
            },
            'implied_volatility_statistics': {
                'avg_call_iv': event_metrics_df['avg_call_iv'].mean(),
                'avg_put_iv': event_metrics_df['avg_put_iv'].mean(),
                'avg_iv_skew': event_metrics_df['iv_skew'].mean(),
                'std_iv_skew': event_metrics_df['iv_skew'].std()
            },
            'unusual_activity': {
                'events_with_unusual_volume': event_metrics_df['unusual_volume'].sum(),
                'unusual_volume_rate': event_metrics_df['unusual_volume'].mean(),
                'high_pcr_events': (event_metrics_df['put_call_ratio_volume'] > 1.5).sum(),
                'low_pcr_events': (event_metrics_df['put_call_ratio_volume'] < 0.5).sum()
            }
        }
        
        return {
            'event_metrics': event_metrics_df,
            'aggregate_statistics': aggregate_stats
        }
    
    def correlate_with_equity_results(self, options_results: Dict) -> Dict:
        """Correlate options metrics with equity abnormal returns."""
        if 'event_metrics' not in options_results:
            return {'error': 'No options metrics available'}
        
        options_df = options_results['event_metrics']
        
        # Load equity results
        equity_file = Path("results/final_analysis_results.csv")
        if not equity_file.exists():
            return {'error': 'Equity results not found'}
        
        equity_df = pd.read_csv(equity_file)
        
        # Merge on ticker and event date
        merged = pd.merge(
            options_df,
            equity_df,
            left_on=['ticker', 'event_date'],
            right_on=['ticker', 'announcement_date'],
            how='inner'
        )
        
        if merged.empty:
            return {'error': 'No matching events for correlation analysis'}
        
        # Find abnormal return column
        ar_col = None
        for col in merged.columns:
            if 'abnormal_return' in col.lower() and 'minus5_0' in col:
                ar_col = col
                break
        
        if not ar_col:
            return {'error': 'Abnormal return column not found'}
        
        # Calculate correlations
        correlations = {
            'sample_size': len(merged),
            'correlations': {
                'pcr_volume_vs_abnormal_returns': merged['put_call_ratio_volume'].corr(merged[ar_col]),
                'total_volume_vs_abnormal_returns': merged['total_volume'].corr(merged[ar_col]),
                'iv_skew_vs_abnormal_returns': merged['iv_skew'].corr(merged[ar_col]),
                'call_volume_vs_abnormal_returns': merged['total_call_volume'].corr(merged[ar_col]),
                'put_volume_vs_abnormal_returns': merged['total_put_volume'].corr(merged[ar_col])
            },
            'statistical_tests': {}
        }
        
        # Perform statistical significance tests
        for metric in ['put_call_ratio_volume', 'iv_skew', 'total_volume']:
            if metric in merged.columns:
                corr, p_value = stats.pearsonr(merged[metric], merged[ar_col])
                correlations['statistical_tests'][f'{metric}_test'] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return correlations
    
    def generate_options_report(self, options_analysis: Dict, correlation_analysis: Dict) -> str:
        """Generate options analysis report."""
        report_lines = [
            "# Options Flow Analysis Report",
            "",
            "## Summary",
            "",
            f"This analysis examines options market activity around {options_analysis['aggregate_statistics']['sample_size']} technology product launch events."
        ]
        
        # Volume Statistics
        vol_stats = options_analysis['aggregate_statistics']['volume_statistics']
        report_lines.extend([
            "",
            "## Volume Statistics",
            "",
            f"- **Average Call Volume**: {vol_stats['avg_call_volume']:,.0f}",
            f"- **Average Put Volume**: {vol_stats['avg_put_volume']:,.0f}", 
            f"- **Average P/C Ratio**: {vol_stats['avg_pcr_volume']:.3f}",
            f"- **Median P/C Ratio**: {vol_stats['median_pcr_volume']:.3f}",
            f"- **P/C Ratio Std Dev**: {vol_stats['std_pcr_volume']:.3f}"
        ])
        
        # Implied Volatility Statistics
        iv_stats = options_analysis['aggregate_statistics']['implied_volatility_statistics']
        report_lines.extend([
            "",
            "## Implied Volatility Analysis",
            "",
            f"- **Average Call IV**: {iv_stats['avg_call_iv']:.1%}",
            f"- **Average Put IV**: {iv_stats['avg_put_iv']:.1%}",
            f"- **Average IV Skew**: {iv_stats['avg_iv_skew']:.1%}",
            f"- **IV Skew Std Dev**: {iv_stats['std_iv_skew']:.1%}"
        ])
        
        # Unusual Activity
        unusual_stats = options_analysis['aggregate_statistics']['unusual_activity']
        report_lines.extend([
            "",
            "## Unusual Activity Detection",
            "",
            f"- **Events with Unusual Volume**: {unusual_stats['events_with_unusual_volume']}/{options_analysis['aggregate_statistics']['sample_size']}",
            f"- **Unusual Volume Rate**: {unusual_stats['unusual_volume_rate']:.1%}",
            f"- **High P/C Ratio Events (>1.5)**: {unusual_stats['high_pcr_events']}",
            f"- **Low P/C Ratio Events (<0.5)**: {unusual_stats['low_pcr_events']}"
        ])
        
        # Correlation Analysis
        if 'correlations' in correlation_analysis:
            corr = correlation_analysis['correlations']
            report_lines.extend([
                "",
                "## Correlation with Equity Abnormal Returns",
                "",
                f"**Sample Size**: {correlation_analysis['sample_size']} matched events",
                "",
                f"- **P/C Ratio vs Abnormal Returns**: {corr['pcr_volume_vs_abnormal_returns']:.3f}",
                f"- **Total Volume vs Abnormal Returns**: {corr['total_volume_vs_abnormal_returns']:.3f}",
                f"- **IV Skew vs Abnormal Returns**: {corr['iv_skew_vs_abnormal_returns']:.3f}",
                f"- **Call Volume vs Abnormal Returns**: {corr['call_volume_vs_abnormal_returns']:.3f}",
                f"- **Put Volume vs Abnormal Returns**: {corr['put_volume_vs_abnormal_returns']:.3f}"
            ])
            
            # Statistical significance
            if 'statistical_tests' in correlation_analysis:
                report_lines.extend([
                    "",
                    "### Statistical Significance Tests",
                    ""
                ])
                for test_name, test_result in correlation_analysis['statistical_tests'].items():
                    sig_marker = "***" if test_result['significant'] else ""
                    report_lines.append(f"- **{test_name}**: r={test_result['correlation']:.3f}, p={test_result['p_value']:.3f} {sig_marker}")
        
        # Methodology Note
        report_lines.extend([
            "",
            "## Methodology Notes",
            "",
            "- Options data sourced from Alpha Vantage Historical Options API",
            "- Analysis focuses on total daily volume and open interest",
            "- Put/Call ratios calculated as Put Volume / Call Volume",
            "- IV Skew calculated as Put IV - Call IV",
            "- Unusual activity defined as top 20% of volume distribution",
            "- Correlations calculated using Pearson correlation coefficient",
            "",
            "## Limitations",
            "",
            "- Limited sample size due to API rate constraints",
            "- Analysis based on end-of-day options data",
            "- Historical options data availability varies by date",
            "- No intraday flow analysis or order flow direction"
        ])
        
        return "\n".join(report_lines)
    
    def run_comprehensive_options_analysis(self) -> Dict:
        """Run complete options analysis pipeline."""
        logger.info("Starting comprehensive options analysis...")
        
        # Load events
        events_df = self.load_events()
        
        # Collect sample options data
        options_data = self.collect_options_data(events_df, sample_events=3)  # Small sample due to rate limits
        
        if options_data.empty:
            return {'error': 'No options data collected'}
        
        # Analyze options flow
        options_analysis = self.analyze_options_flow(options_data)
        
        # Correlate with equity results
        correlation_analysis = self.correlate_with_equity_results(options_analysis)
        
        # Generate report
        report = self.generate_options_report(options_analysis, correlation_analysis)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save options data
        options_file = self.results_dir / f"options_data_{timestamp}.csv"
        options_data.to_csv(options_file, index=False)
        
        # Save event metrics
        if 'event_metrics' in options_analysis:
            metrics_file = self.results_dir / f"options_event_metrics_{timestamp}.csv"
            options_analysis['event_metrics'].to_csv(metrics_file, index=False)
        
        # Save report
        report_file = self.results_dir / f"options_analysis_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save full results
        results = {
            'options_analysis': options_analysis,
            'correlation_analysis': correlation_analysis,
            'report': report,
            'files_saved': {
                'options_data': str(options_file),
                'event_metrics': str(metrics_file) if 'event_metrics' in options_analysis else None,
                'report': str(report_file)
            }
        }
        
        results_file = self.results_dir / f"options_analysis_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Options analysis complete. Results saved to {results_file}")
        return results


def main():
    """Run comprehensive options analysis."""
    print("="*80)
    print("COMPREHENSIVE OPTIONS FLOW ANALYSIS")
    print("Integration with Equity Event Study")
    print("="*80)
    
    analyzer = ComprehensiveOptionsAnalyzer()
    
    try:
        results = analyzer.run_comprehensive_options_analysis()
        
        if 'error' in results:
            print(f"[ERROR] {results['error']}")
            return
        
        # Display summary results
        if 'options_analysis' in results:
            stats = results['options_analysis']['aggregate_statistics']
            
            print(f"\n[SAMPLE] Options Analysis:")
            print(f"  Events Analyzed: {stats['sample_size']}")
            
            vol_stats = stats['volume_statistics']
            print(f"\n[VOLUME] Average Metrics:")
            print(f"  Call Volume: {vol_stats['avg_call_volume']:,.0f}")
            print(f"  Put Volume: {vol_stats['avg_put_volume']:,.0f}")
            print(f"  P/C Ratio: {vol_stats['avg_pcr_volume']:.3f}")
            
            iv_stats = stats['implied_volatility_statistics']
            print(f"\n[IV] Implied Volatility:")
            print(f"  Call IV: {iv_stats['avg_call_iv']:.1%}")
            print(f"  Put IV: {iv_stats['avg_put_iv']:.1%}")
            print(f"  IV Skew: {iv_stats['avg_iv_skew']:.1%}")
        
        if 'correlation_analysis' in results and 'correlations' in results['correlation_analysis']:
            corr = results['correlation_analysis']['correlations']
            print(f"\n[CORRELATION] With Abnormal Returns:")
            print(f"  P/C Ratio: {corr['pcr_volume_vs_abnormal_returns']:.3f}")
            print(f"  IV Skew: {corr['iv_skew_vs_abnormal_returns']:.3f}")
            print(f"  Total Volume: {corr['total_volume_vs_abnormal_returns']:.3f}")
            print(f"  Sample Size: {results['correlation_analysis']['sample_size']}")
        
        print(f"\n[COMPLETE] Options analysis finished!")
        print(f"[SAVED] Results saved to results/ directory")
        
    except Exception as e:
        logger.error(f"Options analysis failed: {e}")
        print(f"[ERROR] Analysis failed: {e}")


if __name__ == "__main__":
    main()