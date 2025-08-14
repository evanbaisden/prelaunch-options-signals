"""
Earnings Analysis Framework
Analyzes earnings surprises in relation to options anomalies and stock price movements
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
import json

class EarningsAnalyzer:
    def __init__(self, events_file: str = "data/processed/events_with_equity_data.csv"):
        """
        Initialize earnings analyzer
        
        Args:
            events_file: Path to events data file
        """
        self.events_file = events_file
        self.events_df = None
        self.earnings_data = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load events data
        self.load_events_data()
    
    def load_events_data(self):
        """Load events data with equity returns"""
        try:
            self.events_df = pd.read_csv(self.events_file)
            self.events_df['event_date'] = pd.to_datetime(self.events_df['event_date'])
            self.logger.info(f"Loaded {len(self.events_df)} events from {self.events_file}")
        except Exception as e:
            self.logger.error(f"Error loading events data: {str(e)}")
            self.events_df = pd.DataFrame()
    
    def load_earnings_data(self, earnings_dir: str = "data/raw/earnings") -> Dict[str, pd.DataFrame]:
        """
        Load earnings data for all companies
        
        Args:
            earnings_dir: Directory containing earnings CSV files
            
        Returns:
            Dictionary mapping symbols to earnings DataFrames
        """
        earnings_data = {}
        
        if not os.path.exists(earnings_dir):
            self.logger.warning(f"Earnings directory not found: {earnings_dir}")
            return earnings_data
        
        for filename in os.listdir(earnings_dir):
            if filename.endswith('_earnings.csv'):
                symbol = filename.replace('_earnings.csv', '')
                filepath = os.path.join(earnings_dir, filename)
                
                try:
                    df = pd.read_csv(filepath)
                    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                    earnings_data[symbol] = df
                    self.logger.info(f"Loaded earnings data for {symbol}: {len(df)} records")
                except Exception as e:
                    self.logger.error(f"Error loading earnings for {symbol}: {str(e)}")
        
        self.earnings_data = earnings_data
        return earnings_data
    
    def find_nearest_earnings(self, symbol: str, event_date: datetime, max_days: int = 90) -> Optional[pd.Series]:
        """
        Find the nearest earnings announcement to an event date
        
        Args:
            symbol: Stock ticker symbol
            event_date: Event date to search around
            max_days: Maximum days to search before/after event
            
        Returns:
            Series with nearest earnings data or None if not found
        """
        if symbol not in self.earnings_data:
            return None
        
        earnings_df = self.earnings_data[symbol]
        
        # Calculate days difference from event date
        earnings_df['days_from_event'] = (earnings_df['fiscalDateEnding'] - event_date).dt.days
        
        # Filter to within max_days window
        nearby_earnings = earnings_df[earnings_df['days_from_event'].abs() <= max_days]
        
        if nearby_earnings.empty:
            return None
        
        # Find nearest earnings (prefer earnings after event, then before)
        future_earnings = nearby_earnings[nearby_earnings['days_from_event'] >= 0]
        if not future_earnings.empty:
            nearest = future_earnings.loc[future_earnings['days_from_event'].idxmin()]
        else:
            past_earnings = nearby_earnings[nearby_earnings['days_from_event'] < 0]
            nearest = past_earnings.loc[past_earnings['days_from_event'].idxmax()]
        
        return nearest
    
    def match_events_to_earnings(self, max_days: int = 90) -> pd.DataFrame:
        """
        Match product launch events to nearest earnings announcements
        
        Args:
            max_days: Maximum days to search for earnings around event
            
        Returns:
            DataFrame with events matched to earnings data
        """
        if self.events_df is None or self.events_df.empty:
            self.logger.error("No events data loaded")
            return pd.DataFrame()
        
        matched_data = []
        
        for _, event in self.events_df.iterrows():
            symbol = event['symbol']
            event_date = event['event_date']
            
            # Find nearest earnings
            nearest_earnings = self.find_nearest_earnings(symbol, event_date, max_days)
            
            if nearest_earnings is not None:
                # Combine event and earnings data
                matched_row = event.to_dict()
                
                # Add earnings data
                matched_row.update({
                    'earnings_date': nearest_earnings['fiscalDateEnding'],
                    'days_to_earnings': nearest_earnings['days_from_event'],
                    'reported_eps': nearest_earnings.get('reportedEPS', np.nan),
                    'estimated_eps': nearest_earnings.get('estimatedEPS', np.nan),
                    'earnings_surprise': nearest_earnings.get('surprise', np.nan),
                    'surprise_percentage': nearest_earnings.get('surprisePercentage', np.nan)
                })
                
                matched_data.append(matched_row)
                
                self.logger.info(f"Matched {symbol} event on {event_date.date()} to earnings on {nearest_earnings['fiscalDateEnding'].date()} ({nearest_earnings['days_from_event']} days)")
            else:
                self.logger.warning(f"No earnings found for {symbol} within {max_days} days of {event_date.date()}")
        
        matched_df = pd.DataFrame(matched_data)
        self.logger.info(f"Successfully matched {len(matched_df)} events to earnings announcements")
        
        return matched_df
    
    def load_options_analysis_data(self, options_file: str = "results/options_analysis_data.json") -> Dict:
        """
        Load options analysis results
        
        Args:
            options_file: Path to options analysis JSON file
            
        Returns:
            Dictionary with options analysis data
        """
        try:
            with open(options_file, 'r') as f:
                options_data = json.load(f)
            self.logger.info(f"Loaded options analysis data from {options_file}")
            return options_data
        except Exception as e:
            self.logger.error(f"Error loading options analysis data: {str(e)}")
            return {}
    
    def analyze_earnings_options_correlation(self, matched_df: pd.DataFrame, options_data: Dict) -> Dict:
        """
        Analyze correlation between earnings surprises and options anomalies
        
        Args:
            matched_df: DataFrame with events matched to earnings
            options_data: Options analysis data
            
        Returns:
            Dictionary with correlation analysis results
        """
        if matched_df.empty:
            return {}
        
        # Extract options metrics for matched events
        options_metrics = []
        
        for _, row in matched_df.iterrows():
            event_key = f"{row['symbol']}_{row['event_date'].strftime('%Y-%m-%d')}_{row['product']}"
            
            # Find matching options data
            event_options = None
            if 'detailed_results' in options_data:
                for event_data in options_data['detailed_results']:
                    if (event_data['symbol'] == row['symbol'] and 
                        event_data['event_date'] == row['event_date'].strftime('%Y-%m-%d')):
                        event_options = event_data
                        break
            
            if event_options:
                options_metrics.append({
                    'symbol': row['symbol'],
                    'event_date': row['event_date'],
                    'earnings_surprise': row['earnings_surprise'],
                    'surprise_percentage': row['surprise_percentage'],
                    'days_to_earnings': row['days_to_earnings'],
                    'total_volume': event_options.get('total_volume', 0),
                    'put_call_ratio': event_options.get('put_call_ratio', 0),
                    'avg_implied_volatility': event_options.get('avg_implied_volatility', 0),
                    'avg_bid_ask_spread': event_options.get('avg_bid_ask_spread', 0),
                    'volume_anomaly': event_options.get('volume_anomaly', False),
                    'pcr_anomaly': event_options.get('pcr_anomaly', False),
                    'composite_anomaly_score': event_options.get('composite_anomaly_score', 0)
                })
        
        options_df = pd.DataFrame(options_metrics)
        
        if options_df.empty:
            self.logger.warning("No matching options data found for earnings analysis")
            return {}
        
        # Calculate correlations
        correlations = {}
        
        # Numeric columns for correlation analysis
        numeric_cols = ['earnings_surprise', 'surprise_percentage', 'days_to_earnings', 
                       'total_volume', 'put_call_ratio', 'avg_implied_volatility', 
                       'avg_bid_ask_spread', 'composite_anomaly_score']
        
        # Filter to numeric columns that exist and have valid data
        available_cols = [col for col in numeric_cols if col in options_df.columns]
        numeric_df = options_df[available_cols].apply(pd.to_numeric, errors='coerce')
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Extract significant correlations with earnings metrics
        earnings_cols = ['earnings_surprise', 'surprise_percentage']
        options_cols = [col for col in available_cols if col not in earnings_cols + ['days_to_earnings']]
        
        significant_correlations = []
        
        for earnings_col in earnings_cols:
            if earnings_col in corr_matrix.columns:
                for options_col in options_cols:
                    if options_col in corr_matrix.columns:
                        correlation = corr_matrix.loc[earnings_col, options_col]
                        
                        if not np.isnan(correlation) and abs(correlation) > 0.3:
                            # Calculate p-value
                            valid_data = numeric_df[[earnings_col, options_col]].dropna()
                            if len(valid_data) > 2:
                                _, p_value = stats.pearsonr(valid_data[earnings_col], valid_data[options_col])
                                
                                significant_correlations.append({
                                    'earnings_metric': earnings_col,
                                    'options_metric': options_col,
                                    'correlation': correlation,
                                    'p_value': p_value,
                                    'sample_size': len(valid_data)
                                })
        
        correlations['significant_correlations'] = significant_correlations
        correlations['correlation_matrix'] = corr_matrix.to_dict()
        correlations['sample_size'] = len(options_df)
        
        self.logger.info(f"Found {len(significant_correlations)} significant earnings-options correlations")
        
        return correlations
    
    def analyze_earnings_returns_correlation(self, matched_df: pd.DataFrame) -> Dict:
        """
        Analyze correlation between earnings surprises and stock returns
        
        Args:
            matched_df: DataFrame with events matched to earnings
            
        Returns:
            Dictionary with correlation analysis results
        """
        if matched_df.empty:
            return {}
        
        # Return columns to analyze
        return_cols = [col for col in matched_df.columns if 'abnormal_return' in col]
        earnings_cols = ['earnings_surprise', 'surprise_percentage']
        
        correlations = {}
        significant_correlations = []
        
        for earnings_col in earnings_cols:
            if earnings_col in matched_df.columns:
                for return_col in return_cols:
                    if return_col in matched_df.columns:
                        # Get valid data
                        valid_data = matched_df[[earnings_col, return_col]].dropna()
                        
                        if len(valid_data) > 2:
                            correlation, p_value = stats.pearsonr(valid_data[earnings_col], valid_data[return_col])
                            
                            if abs(correlation) > 0.2:  # Lower threshold for earnings-returns
                                significant_correlations.append({
                                    'earnings_metric': earnings_col,
                                    'return_metric': return_col,
                                    'correlation': correlation,
                                    'p_value': p_value,
                                    'sample_size': len(valid_data)
                                })
        
        correlations['significant_correlations'] = significant_correlations
        correlations['sample_size'] = len(matched_df)
        
        self.logger.info(f"Found {len(significant_correlations)} significant earnings-returns correlations")
        
        return correlations
    
    def generate_earnings_report(self, output_file: str = "results/earnings_analysis_report.md"):
        """
        Generate comprehensive earnings analysis report
        
        Args:
            output_file: Path to output report file
        """
        # Load earnings data
        self.load_earnings_data()
        
        # Match events to earnings
        matched_df = self.match_events_to_earnings()
        
        # Load options analysis data
        options_data = self.load_options_analysis_data()
        
        # Analyze correlations
        earnings_options_corr = self.analyze_earnings_options_correlation(matched_df, options_data)
        earnings_returns_corr = self.analyze_earnings_returns_correlation(matched_df)
        
        # Generate report
        report = self._create_earnings_report(matched_df, earnings_options_corr, earnings_returns_corr)
        
        # Save report
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Earnings analysis report saved to {output_file}")
        
        return {
            'matched_events': matched_df,
            'earnings_options_correlations': earnings_options_corr,
            'earnings_returns_correlations': earnings_returns_corr
        }
    
    def _create_earnings_report(self, matched_df: pd.DataFrame, earnings_options_corr: Dict, earnings_returns_corr: Dict) -> str:
        """
        Create formatted earnings analysis report
        
        Args:
            matched_df: DataFrame with matched events and earnings
            earnings_options_corr: Earnings-options correlation results
            earnings_returns_corr: Earnings-returns correlation results
            
        Returns:
            Formatted report string
        """
        report = f"""# Earnings Analysis Report
## Pre-Launch Options Signals: Earnings Integration Study

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Events Matched to Earnings**: {len(matched_df)}

---

## Executive Summary

This analysis examines the relationship between earnings surprises and pre-launch options anomalies, providing insights into whether options markets contain information about upcoming earnings outcomes around product launch events.

### Key Findings

- **Events with Earnings Data**: {len(matched_df)} out of {len(self.events_df) if self.events_df is not None else 0} total events
- **Earnings Coverage**: {len(matched_df) / len(self.events_df) * 100:.1f}% of events have nearby earnings data
- **Significant Earnings-Options Correlations**: {len(earnings_options_corr.get('significant_correlations', []))}
- **Significant Earnings-Returns Correlations**: {len(earnings_returns_corr.get('significant_correlations', []))}

---

## I. Earnings Data Summary

"""
        
        if not matched_df.empty:
            # Earnings statistics
            earnings_stats = matched_df[['earnings_surprise', 'surprise_percentage', 'days_to_earnings']].describe()
            
            report += f"""
### Earnings Surprise Statistics
```
{earnings_stats.to_string()}
```

### Distribution by Company
"""
            company_counts = matched_df['symbol'].value_counts()
            for symbol, count in company_counts.items():
                report += f"- **{symbol}**: {count} events with earnings data\n"
            
            report += f"""

### Time to Earnings Distribution
- **Same Quarter**: {len(matched_df[matched_df['days_to_earnings'].abs() <= 30])} events
- **Next Quarter**: {len(matched_df[(matched_df['days_to_earnings'].abs() > 30) & (matched_df['days_to_earnings'].abs() <= 90)])} events
"""
        
        report += """

---

## II. Earnings-Options Correlations

This section analyzes whether options anomalies predict earnings surprises.

"""
        
        if earnings_options_corr.get('significant_correlations'):
            report += "### Significant Correlations (|r| > 0.3, p < 0.05)\n\n"
            
            for corr in earnings_options_corr['significant_correlations']:
                report += f"- **{corr['options_metric']}** → **{corr['earnings_metric']}**: r={corr['correlation']:.3f} (p={corr['p_value']:.3f}, N={corr['sample_size']})\n"
        else:
            report += "### No Significant Correlations Found\n\nNo statistically significant correlations (|r| > 0.3, p < 0.05) were found between options anomalies and earnings surprises.\n"
        
        report += """

---

## III. Earnings-Returns Correlations

This section analyzes the relationship between earnings surprises and stock price movements.

"""
        
        if earnings_returns_corr.get('significant_correlations'):
            report += "### Significant Correlations (|r| > 0.2, p < 0.05)\n\n"
            
            for corr in earnings_returns_corr['significant_correlations']:
                report += f"- **{corr['earnings_metric']}** → **{corr['return_metric']}**: r={corr['correlation']:.3f} (p={corr['p_value']:.3f}, N={corr['sample_size']})\n"
        else:
            report += "### No Significant Correlations Found\n\nNo statistically significant correlations (|r| > 0.2, p < 0.05) were found between earnings surprises and stock returns.\n"
        
        report += f"""

---

## IV. Methodology and Limitations

### Data Matching Methodology
1. **Earnings Window**: Events matched to earnings within 90 days
2. **Preference Order**: Future earnings preferred over past earnings
3. **Quality Control**: Only events with complete earnings data included

### Limitations
1. **Sample Size**: Limited to {len(matched_df)} events with earnings data
2. **Timing Effects**: Product launches may not align with earnings cycles
3. **Confounding Factors**: Other market events may affect correlations
4. **Data Availability**: Historical earnings estimates may be incomplete

---

## V. Future Research Directions

1. **Extended Time Window**: Analyze earnings impact over longer periods
2. **Sector Analysis**: Compare results across technology subsectors
3. **Guidance Integration**: Include management guidance in analysis
4. **Event Classification**: Separate by product launch magnitude and success

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Framework**: Earnings Integration v1.0
**Contact**: Academic Research Project

"""
        
        return report

def main():
    """
    Main function to run earnings analysis
    """
    analyzer = EarningsAnalyzer()
    
    # Generate comprehensive earnings analysis report
    results = analyzer.generate_earnings_report()
    
    print("Earnings Analysis Summary:")
    print("-" * 40)
    print(f"Events matched to earnings: {len(results['matched_events'])}")
    print(f"Earnings-options correlations: {len(results['earnings_options_correlations'].get('significant_correlations', []))}")
    print(f"Earnings-returns correlations: {len(results['earnings_returns_correlations'].get('significant_correlations', []))}")

if __name__ == "__main__":
    main()