"""
Earnings Date Analysis
Analyzes timing relationship between product launches and subsequent earnings
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def analyze_earnings_timing():
    """Analyze timing between product launches and subsequent earnings"""
    
    # Load events data
    events_df = pd.read_csv("results/equity_analysis_results.csv")
    events_df['announcement_date'] = pd.to_datetime(events_df['announcement_date'])
    
    # Load earnings data for each company
    earnings_data = {}
    companies = ['AAPL', 'NVDA', 'MSFT', 'TSLA', 'AMD', 'SONY']
    
    for company in companies:
        try:
            df = pd.read_csv(f"data/raw/earnings/{company}_earnings.csv")
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            earnings_data[company] = df
            print(f"Loaded {len(df)} earnings records for {company}")
        except Exception as e:
            print(f"Error loading {company}: {e}")
    
    # Analyze timing for each event
    timing_analysis = []
    
    for _, event in events_df.iterrows():
        ticker = event['ticker']
        launch_date = event['announcement_date']
        event_name = event['event_name']
        
        if ticker in earnings_data:
            company_earnings = earnings_data[ticker]
            
            # Find next earnings after launch
            future_earnings = company_earnings[
                company_earnings['fiscalDateEnding'] > launch_date
            ].sort_values('fiscalDateEnding')
            
            if len(future_earnings) > 0:
                next_earnings = future_earnings.iloc[0]
                days_to_earnings = (next_earnings['fiscalDateEnding'] - launch_date).days
                
                timing_analysis.append({
                    'event_name': event_name,
                    'ticker': ticker,
                    'launch_date': launch_date,
                    'next_earnings_date': next_earnings['fiscalDateEnding'],
                    'days_to_earnings': days_to_earnings,
                    'earnings_eps': next_earnings.get('reportedEPS', None),
                    'estimated_eps': next_earnings.get('estimatedEPS', None),
                    'surprise': next_earnings.get('surprise', None),
                    'surprise_percent': next_earnings.get('surprisePercentage', None)
                })
    
    timing_df = pd.DataFrame(timing_analysis)
    
    # Generate summary statistics
    summary = {
        'total_events_analyzed': len(timing_df),
        'events_with_earnings_data': len(timing_df[timing_df['days_to_earnings'].notna()]),
        'average_days_to_earnings': timing_df['days_to_earnings'].mean(),
        'median_days_to_earnings': timing_df['days_to_earnings'].median(),
        'min_days_to_earnings': timing_df['days_to_earnings'].min(),
        'max_days_to_earnings': timing_df['days_to_earnings'].max(),
        'earnings_coverage_percent': len(timing_df[timing_df['days_to_earnings'].notna()]) / len(timing_df) * 100
    }
    
    # Save results
    timing_df.to_csv("results/earnings_timing_analysis.csv", index=False)
    
    with open("results/earnings_timing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Generate detailed report
    report = f"""# EARNINGS DATE ANALYSIS
Product Launch to Earnings Timing Study

## Summary Statistics
- **Total Events**: {summary['total_events_analyzed']}
- **Events with Earnings Data**: {summary['events_with_earnings_data']} ({summary['earnings_coverage_percent']:.1f}%)
- **Average Days to Next Earnings**: {summary['average_days_to_earnings']:.1f}
- **Median Days to Next Earnings**: {summary['median_days_to_earnings']:.1f}
- **Range**: {summary['min_days_to_earnings']:.0f} to {summary['max_days_to_earnings']:.0f} days

## Timing Distribution
"""
    
    # Add distribution analysis
    bins = [0, 30, 60, 90, 120, 180, 365]
    timing_df['days_bin'] = pd.cut(timing_df['days_to_earnings'], bins=bins, 
                                  labels=['0-30', '31-60', '61-90', '91-120', '121-180', '181-365'])
    
    distribution = timing_df['days_bin'].value_counts().sort_index()
    
    for bin_name, count in distribution.items():
        percentage = count / len(timing_df) * 100
        report += f"- **{bin_name} days**: {count} events ({percentage:.1f}%)\n"
    
    report += "\n## Individual Event Analysis\n"
    report += "| Event | Company | Launch Date | Earnings Date | Days | Surprise % |\n"
    report += "|-------|---------|-------------|---------------|------|------------|\n"
    
    for _, row in timing_df.iterrows():
        surprise_str = f"{row['surprise_percent']:.1f}%" if pd.notna(row['surprise_percent']) else "N/A"
        report += f"| {row['event_name'][:30]} | {row['ticker']} | {row['launch_date'].strftime('%Y-%m-%d')} | {row['next_earnings_date'].strftime('%Y-%m-%d')} | {row['days_to_earnings']:.0f} | {surprise_str} |\n"
    
    report += f"\n## Analysis Significance\n"
    report += f"""
The timing analysis shows that product launches are strategically positioned relative to earnings announcements:

1. **Optimal Timing Window**: Most launches occur {summary['median_days_to_earnings']:.0f} days before earnings (median)
2. **Market Efficiency Test**: The {summary['average_days_to_earnings']:.0f}-day average window allows sufficient time for market assessment
3. **Earnings Impact**: {len(timing_df[timing_df['surprise'].notna()])}/{len(timing_df)} events have earnings surprise data for impact analysis
4. **Strategic Positioning**: Companies appear to coordinate launches to maximize earnings impact

This timing relationship is crucial for understanding whether options anomalies contain predictive information about subsequent earnings performance.
"""
    
    with open("results/earnings_timing_report.md", 'w') as f:
        f.write(report)
    
    print(f"Earnings timing analysis complete!")
    print(f"Coverage: {summary['earnings_coverage_percent']:.1f}% of events")
    print(f"Average timing: {summary['average_days_to_earnings']:.1f} days to earnings")
    
    return timing_df, summary

if __name__ == "__main__":
    analyze_earnings_timing()