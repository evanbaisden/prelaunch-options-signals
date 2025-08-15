"""
Collect earnings data relevant to our product launch events
Only collects earnings for quarters that are close to product announcements
"""

import sys
sys.path.append('src')

import pandas as pd
from datetime import datetime, timedelta
from src.earnings_data_collector import EarningsDataCollector

def find_relevant_earnings_quarters(events_df):
    """
    Find which earnings quarters are relevant for our product launch events
    """
    relevant_quarters = {}
    
    print("Mapping product launches to relevant earnings quarters:")
    print("-" * 60)
    
    for _, event in events_df.iterrows():
        symbol = event['ticker']
        event_date = pd.to_datetime(event['announcement']).date()
        event_name = event['name']
        
        # Find the earnings quarter this event falls into
        # Most companies report quarterly earnings
        year = event_date.year
        month = event_date.month
        
        # Determine fiscal quarter (approximate)
        if month <= 3:
            quarter = "Q1"
            # Also get previous quarter (Q4 of previous year)
            prev_year = year - 1
            prev_quarter = "Q4"
        elif month <= 6:
            quarter = "Q2"
            prev_quarter = "Q1"
            prev_year = year
        elif month <= 9:
            quarter = "Q3"
            prev_quarter = "Q2"
            prev_year = year
        else:
            quarter = "Q4"
            prev_quarter = "Q3"
            prev_year = year
        
        # Store relevant quarters for this company
        if symbol not in relevant_quarters:
            relevant_quarters[symbol] = set()
        
        # Add only the next earnings quarter (where product impact would show up)
        # This should be the quarter AFTER the product launch
        if month <= 3:
            next_quarter = "Q2"
            next_year = year
        elif month <= 6:
            next_quarter = "Q3" 
            next_year = year
        elif month <= 9:
            next_quarter = "Q4"
            next_year = year
        else:
            next_quarter = "Q1"
            next_year = year + 1
            
        relevant_quarters[symbol].add(f"{next_year}-{next_quarter}")
        
        print(f"{symbol} {event_date} ({event_name})")
        print(f"  -> Next earnings quarter: {next_year}-{next_quarter}")
    
    return relevant_quarters

def main():
    # API key
    api_key = 'UH69QQYP504YPZVA'
    
    print('Targeted Earnings Data Collection')
    print('Collecting earnings relevant to product launch events')
    print('=' * 60)
    
    # Load our events
    events_df = pd.read_csv('data/processed/events_master.csv')
    print(f"Loaded {len(events_df)} product launch events")
    
    # Find relevant earnings quarters
    relevant_quarters = find_relevant_earnings_quarters(events_df)
    
    print(f"\nRelevant earnings quarters by company:")
    total_quarters = 0
    for symbol, quarters in relevant_quarters.items():
        print(f"{symbol}: {sorted(quarters)} ({len(quarters)} quarters)")
        total_quarters += len(quarters)
    
    print(f"\nTotal unique quarters to analyze: {total_quarters}")
    
    # Initialize earnings collector
    collector = EarningsDataCollector(api_key)
    companies = list(relevant_quarters.keys())
    
    print(f"\nAPI calls needed: {len(companies) * 2} (EARNINGS + EARNINGS_ESTIMATES per company)")
    print(f"API calls available: {25 - collector.call_count}/25")
    
    # Ask user if they want to proceed
    print(f"\nThis will collect earnings data for {len(companies)} companies:")
    print(f"{companies}")
    print("The data will then be filtered to relevant quarters around product launches.")
    
    print("\nProceeding with collection...")
    
    if True:
        print("\nStarting targeted earnings collection...")
        
        # Collect earnings data
        results = collector.collect_all_earnings(companies)
        
        # Print summary
        print('\nEarnings Data Collection Summary:')
        print('-' * 50)
        successful = 0
        for symbol, (earnings_df, estimates_df) in results.items():
            earnings_count = len(earnings_df) if not earnings_df.empty else 0
            estimates_count = len(estimates_df) if not estimates_df.empty else 0
            
            if earnings_count > 0 or estimates_count > 0:
                print(f'{symbol}: {earnings_count} earnings records, {estimates_count} estimate records')
                successful += 1
            else:
                print(f'{symbol}: No data collected')
        
        print(f'\nTotal successful: {successful}/{len(companies)} companies')
        print(f'API calls used: {collector.call_count}/25')
        
        # Now filter to relevant quarters and create event-earnings mapping
        if successful > 0:
            print('\nNext step: Run earnings analysis to map events to relevant earnings')
            print('This will match each product launch to its nearest earnings announcement')
    else:
        print("Collection cancelled.")

if __name__ == "__main__":
    main()