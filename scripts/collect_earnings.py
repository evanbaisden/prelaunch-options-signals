"""
Collect earnings data for all companies in our dataset
"""

import sys
sys.path.append('src')

from src.earnings_data_collector import EarningsDataCollector

def main():
    # API key from .env file
    api_key = 'UH69QQYP504YPZVA'
    
    print('AlphaVantage Earnings Data Collection')
    print('=' * 40)
    
    # Initialize collector
    collector = EarningsDataCollector(api_key)
    
    # Companies from our dataset
    companies = ['AAPL', 'NVDA', 'MSFT', 'TSLA', 'AMD', 'SONY']
    
    print(f'Companies to collect: {companies}')
    print(f'Total API calls needed: {len(companies) * 2} (EARNINGS + EARNINGS_ESTIMATES per company)')
    print(f'API calls available: {25 - collector.call_count}/25')
    print('')
    
    # Collect earnings data for all companies
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
    
    if successful == len(companies):
        print('\nAll earnings data collected successfully!')
    else:
        print(f'\nPartial collection - {len(companies) - successful} companies failed')

if __name__ == "__main__":
    main()