"""
Complete the earnings collection for AMD and SONY
"""

import sys
sys.path.append('src')

from src.earnings_data_collector import EarningsDataCollector

def main():
    # API key
    api_key = 'UH69QQYP504YPZVA'
    
    print('Completing Earnings Collection - AMD and SONY')
    print('=' * 50)
    
    # Initialize collector
    collector = EarningsDataCollector(api_key)
    
    # Only collect missing companies
    remaining_companies = ['AMD', 'SONY']
    
    print(f'Collecting earnings for: {remaining_companies}')
    print(f'API calls needed: {len(remaining_companies) * 2}')
    
    # Collect earnings data
    results = collector.collect_all_earnings(remaining_companies)
    
    # Print summary
    print('\nRemaining Earnings Collection Summary:')
    print('-' * 40)
    for symbol, (earnings_df, estimates_df) in results.items():
        earnings_count = len(earnings_df) if not earnings_df.empty else 0
        estimates_count = len(estimates_df) if not estimates_df.empty else 0
        print(f'{symbol}: {earnings_count} earnings records, {estimates_count} estimate records')
    
    print(f'\nAPI calls used: {collector.call_count}/25')
    print('Earnings collection complete!')

if __name__ == "__main__":
    main()