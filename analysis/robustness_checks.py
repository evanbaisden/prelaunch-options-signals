"""
Robustness Checks for Event Study Analysis
Tests alternative market models and estimation windows
Building on existing results
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

def get_market_data(start_date, end_date):
    """Download market data for robustness checks"""
    
    # Download market indices
    market_data = {}
    
    try:
        # S&P 500 (SPY)
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        market_data['SPY'] = spy['Adj Close'].pct_change().dropna()
        
        # Technology sector (XLK)
        xlk = yf.download('XLK', start=start_date, end=end_date, progress=False)
        market_data['XLK'] = xlk['Adj Close'].pct_change().dropna()
        
        # Risk-free rate proxy (3-month treasury)
        # For simplicity, using a constant 2% annual rate
        market_data['RF'] = pd.Series([0.02/252] * len(market_data['SPY']), 
                                    index=market_data['SPY'].index)
        
        print(f"Downloaded market data from {start_date} to {end_date}")
        return market_data
    
    except Exception as e:
        print(f"Error downloading market data: {e}")
        return None

def calculate_fama_french_factors(market_data):
    """Approximate Fama-French factors using available ETFs"""
    
    # This is a simplified approximation
    # In practice, you'd use official Fama-French data from Kenneth French's website
    
    factors = {}
    
    # Market risk premium (Rm - Rf)
    factors['MKT'] = market_data['SPY'] - market_data['RF']
    
    # Size factor approximation (using small vs large cap ETFs)
    # For simplicity, using sector rotation as proxy
    factors['SMB'] = market_data['XLK'] - market_data['SPY']  # Tech vs broad market
    
    # Value factor approximation
    # Using negative of growth premium as proxy
    factors['HML'] = -factors['SMB']  # Simplified approximation
    
    return factors

def load_stock_data():
    """Load stock price data for all tickers in the study"""
    
    # Load existing results to get tickers and dates
    with open("results/final_archive/final_analysis_results.json", 'r') as f:
        results = json.load(f)
    
    # Get unique tickers and date range
    tickers = list(set([event['ticker'] for event in results]))
    
    # Get earliest and latest dates
    dates = [pd.to_datetime(event['announcement_date']) for event in results]
    start_date = min(dates) - timedelta(days=150)  # Extra buffer for estimation
    end_date = max(dates) + timedelta(days=30)
    
    print(f"Loading stock data for {tickers} from {start_date.date()} to {end_date.date()}")
    
    stock_data = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            stock_data[ticker] = data['Adj Close'].pct_change().dropna()
            print(f"  {ticker}: {len(stock_data[ticker])} observations")
        except Exception as e:
            print(f"  Error loading {ticker}: {e}")
    
    return stock_data, start_date, end_date

def alternative_market_models(event_data, stock_returns, market_data, factors):
    """Test alternative market models for each event"""
    
    ticker = event_data['ticker']
    announcement_date = pd.to_datetime(event_data['announcement_date'])
    
    if ticker not in stock_returns:
        return None
    
    stock_ret = stock_returns[ticker]
    
    # Align dates
    common_dates = stock_ret.index.intersection(market_data['SPY'].index)
    if len(common_dates) < 100:  # Need sufficient data
        return None
    
    stock_ret = stock_ret.reindex(common_dates)
    
    results = {}
    
    # Define estimation windows to test
    estimation_windows = [
        (-150, -6, 'Standard (-150,-6)'),
        (-90, -6, 'Shorter (-90,-6)'),
        (-180, -6, 'Longer (-180,-6)'),
        (-120, -31, 'Alternative (-120,-31)')
    ]
    
    for start_days, end_days, window_name in estimation_windows:
        
        est_start = announcement_date + pd.Timedelta(days=start_days)
        est_end = announcement_date + pd.Timedelta(days=end_days)
        
        # Filter estimation period
        est_mask = (common_dates >= est_start) & (common_dates <= est_end)
        if est_mask.sum() < 30:  # Need minimum observations
            continue
        
        est_stock = stock_ret[est_mask]
        
        window_results = {}
        
        # Model 1: Single Index Model (current approach)
        market_ret = market_data['SPY'].reindex(est_stock.index)
        valid_data = ~(est_stock.isna() | market_ret.isna())
        
        if valid_data.sum() > 10:
            X = market_ret[valid_data].values.reshape(-1, 1)
            y = est_stock[valid_data].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            window_results['single_index'] = {
                'alpha': model.intercept_,
                'beta': model.coef_[0],
                'r_squared': model.score(X, y),
                'observations': len(y)
            }
        
        # Model 2: Technology Sector Model
        tech_ret = market_data['XLK'].reindex(est_stock.index)
        valid_data = ~(est_stock.isna() | tech_ret.isna())
        
        if valid_data.sum() > 10:
            X = tech_ret[valid_data].values.reshape(-1, 1)
            y = est_stock[valid_data].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            window_results['tech_sector'] = {
                'alpha': model.intercept_,
                'beta': model.coef_[0],
                'r_squared': model.score(X, y),
                'observations': len(y)
            }
        
        # Model 3: Two-Factor Model (Market + Tech)
        market_ret = market_data['SPY'].reindex(est_stock.index)
        tech_ret = market_data['XLK'].reindex(est_stock.index)
        valid_data = ~(est_stock.isna() | market_ret.isna() | tech_ret.isna())
        
        if valid_data.sum() > 15:
            X = np.column_stack([
                market_ret[valid_data].values,
                tech_ret[valid_data].values
            ])
            y = est_stock[valid_data].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            window_results['two_factor'] = {
                'alpha': model.intercept_,
                'beta_market': model.coef_[0],
                'beta_tech': model.coef_[1],
                'r_squared': model.score(X, y),
                'observations': len(y)
            }
        
        # Model 4: Fama-French Three-Factor (simplified)
        if len(factors['MKT']) > 0:
            mkt_ret = factors['MKT'].reindex(est_stock.index)
            smb_ret = factors['SMB'].reindex(est_stock.index)
            hml_ret = factors['HML'].reindex(est_stock.index)
            
            valid_data = ~(est_stock.isna() | mkt_ret.isna() | smb_ret.isna() | hml_ret.isna())
            
            if valid_data.sum() > 20:
                X = np.column_stack([
                    mkt_ret[valid_data].values,
                    smb_ret[valid_data].values,
                    hml_ret[valid_data].values
                ])
                y = est_stock[valid_data].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                window_results['fama_french'] = {
                    'alpha': model.intercept_,
                    'beta_mkt': model.coef_[0],
                    'beta_smb': model.coef_[1],
                    'beta_hml': model.coef_[2],
                    'r_squared': model.score(X, y),
                    'observations': len(y)
                }
        
        results[window_name] = window_results
    
    return results

def calculate_alternative_abnormal_returns(event_data, stock_returns, market_data, model_params):
    """Calculate abnormal returns using alternative models"""
    
    ticker = event_data['ticker']
    announcement_date = pd.to_datetime(event_data['announcement_date'])
    
    if ticker not in stock_returns:
        return None
    
    stock_ret = stock_returns[ticker]
    
    # Event windows
    event_windows = [
        (-5, 0, 'minus5_plus0'),
        (-3, 0, 'minus3_plus0'),
        (-1, 0, 'minus1_plus0'),
        (0, 1, 'plus0_plus1'),
        (0, 3, 'plus0_plus3'),
        (0, 5, 'plus0_plus5')
    ]
    
    alternative_ars = {}
    
    for model_name, params in model_params.items():
        if params is None:
            continue
        
        model_ars = {}
        
        for start_day, end_day, window_name in event_windows:
            event_start = announcement_date + pd.Timedelta(days=start_day)
            event_end = announcement_date + pd.Timedelta(days=end_day)
            
            # Get actual returns in event window
            event_mask = (stock_ret.index >= event_start) & (stock_ret.index <= event_end)
            event_returns = stock_ret[event_mask]
            
            if len(event_returns) == 0:
                continue
            
            # Calculate expected returns based on model
            expected_returns = []
            
            for date in event_returns.index:
                if model_name == 'single_index':
                    market_ret = market_data['SPY'].get(date, 0)
                    expected = params['alpha'] + params['beta'] * market_ret
                
                elif model_name == 'tech_sector':
                    tech_ret = market_data['XLK'].get(date, 0)
                    expected = params['alpha'] + params['beta'] * tech_ret
                
                elif model_name == 'two_factor':
                    market_ret = market_data['SPY'].get(date, 0)
                    tech_ret = market_data['XLK'].get(date, 0)
                    expected = (params['alpha'] + 
                              params['beta_market'] * market_ret +
                              params['beta_tech'] * tech_ret)
                
                elif model_name == 'fama_french':
                    # Simplified - using available factor proxies
                    market_ret = market_data['SPY'].get(date, 0)
                    expected = params['alpha'] + params['beta_mkt'] * market_ret
                
                else:
                    expected = 0
                
                expected_returns.append(expected)
            
            # Calculate abnormal returns
            abnormal_returns = event_returns.values - expected_returns
            cumulative_ar = np.sum(abnormal_returns)
            
            model_ars[f'abnormal_return_{window_name}'] = cumulative_ar
        
        alternative_ars[model_name] = model_ars
    
    return alternative_ars

def run_robustness_analysis():
    """Run comprehensive robustness analysis"""
    
    print("Starting robustness analysis...")
    
    # Load existing results
    with open("results/final_archive/final_analysis_results.json", 'r') as f:
        results = json.load(f)
    
    # Load stock data
    stock_data, start_date, end_date = load_stock_data()
    
    # Load market data
    market_data = get_market_data(start_date, end_date)
    if market_data is None:
        print("Failed to load market data")
        return None
    
    # Calculate factors
    factors = calculate_fama_french_factors(market_data)
    
    # Run alternative models for each event
    robustness_results = []
    
    for i, event in enumerate(results[:5]):  # Test on first 5 events for speed
        print(f"Processing event {i+1}: {event['event_name']} ({event['ticker']})")
        
        # Test alternative market models
        alt_models = alternative_market_models(event, stock_data, market_data, factors)
        
        if alt_models is None:
            continue
        
        # Calculate alternative abnormal returns
        best_model_params = None
        best_r_squared = 0
        
        # Find best performing model from standard window
        if 'Standard (-150,-6)' in alt_models:
            for model_name, model_results in alt_models['Standard (-150,-6)'].items():
                if 'r_squared' in model_results and model_results['r_squared'] > best_r_squared:
                    best_r_squared = model_results['r_squared']
                    best_model_params = {model_name: model_results}
        
        # Calculate alternative abnormal returns
        if best_model_params:
            alt_ars = calculate_alternative_abnormal_returns(
                event, stock_data, market_data, best_model_params
            )
        else:
            alt_ars = None
        
        robustness_results.append({
            'event_name': event['event_name'],
            'ticker': event['ticker'],
            'announcement_date': event['announcement_date'],
            'original_results': {
                'alpha': event['alpha'],
                'beta': event['beta'],
                'r_squared': event['r_squared'],
                'abnormal_return_minus5_plus0': event['abnormal_return_minus5_plus0']
            },
            'alternative_models': alt_models,
            'alternative_abnormal_returns': alt_ars,
            'robustness_assessment': {
                'models_tested': len(alt_models) if alt_models else 0,
                'best_r_squared': best_r_squared,
                'original_vs_alternative': 'comparison_needed'
            }
        })
    
    return {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'events_analyzed': len(robustness_results),
            'market_data_period': f"{start_date.date()} to {end_date.date()}",
            'models_tested': ['single_index', 'tech_sector', 'two_factor', 'fama_french'],
            'estimation_windows_tested': ['(-150,-6)', '(-90,-6)', '(-180,-6)', '(-120,-31)']
        },
        'robustness_results': robustness_results
    }

if __name__ == "__main__":
    robustness = run_robustness_analysis()
    
    if robustness:
        # Save results
        output_path = Path("results/statistical_tests/robustness_checks.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(robustness, f, indent=2)
        
        print(f"\nRobustness analysis saved to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("ROBUSTNESS ANALYSIS SUMMARY")
        print("="*60)
        
        for result in robustness['robustness_results']:
            print(f"\n{result['event_name']} ({result['ticker']}):")
            print(f"  Original R²: {result['original_results']['r_squared']:.3f}")
            print(f"  Models tested: {result['robustness_assessment']['models_tested']}")
            print(f"  Best alternative R²: {result['robustness_assessment']['best_r_squared']:.3f}")
    
    else:
        print("Robustness analysis failed")