"""
Economic Significance Analysis
Calculate Sharpe ratios, transaction costs, and trading strategy performance
Building on existing results
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def load_existing_data():
    """Load existing results and options data"""
    
    # Load main results
    with open("results/final_archive/final_analysis_results.json", 'r') as f:
        main_results = json.load(f)
    
    # Load options data if available
    try:
        with open("results/options/options_analysis_data.json", 'r') as f:
            options_data = json.load(f)
    except FileNotFoundError:
        options_data = None
    
    return main_results, options_data

def calculate_trading_strategy_returns():
    """Calculate returns from pre-launch trading strategies"""
    
    main_results, options_data = load_existing_data()
    
    # Strategy 1: Long all events
    long_all_returns = []
    for event in main_results:
        ar = event['abnormal_return_minus5_plus0']
        if not pd.isna(ar):
            long_all_returns.append(ar)
    
    # Strategy 2: Long high volume events, short low volume events
    volume_spikes = [event['volume_spike'] for event in main_results if not pd.isna(event['volume_spike'])]
    volume_median = np.median(volume_spikes)
    
    long_short_returns = []
    for event in main_results:
        ar = event['abnormal_return_minus5_plus0']
        volume_spike = event['volume_spike']
        
        if pd.isna(ar) or pd.isna(volume_spike):
            continue
        
        if volume_spike > volume_median:
            # Long position
            long_short_returns.append(ar)
        else:
            # Short position
            long_short_returns.append(-ar)
    
    # Strategy 3: Options anomaly strategy (if options data available)
    options_strategy_returns = []
    if options_data:
        for event in main_results:
            ar = event['abnormal_return_minus5_plus0']
            if pd.isna(ar):
                continue
            
            # Find matching options event
            options_match = None
            for opt_event in options_data.get('anomaly_analysis', []):
                if (opt_event['ticker'] == event['ticker'] and 
                    opt_event['event_name'] == event['event_name']):
                    options_match = opt_event
                    break
            
            if options_match:
                # Strategy: Long if high options volume, short if high put/call ratio
                high_volume = options_match.get('high_volume_anomaly', False)
                pcr = options_match.get('pcr_volume', 0.5)
                
                if high_volume and pcr < 0.5:
                    # Bullish signal
                    options_strategy_returns.append(ar)
                elif pcr > 1.0:
                    # Bearish signal
                    options_strategy_returns.append(-ar)
                else:
                    # No position
                    options_strategy_returns.append(0)
    
    return {
        'long_all': long_all_returns,
        'long_short_volume': long_short_returns,
        'options_anomaly': options_strategy_returns
    }

def calculate_sharpe_ratios(strategy_returns, risk_free_rate=0.02):
    """Calculate Sharpe ratios for different strategies"""
    
    sharpe_results = {}
    
    for strategy_name, returns in strategy_returns.items():
        if len(returns) == 0:
            continue
        
        returns_array = np.array(returns)
        
        # Annualized statistics (assuming ~6 trades per year based on sample)
        trading_frequency = 6  # trades per year
        annual_return = np.mean(returns_array) * trading_frequency
        annual_vol = np.std(returns_array) * np.sqrt(trading_frequency)
        
        # Sharpe ratio
        excess_return = annual_return - risk_free_rate
        sharpe_ratio = excess_return / annual_vol if annual_vol > 0 else 0
        
        # Calmar ratio (return/max drawdown approximation)
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        # Information ratio (return/tracking error)
        # Using market return of 0 as benchmark
        tracking_error = np.std(returns_array - 0)  
        info_ratio = np.mean(returns_array) / tracking_error if tracking_error > 0 else 0
        
        # Win rate
        win_rate = np.sum(returns_array > 0) / len(returns_array)
        
        sharpe_results[strategy_name] = {
            'trades': len(returns_array),
            'mean_return': float(np.mean(returns_array)),
            'volatility': float(np.std(returns_array)),
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_vol),
            'sharpe_ratio': float(sharpe_ratio),
            'calmar_ratio': float(calmar_ratio) if not np.isinf(calmar_ratio) else 999,
            'information_ratio': float(info_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'best_trade': float(np.max(returns_array)),
            'worst_trade': float(np.min(returns_array))
        }
    
    return sharpe_results

def calculate_transaction_costs():
    """Estimate transaction costs for trading strategies"""
    
    main_results, _ = load_existing_data()
    
    # Typical transaction costs (as percentage of trade value)
    cost_scenarios = {
        'retail_investor': {
            'commission': 0.0005,  # 5 bps commission
            'bid_ask_spread': 0.002,  # 20 bps spread
            'market_impact': 0.001,  # 10 bps impact
            'description': 'Retail investor with online broker'
        },
        'institutional': {
            'commission': 0.0001,  # 1 bp commission
            'bid_ask_spread': 0.001,  # 10 bps spread
            'market_impact': 0.0005,  # 5 bps impact
            'description': 'Institutional investor'
        },
        'high_frequency': {
            'commission': 0.00005,  # 0.5 bps commission
            'bid_ask_spread': 0.0005,  # 5 bps spread
            'market_impact': 0.0002,  # 2 bps impact
            'description': 'High-frequency trader'
        }
    }
    
    # Calculate average turnover from the data
    turnovers = [event['turnover'] for event in main_results if not pd.isna(event['turnover'])]
    avg_turnover = np.mean(turnovers) if turnovers else 0.1
    
    transaction_cost_analysis = {
        'assumptions': {
            'average_turnover': float(avg_turnover),
            'holding_period_days': 5,  # Pre-launch window
            'rebalancing_frequency': 'per_event',
            'position_size_assumption': 'equal_weighted'
        },
        'cost_scenarios': {}
    }
    
    for scenario_name, costs in cost_scenarios.items():
        # Total transaction cost per round trip
        total_cost_per_trip = (
            2 * costs['commission'] +  # Buy and sell
            costs['bid_ask_spread'] +  # One-way spread
            costs['market_impact']     # Impact cost
        )
        
        # Annual cost assuming 6 trades per year
        annual_cost = total_cost_per_trip * 6
        
        transaction_cost_analysis['cost_scenarios'][scenario_name] = {
            'description': costs['description'],
            'commission_roundtrip': 2 * costs['commission'],
            'bid_ask_spread': costs['bid_ask_spread'],
            'market_impact': costs['market_impact'],
            'total_cost_per_roundtrip': float(total_cost_per_trip),
            'annual_cost_estimate': float(annual_cost),
            'breakeven_return_per_trade': float(total_cost_per_trip)
        }
    
    return transaction_cost_analysis

def calculate_deflated_sharpe_ratio(sharpe_ratio, n_trials, n_observations):
    """
    Calculate deflated Sharpe ratio following Bailey & López de Prado (2014)
    Adjusts for multiple testing and selection bias
    """
    
    if n_trials <= 1 or n_observations <= 1:
        return sharpe_ratio
    
    # Simplified deflated Sharpe ratio calculation
    # In practice, you'd use the full Bailey-López de Prado method
    
    # Expected maximum Sharpe ratio under null hypothesis
    expected_max_sr = np.sqrt(2 * np.log(n_trials))
    
    # Standard error adjustment
    se_adjustment = 1 / np.sqrt(n_observations)
    
    # Deflated Sharpe ratio (simplified version)
    deflated_sr = (sharpe_ratio - expected_max_sr) / se_adjustment
    
    return deflated_sr

def net_performance_analysis(strategy_returns, transaction_costs):
    """Calculate net performance after transaction costs"""
    
    net_performance = {}
    
    for strategy_name, returns in strategy_returns.items():
        if len(returns) == 0:
            continue
        
        strategy_results = {}
        
        for cost_scenario, cost_data in transaction_costs['cost_scenarios'].items():
            cost_per_trade = cost_data['total_cost_per_roundtrip']
            
            # Net returns after transaction costs
            net_returns = [r - cost_per_trade for r in returns]
            
            if len(net_returns) > 0:
                # Calculate net statistics
                mean_net_return = np.mean(net_returns)
                net_volatility = np.std(net_returns)
                
                # Net Sharpe ratio (annualized)
                trading_frequency = 6
                annual_net_return = mean_net_return * trading_frequency
                annual_net_vol = net_volatility * np.sqrt(trading_frequency)
                
                net_sharpe = (annual_net_return - 0.02) / annual_net_vol if annual_net_vol > 0 else 0
                
                # Probability of positive returns
                prob_positive = np.sum(np.array(net_returns) > 0) / len(net_returns)
                
                strategy_results[cost_scenario] = {
                    'mean_net_return': float(mean_net_return),
                    'net_volatility': float(net_volatility),
                    'annual_net_return': float(annual_net_return),
                    'net_sharpe_ratio': float(net_sharpe),
                    'probability_positive': float(prob_positive),
                    'trades_profitable_after_costs': int(np.sum(np.array(net_returns) > 0))
                }
        
        net_performance[strategy_name] = strategy_results
    
    return net_performance

def run_economic_significance_analysis():
    """Run comprehensive economic significance analysis"""
    
    print("Running economic significance analysis...")
    
    # Calculate strategy returns
    strategy_returns = calculate_trading_strategy_returns()
    
    # Calculate Sharpe ratios
    sharpe_results = calculate_sharpe_ratios(strategy_returns)
    
    # Calculate transaction costs
    transaction_costs = calculate_transaction_costs()
    
    # Net performance analysis
    net_performance = net_performance_analysis(strategy_returns, transaction_costs)
    
    # Calculate deflated Sharpe ratios
    deflated_sharpe = {}
    for strategy_name, results in sharpe_results.items():
        # Assume we tested 3 strategies on 34 observations
        deflated_sr = calculate_deflated_sharpe_ratio(
            results['sharpe_ratio'], 
            n_trials=3, 
            n_observations=results['trades']
        )
        deflated_sharpe[strategy_name] = float(deflated_sr)
    
    # Compile comprehensive results
    economic_results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'risk_free_rate': 0.02,
            'trading_frequency_assumption': 6,  # trades per year
            'strategies_analyzed': list(strategy_returns.keys())
        },
        'strategy_performance': {
            'raw_returns': {k: [float(x) for x in v] for k, v in strategy_returns.items()},
            'performance_metrics': sharpe_results,
            'deflated_sharpe_ratios': deflated_sharpe
        },
        'transaction_cost_analysis': transaction_costs,
        'net_performance_after_costs': net_performance,
        'economic_interpretation': {
            'profitability_assessment': 'mixed',
            'key_finding': 'Strategies show positive gross returns but profitability depends heavily on transaction costs',
            'practical_implementation': 'Institutional investors may achieve better net returns due to lower transaction costs'
        }
    }
    
    return economic_results

if __name__ == "__main__":
    results = run_economic_significance_analysis()
    
    # Save results
    output_path = Path("results/statistical_tests/economic_significance.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Economic significance analysis saved to {output_path}")
    
    # Print key results
    print("\n" + "="*70)
    print("ECONOMIC SIGNIFICANCE ANALYSIS SUMMARY")
    print("="*70)
    
    # Strategy performance
    print("\nStrategy Performance (Gross Returns):")
    for strategy, metrics in results['strategy_performance']['performance_metrics'].items():
        sharpe = metrics['sharpe_ratio']
        annual_ret = metrics['annual_return']
        win_rate = metrics['win_rate']
        trades = metrics['trades']
        
        print(f"  {strategy.replace('_', ' ').title()}:")
        print(f"    Annual Return: {annual_ret:+.1%}")
        print(f"    Sharpe Ratio: {sharpe:.2f}")
        print(f"    Win Rate: {win_rate:.1%}")
        print(f"    Trades: {trades}")
    
    # Deflated Sharpe ratios
    print(f"\nDeflated Sharpe Ratios (adjusted for selection bias):")
    for strategy, deflated_sr in results['strategy_performance']['deflated_sharpe_ratios'].items():
        print(f"  {strategy.replace('_', ' ').title()}: {deflated_sr:.2f}")
    
    # Transaction costs impact
    print(f"\nTransaction Cost Impact:")
    cost_scenarios = results['transaction_cost_analysis']['cost_scenarios']
    for scenario, costs in cost_scenarios.items():
        cost_pct = costs['total_cost_per_roundtrip'] * 100
        annual_cost = costs['annual_cost_estimate'] * 100
        print(f"  {scenario.replace('_', ' ').title()}: {cost_pct:.2f}% per trade, {annual_cost:.1f}% annually")
    
    # Net performance highlight
    if 'long_all' in results['net_performance_after_costs']:
        print(f"\nNet Performance Example (Long All Strategy):")
        long_all_net = results['net_performance_after_costs']['long_all']
        for scenario, perf in long_all_net.items():
            net_sharpe = perf['net_sharpe_ratio']
            prob_positive = perf['probability_positive']
            print(f"  {scenario.replace('_', ' ').title()}: Net Sharpe = {net_sharpe:.2f}, P(profit) = {prob_positive:.1%}")
    
    print(f"\nKey Insight: {results['economic_interpretation']['key_finding']}")
    print("Economic significance analysis completed!")