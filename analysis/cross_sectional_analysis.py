"""
Enhanced Cross-sectional Analysis
Regression analysis with control variables and heterogeneity tests
Building on existing options and event study results
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

def load_all_data():
    """Load all available data sources"""
    
    # Load main results
    with open("results/final_archive/final_analysis_results.json", 'r') as f:
        main_results = json.load(f)
    
    # Load options data
    try:
        with open("results/options/options_analysis_data.json", 'r') as f:
            options_data = json.load(f)
    except FileNotFoundError:
        options_data = None
    
    return main_results, options_data

def create_regression_dataset():
    """Create dataset for cross-sectional regression analysis"""
    
    main_results, options_data = load_all_data()
    
    regression_data = []
    
    for event in main_results:
        # Dependent variables (abnormal returns)
        dependent_vars = {
            'ar_minus5_0': event['abnormal_return_minus5_plus0'],
            'ar_minus3_0': event['abnormal_return_minus3_plus0'],
            'ar_minus1_0': event['abnormal_return_minus1_plus0']
        }
        
        # Independent variables from existing data
        independent_vars = {
            # Market model characteristics
            'beta': event['beta'],
            'r_squared': event['r_squared'],
            'alpha': event['alpha'],
            
            # Volume characteristics
            'volume_spike': event['volume_spike'],
            'dollar_volume': event['dollar_volume'],
            'turnover': event['turnover'],
            
            # Company dummies
            'apple': 1 if event['company'] == 'Apple' else 0,
            'nvidia': 1 if event['company'] == 'NVIDIA' else 0,
            'microsoft': 1 if event['company'] == 'Microsoft' else 0,
            'tesla': 1 if event['company'] == 'Tesla' else 0,
            'amd': 1 if event['company'] == 'AMD' else 0,
            'sony': 1 if event['company'] == 'Sony' else 0,
            
            # Category dummies
            'consumer_hardware': 1 if 'Consumer' in event['category'] or 'Hardware' in event['category'] else 0,
            'ai_hardware': 1 if 'AI' in event['category'] else 0,
            'semiconductor': 1 if 'Semiconductor' in event['category'] else 0,
            'gaming': 1 if 'Gaming' in event['category'] else 0,
            'software': 1 if 'Software' in event['category'] else 0,
            
            # Time controls
            'announcement_year': pd.to_datetime(event['announcement_date']).year,
            'covid_period': 1 if pd.to_datetime(event['announcement_date']) >= pd.to_datetime('2020-03-01') and pd.to_datetime(event['announcement_date']) <= pd.to_datetime('2021-06-01') else 0,
            
            # Market characteristics (estimated)
            'market_cap_proxy': np.log(event['dollar_volume']) if event['dollar_volume'] > 0 else 0,  # Proxy for firm size
            'volatility_proxy': abs(event['raw_return_minus5_plus0']) if not pd.isna(event['raw_return_minus5_plus0']) else 0,
        }
        
        # Add options variables if available
        if options_data:
            # Find matching options data
            options_match = None
            for opt_event in options_data.get('anomaly_analysis', []):
                if (opt_event['ticker'] == event['ticker'] and 
                    opt_event['event_name'] == event['event_name']):
                    options_match = opt_event
                    break
            
            if options_match:
                independent_vars.update({
                    'options_volume': options_match.get('total_volume', 0),
                    'put_call_ratio': options_match.get('pcr_volume', 0),
                    'otm_ratio': options_match.get('otm_volume_ratio', 0),
                    'short_term_ratio': options_match.get('short_term_ratio', 0),
                    'high_volume_anomaly': 1 if options_match.get('high_volume_anomaly', False) else 0,
                    'high_anomaly_event': 1 if options_match.get('high_anomaly_event', False) else 0,
                    'anomaly_score': options_match.get('anomaly_score', 0)
                })
            else:
                # Fill with zeros if no options data available
                independent_vars.update({
                    'options_volume': 0,
                    'put_call_ratio': 0,
                    'otm_ratio': 0,
                    'short_term_ratio': 0,
                    'high_volume_anomaly': 0,
                    'high_anomaly_event': 0,
                    'anomaly_score': 0
                })
        
        # Combine all variables
        row_data = {
            'event_name': event['event_name'],
            'ticker': event['ticker'],
            'company': event['company'],
            'category': event['category'],
            'announcement_date': event['announcement_date']
        }
        row_data.update(dependent_vars)
        row_data.update(independent_vars)
        
        regression_data.append(row_data)
    
    return pd.DataFrame(regression_data)

def run_baseline_regressions(df):
    """Run baseline cross-sectional regressions"""
    
    # Define dependent variables
    dependent_vars = ['ar_minus5_0', 'ar_minus3_0', 'ar_minus1_0']
    
    # Define sets of independent variables for different specifications
    specifications = {
        'basic': ['beta', 'r_squared', 'volume_spike'],
        'options': ['beta', 'r_squared', 'volume_spike', 'put_call_ratio', 'otm_ratio', 'high_volume_anomaly'],
        'company_controls': ['beta', 'r_squared', 'volume_spike', 'apple', 'nvidia', 'microsoft', 'tesla'],
        'full_model': ['beta', 'r_squared', 'volume_spike', 'put_call_ratio', 'otm_ratio', 
                      'apple', 'nvidia', 'microsoft', 'tesla', 'covid_period', 'market_cap_proxy']
    }
    
    regression_results = {}
    
    for dep_var in dependent_vars:
        regression_results[dep_var] = {}
        
        # Remove rows with missing dependent variable
        df_clean = df.dropna(subset=[dep_var])
        
        for spec_name, indep_vars in specifications.items():
            # Remove independent variables that don't exist or have all missing values
            available_vars = [var for var in indep_vars if var in df_clean.columns and not df_clean[var].isna().all()]
            
            if len(available_vars) == 0:
                continue
                
            # Remove rows with missing independent variables
            df_reg = df_clean.dropna(subset=available_vars)
            
            if len(df_reg) < 10:  # Need minimum observations
                continue
            
            # Prepare data
            X = df_reg[available_vars]
            y = df_reg[dep_var]
            
            # Standardize variables for interpretation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Run regression
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Calculate statistics
            y_pred = model.predict(X_scaled)
            residuals = y - y_pred
            
            # R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Adjusted R-squared
            n = len(y)
            k = len(available_vars)
            adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1)) if n > k + 1 else np.nan
            
            # Store results
            regression_results[dep_var][spec_name] = {
                'coefficients': dict(zip(available_vars, model.coef_)),
                'intercept': model.intercept_,
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'observations': n,
                'variables_included': available_vars,
                'mean_dependent_var': np.mean(y),
                'residual_std': np.std(residuals)
            }
    
    return regression_results

def heterogeneity_analysis(df):
    """Analyze heterogeneity across different dimensions"""
    
    heterogeneity_results = {}
    
    # Analysis by company
    company_analysis = {}
    for company in df['company'].unique():
        if pd.isna(company):
            continue
            
        company_data = df[df['company'] == company]
        
        if len(company_data) > 1:
            company_analysis[company] = {
                'events': len(company_data),
                'mean_ar_minus5_0': company_data['ar_minus5_0'].mean(),
                'std_ar_minus5_0': company_data['ar_minus5_0'].std(),
                'mean_volume_spike': company_data['volume_spike'].mean(),
                'mean_beta': company_data['beta'].mean()
            }
    
    heterogeneity_results['by_company'] = company_analysis
    
    # Analysis by product category
    category_analysis = {}
    for category in df['category'].unique():
        if pd.isna(category):
            continue
            
        category_data = df[df['category'] == category]
        
        if len(category_data) > 0:
            category_analysis[category] = {
                'events': len(category_data),
                'mean_ar_minus5_0': category_data['ar_minus5_0'].mean() if not category_data['ar_minus5_0'].isna().all() else np.nan,
                'companies': category_data['company'].unique().tolist()
            }
    
    heterogeneity_results['by_category'] = category_analysis
    
    # High vs Low Volume Events
    volume_median = df['volume_spike'].median()
    high_volume = df[df['volume_spike'] > volume_median]
    low_volume = df[df['volume_spike'] <= volume_median]
    
    heterogeneity_results['volume_split'] = {
        'high_volume': {
            'events': len(high_volume),
            'mean_ar_minus5_0': high_volume['ar_minus5_0'].mean(),
            'mean_volume_spike': high_volume['volume_spike'].mean()
        },
        'low_volume': {
            'events': len(low_volume),
            'mean_ar_minus5_0': low_volume['ar_minus5_0'].mean(),
            'mean_volume_spike': low_volume['volume_spike'].mean()
        }
    }
    
    # COVID period analysis
    covid_events = df[df['covid_period'] == 1]
    non_covid_events = df[df['covid_period'] == 0]
    
    heterogeneity_results['covid_analysis'] = {
        'covid_period': {
            'events': len(covid_events),
            'mean_ar_minus5_0': covid_events['ar_minus5_0'].mean() if len(covid_events) > 0 else np.nan
        },
        'non_covid_period': {
            'events': len(non_covid_events),
            'mean_ar_minus5_0': non_covid_events['ar_minus5_0'].mean() if len(non_covid_events) > 0 else np.nan
        }
    }
    
    return heterogeneity_results

def test_key_hypotheses(df):
    """Test specific hypotheses mentioned in the paper"""
    
    hypothesis_tests = {}
    
    # H1: Options volume predicts abnormal returns
    if 'options_volume' in df.columns and not df['options_volume'].isna().all():
        volume_corr = df[['ar_minus5_0', 'options_volume']].corr().iloc[0,1]
        hypothesis_tests['options_volume_predicts_returns'] = {
            'correlation': volume_corr,
            'interpretation': 'positive' if volume_corr > 0 else 'negative' if volume_corr < 0 else 'no_relationship'
        }
    
    # H2: Put-call ratio relationships
    if 'put_call_ratio' in df.columns and not df['put_call_ratio'].isna().all():
        pcr_corr = df[['ar_minus5_0', 'put_call_ratio']].corr().iloc[0,1]
        hypothesis_tests['put_call_ratio_relationship'] = {
            'correlation': pcr_corr,
            'interpretation': 'higher_pcr_lower_returns' if pcr_corr < 0 else 'higher_pcr_higher_returns' if pcr_corr > 0 else 'no_relationship'
        }
    
    # H3: High anomaly events have different returns
    if 'high_anomaly_event' in df.columns:
        high_anomaly_events = df[df['high_anomaly_event'] == 1]['ar_minus5_0']
        normal_events = df[df['high_anomaly_event'] == 0]['ar_minus5_0']
        
        if len(high_anomaly_events) > 0 and len(normal_events) > 0:
            t_stat, p_value = stats.ttest_ind(high_anomaly_events.dropna(), normal_events.dropna())
            
            hypothesis_tests['high_anomaly_different_returns'] = {
                'high_anomaly_mean': high_anomaly_events.mean(),
                'normal_events_mean': normal_events.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': bool(p_value < 0.05)
            }
    
    return hypothesis_tests

def run_comprehensive_cross_sectional_analysis():
    """Run comprehensive cross-sectional analysis"""
    
    print("Running comprehensive cross-sectional analysis...")
    
    # Create regression dataset
    df = create_regression_dataset()
    print(f"Created dataset with {len(df)} events and {len(df.columns)} variables")
    
    # Run baseline regressions
    regression_results = run_baseline_regressions(df)
    
    # Heterogeneity analysis
    heterogeneity = heterogeneity_analysis(df)
    
    # Hypothesis tests
    hypothesis_tests = test_key_hypotheses(df)
    
    # Compile comprehensive results
    comprehensive_results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'total_events': len(df),
            'variables_available': list(df.columns),
            'regression_specifications': list(regression_results.get('ar_minus5_0', {}).keys())
        },
        'dataset_summary': {
            'descriptive_stats': {k: {k2: float(v2) if not pd.isna(v2) else None for k2, v2 in v.items()} 
                                for k, v in df.describe().to_dict().items()},
            'key_correlations': {
                'ar_volume_corr': float(df[['ar_minus5_0', 'volume_spike']].corr().iloc[0,1]) if len(df) > 1 else None
            }
        },
        'regression_results': regression_results,
        'heterogeneity_analysis': heterogeneity,
        'hypothesis_tests': hypothesis_tests
    }
    
    return comprehensive_results

if __name__ == "__main__":
    results = run_comprehensive_cross_sectional_analysis()
    
    # Save results
    output_path = Path("results/statistical_tests/cross_sectional_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Cross-sectional analysis saved to {output_path}")
    
    # Print key results
    print("\n" + "="*60)
    print("CROSS-SECTIONAL ANALYSIS SUMMARY")
    print("="*60)
    
    # Regression results summary
    if 'ar_minus5_0' in results['regression_results']:
        print("\nRegression Results (Pre-launch AR -5,0):")
        for spec_name, spec_results in results['regression_results']['ar_minus5_0'].items():
            r_sq = spec_results['r_squared']
            n_obs = spec_results['observations']
            print(f"  {spec_name}: R² = {r_sq:.3f}, N = {n_obs}")
            
            # Show significant coefficients (simplified)
            coeffs = spec_results['coefficients']
            for var, coeff in coeffs.items():
                if abs(coeff) > 0.001:  # Rough significance filter
                    print(f"    {var}: {coeff:+.3f}")
    
    # Heterogeneity summary
    print(f"\nHeterogeneity Analysis:")
    if 'by_company' in results['heterogeneity_analysis']:
        print("  Mean abnormal returns by company:")
        for company, stats in results['heterogeneity_analysis']['by_company'].items():
            mean_ar = stats['mean_ar_minus5_0']
            events = stats['events']
            print(f"    {company}: {mean_ar:+.2%} ({events} events)")
    
    # Hypothesis tests
    if results['hypothesis_tests']:
        print(f"\nKey Hypothesis Tests:")
        for test_name, test_results in results['hypothesis_tests'].items():
            print(f"  {test_name.replace('_', ' ').title()}:")
            if 'correlation' in test_results:
                print(f"    Correlation: {test_results['correlation']:.3f}")
            if 'p_value' in test_results:
                significance = "significant" if test_results['p_value'] < 0.05 else "not significant"
                print(f"    P-value: {test_results['p_value']:.3f} ({significance})")
    
    print("\nCross-sectional analysis completed!")