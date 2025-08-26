"""
Non-parametric Statistical Tests for Event Study Analysis
Implements Corrado rank test and sign test for abnormal returns
Building on existing event study results
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import json
from pathlib import Path

def load_existing_results():
    """Load existing final analysis results"""
    results_path = Path("results/final_archive/final_analysis_results.json")
    with open(results_path, 'r') as f:
        return json.load(f)

def corrado_rank_test(abnormal_returns):
    """
    Implements Corrado (1989) rank test for abnormal returns
    
    Parameters:
    abnormal_returns: array of abnormal returns for the event window
    
    Returns:
    test_statistic: Corrado rank test statistic
    p_value: two-tailed p-value
    """
    n = len(abnormal_returns)
    if n <= 1:
        return np.nan, np.nan
    
    # Rank the abnormal returns
    ranks = stats.rankdata(abnormal_returns)
    
    # Expected rank under null hypothesis
    expected_rank = (n + 1) / 2
    
    # Calculate test statistic
    numerator = np.sum(ranks - expected_rank)
    denominator = np.sqrt(n * (n + 1) * (2 * n + 1) / 12)
    
    if denominator == 0:
        return np.nan, np.nan
    
    test_statistic = numerator / denominator
    
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(test_statistic)))
    
    return test_statistic, p_value

def sign_test(abnormal_returns):
    """
    Implements Corrado and Zivney (1992) sign test
    Tests if proportion of positive returns differs from 0.5
    
    Parameters:
    abnormal_returns: array of abnormal returns
    
    Returns:
    test_statistic: standardized sign test statistic
    p_value: two-tailed p-value
    proportion_positive: proportion of positive returns
    """
    n = len(abnormal_returns)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    # Count positive returns
    positive_count = np.sum(abnormal_returns > 0)
    proportion_positive = positive_count / n
    
    # Under null hypothesis, expected proportion is 0.5
    expected_positive = n * 0.5
    
    # Test statistic (standardized)
    if n > 0:
        test_statistic = (positive_count - expected_positive) / np.sqrt(n * 0.5 * 0.5)
    else:
        test_statistic = np.nan
    
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(test_statistic)))
    
    return test_statistic, p_value, proportion_positive

def bootstrap_confidence_interval(abnormal_returns, confidence_level=0.95, n_bootstrap=1000):
    """
    Generate bootstrap confidence intervals for mean abnormal return
    
    Parameters:
    abnormal_returns: array of abnormal returns
    confidence_level: confidence level (default 0.95)
    n_bootstrap: number of bootstrap samples
    
    Returns:
    ci_lower: lower bound of confidence interval
    ci_upper: upper bound of confidence interval
    bootstrap_means: array of bootstrap sample means
    """
    if len(abnormal_returns) == 0:
        return np.nan, np.nan, np.array([])
    
    n = len(abnormal_returns)
    bootstrap_means = []
    
    # Generate bootstrap samples
    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(abnormal_returns, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return ci_lower, ci_upper, bootstrap_means

def run_comprehensive_tests():
    """Run all non-parametric tests on existing results"""
    
    # Load existing results
    results = load_existing_results()
    
    # Define event windows to test
    windows = [
        ('abnormal_return_minus5_plus0', '(-5,0)'),
        ('abnormal_return_minus3_plus0', '(-3,0)'),
        ('abnormal_return_minus1_plus0', '(-1,0)'),
        ('abnormal_return_plus0_plus1', '(0,+1)'),
        ('abnormal_return_plus0_plus3', '(0,+3)'),
        ('abnormal_return_plus0_plus5', '(0,+5)')
    ]
    
    comprehensive_results = {
        'metadata': {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'sample_size': len(results),
            'bootstrap_iterations': 1000,
            'confidence_level': 0.95
        },
        'window_tests': {}
    }
    
    for window_field, window_name in windows:
        # Extract abnormal returns for this window
        abnormal_returns = [event[window_field] for event in results if not pd.isna(event[window_field])]
        abnormal_returns = np.array(abnormal_returns)
        
        if len(abnormal_returns) == 0:
            continue
        
        # Calculate basic statistics
        mean_ar = np.mean(abnormal_returns)
        median_ar = np.median(abnormal_returns)
        std_ar = np.std(abnormal_returns)
        
        # Run non-parametric tests
        rank_stat, rank_p = corrado_rank_test(abnormal_returns)
        sign_stat, sign_p, prop_positive = sign_test(abnormal_returns)
        
        # Bootstrap confidence intervals
        ci_lower, ci_upper, bootstrap_means = bootstrap_confidence_interval(abnormal_returns)
        
        # Traditional t-test for comparison
        if len(abnormal_returns) > 1:
            t_stat, t_p = stats.ttest_1samp(abnormal_returns, 0)
        else:
            t_stat, t_p = np.nan, np.nan
        
        window_results = {
            'window_name': window_name,
            'sample_size': len(abnormal_returns),
            'descriptive_stats': {
                'mean': float(mean_ar),
                'median': float(median_ar),
                'std': float(std_ar),
                'min': float(np.min(abnormal_returns)),
                'max': float(np.max(abnormal_returns))
            },
            'parametric_tests': {
                't_statistic': float(t_stat) if not pd.isna(t_stat) else None,
                't_p_value': float(t_p) if not pd.isna(t_p) else None,
                'significant_5pct': bool(t_p < 0.05) if not pd.isna(t_p) else False,
                'significant_1pct': bool(t_p < 0.01) if not pd.isna(t_p) else False
            },
            'nonparametric_tests': {
                'corrado_rank': {
                    'test_statistic': float(rank_stat) if not pd.isna(rank_stat) else None,
                    'p_value': float(rank_p) if not pd.isna(rank_p) else None,
                    'significant_5pct': bool(rank_p < 0.05) if not pd.isna(rank_p) else False,
                    'significant_1pct': bool(rank_p < 0.01) if not pd.isna(rank_p) else False
                },
                'sign_test': {
                    'test_statistic': float(sign_stat) if not pd.isna(sign_stat) else None,
                    'p_value': float(sign_p) if not pd.isna(sign_p) else None,
                    'proportion_positive': float(prop_positive) if not pd.isna(prop_positive) else None,
                    'significant_5pct': bool(sign_p < 0.05) if not pd.isna(sign_p) else False,
                    'significant_1pct': bool(sign_p < 0.01) if not pd.isna(sign_p) else False
                }
            },
            'bootstrap_ci': {
                'ci_lower_95': float(ci_lower) if not pd.isna(ci_lower) else None,
                'ci_upper_95': float(ci_upper) if not pd.isna(ci_upper) else None,
                'bootstrap_std': float(np.std(bootstrap_means)) if len(bootstrap_means) > 0 else None
            }
        }
        
        comprehensive_results['window_tests'][window_field] = window_results
    
    return comprehensive_results

def generate_summary_table():
    """Generate summary table matching paper's Appendix A format"""
    
    results = load_existing_results()
    
    windows = [
        ('abnormal_return_minus5_plus0', '(-5,0)'),
        ('abnormal_return_minus3_plus0', '(-3,0)'),
        ('abnormal_return_minus1_plus0', '(-1,0)'),
        ('abnormal_return_plus0_plus1', '(0,+1)'),
        ('abnormal_return_plus0_plus3', '(0,+3)'),
        ('abnormal_return_plus0_plus5', '(0,+5)')
    ]
    
    summary_table = []
    
    for window_field, window_name in windows:
        abnormal_returns = [event[window_field] for event in results if not pd.isna(event[window_field])]
        abnormal_returns = np.array(abnormal_returns)
        
        if len(abnormal_returns) == 0:
            continue
        
        mean_car = np.mean(abnormal_returns)
        median_car = np.median(abnormal_returns)
        std_car = np.std(abnormal_returns)
        
        # 95% confidence interval (approximation)
        se = std_car / np.sqrt(len(abnormal_returns))
        ci_lower_approx = mean_car - 1.96 * se
        ci_upper_approx = mean_car + 1.96 * se
        
        summary_table.append({
            'window': window_name,
            'mean_car': f"{mean_car:.2%}",
            'median_car': f"{median_car:.2%}",
            'std_dev': f"{std_car:.2%}",
            'ci_95_approx': f"[{ci_lower_approx:.2%}, {ci_upper_approx:.2%}]",
            'sample_size': len(abnormal_returns)
        })
    
    return summary_table

if __name__ == "__main__":
    # Run comprehensive tests
    print("Running comprehensive non-parametric tests...")
    test_results = run_comprehensive_tests()
    
    # Save results
    output_path = Path("results/statistical_tests/nonparametric_tests.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Generate summary table
    summary = generate_summary_table()
    summary_path = Path("results/statistical_tests/summary_table.json")
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"Summary table saved to {summary_path}")
    
    # Print key results
    print("\nKey Findings:")
    for window, results in test_results['window_tests'].items():
        window_name = results['window_name']
        mean_ar = results['descriptive_stats']['mean']
        rank_p = results['nonparametric_tests']['corrado_rank']['p_value']
        sign_p = results['nonparametric_tests']['sign_test']['p_value']
        prop_pos = results['nonparametric_tests']['sign_test']['proportion_positive']
        
        print(f"\n{window_name}: Mean AR = {mean_ar:.3%}")
        print(f"  Rank test p-value: {rank_p:.4f}" if rank_p else "  Rank test: N/A")
        print(f"  Sign test p-value: {sign_p:.4f}" if sign_p else "  Sign test: N/A")
        print(f"  Proportion positive: {prop_pos:.1%}" if prop_pos else "  Proportion: N/A")