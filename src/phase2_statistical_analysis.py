"""
Phase II Statistical Analysis: Core Hypothesis Testing
Tests correlations between pre-launch options anomalies and subsequent:
1. Earnings surprises
2. Stock price movements
3. Regression testing
4. Robustness checks
5. Traditional forecasting comparison
"""
import pandas as pd
import numpy as np
import json
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class Phase2StatisticalAnalyzer:
    """Complete Phase II statistical analysis framework."""
    
    def __init__(self):
        self.results_dir = Path("results")
        self.data_dir = Path("data")
        
    def load_data(self):
        """Load all required datasets."""
        print("Loading datasets...")
        
        # Load events master
        self.events_df = pd.read_csv(self.data_dir / "processed" / "events_master.csv")
        self.events_df['announcement'] = pd.to_datetime(self.events_df['announcement'])
        self.events_df['next_earnings'] = pd.to_datetime(self.events_df['next_earnings'])
        
        # Load options analysis data
        with open(self.results_dir / "options_analysis_data.json", 'r') as f:
            options_data = json.load(f)
        self.options_df = pd.DataFrame(options_data.get('anomaly_analysis', []))
        if not self.options_df.empty:
            self.options_df['event_date'] = pd.to_datetime(self.options_df['event_date'])
        
        # Load equity analysis results
        self.equity_df = pd.read_csv(self.results_dir / "equity_analysis_results.csv")
        self.equity_df['announcement_date'] = pd.to_datetime(self.equity_df['announcement_date'])
        
        print(f"Loaded: {len(self.events_df)} events, {len(self.options_df)} options events, {len(self.equity_df)} equity events")
        
    def get_earnings_surprises(self):
        """Fetch actual earnings data and calculate surprises."""
        print("Fetching earnings data and calculating surprises...")
        
        earnings_data = []
        
        for _, event in self.events_df.iterrows():
            if pd.isna(event['next_earnings']):
                continue
                
            ticker = event['ticker']
            earnings_date = event['next_earnings']
            
            try:
                # Get earnings data from yfinance
                stock = yf.Ticker(ticker)
                
                # Get earnings dates around the expected date
                earnings_calendar = stock.earnings_dates
                if earnings_calendar is not None and len(earnings_calendar) > 0:
                    # Find closest earnings to our expected date
                    earnings_calendar = earnings_calendar.reset_index()
                    earnings_calendar['date_diff'] = abs((earnings_calendar['Earnings Date'] - earnings_date).dt.days)
                    closest_earnings = earnings_calendar.loc[earnings_calendar['date_diff'].idxmin()]
                    
                    # Calculate earnings surprise
                    eps_actual = closest_earnings.get('Reported EPS', np.nan)
                    eps_estimate = closest_earnings.get('EPS Estimate', np.nan)
                    
                    if pd.notna(eps_actual) and pd.notna(eps_estimate) and eps_estimate != 0:
                        eps_surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
                    else:
                        eps_surprise_pct = np.nan
                    
                    earnings_data.append({
                        'event_name': event['name'],
                        'ticker': ticker,
                        'announcement_date': event['announcement'],
                        'earnings_date': closest_earnings['Earnings Date'],
                        'eps_actual': eps_actual,
                        'eps_estimate': eps_estimate,
                        'eps_surprise_pct': eps_surprise_pct
                    })
                
            except Exception as e:
                print(f"Error fetching earnings for {ticker}: {e}")
                continue
        
        self.earnings_df = pd.DataFrame(earnings_data)
        print(f"Collected earnings data for {len(self.earnings_df)} events")
        return self.earnings_df
    
    def get_post_announcement_returns(self, windows=[1, 3, 5, 10]):
        """Calculate post-announcement stock returns for various windows."""
        print("Calculating post-announcement returns...")
        
        returns_data = []
        
        for _, event in self.events_df.iterrows():
            ticker = event['ticker']
            announcement_date = event['announcement']
            
            try:
                # Download stock data
                end_date = announcement_date + timedelta(days=20)
                start_date = announcement_date - timedelta(days=5)
                
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if len(stock_data) > 0:
                    # Find announcement date in data
                    stock_data.index = pd.to_datetime(stock_data.index)
                    announcement_price = None
                    
                    # Get price on announcement date or closest trading day
                    for i in range(5):  # Look up to 5 days ahead
                        check_date = announcement_date + timedelta(days=i)
                        if check_date in stock_data.index:
                            announcement_price = stock_data.loc[check_date, 'Close']
                            break
                    
                    if announcement_price is not None:
                        returns_entry = {
                            'event_name': event['name'],
                            'ticker': ticker,
                            'announcement_date': announcement_date,
                            'announcement_price': announcement_price
                        }
                        
                        # Calculate returns for different windows
                        for window in windows:
                            try:
                                future_date = announcement_date + timedelta(days=window)
                                # Find closest trading day
                                future_price = None
                                for i in range(window, window + 5):
                                    check_date = announcement_date + timedelta(days=i)
                                    if check_date in stock_data.index:
                                        future_price = stock_data.loc[check_date, 'Close']
                                        break
                                
                                if future_price is not None:
                                    return_pct = ((future_price - announcement_price) / announcement_price) * 100
                                    returns_entry[f'return_{window}d'] = return_pct
                                else:
                                    returns_entry[f'return_{window}d'] = np.nan
                            except:
                                returns_entry[f'return_{window}d'] = np.nan
                        
                        returns_data.append(returns_entry)
            
            except Exception as e:
                print(f"Error fetching returns for {ticker}: {e}")
                continue
        
        self.returns_df = pd.DataFrame(returns_data)
        print(f"Collected returns data for {len(self.returns_df)} events")
        return self.returns_df
    
    def correlate_options_with_earnings(self):
        """Test correlation between options anomalies and earnings surprises."""
        print("Testing options anomalies vs earnings surprises correlation...")
        
        if self.options_df.empty or self.earnings_df.empty:
            print("Missing options or earnings data for correlation test")
            return {}
        
        # Merge options and earnings data
        merged_df = pd.merge(
            self.options_df,
            self.earnings_df,
            left_on=['event_name', 'ticker'],
            right_on=['event_name', 'ticker'],
            how='inner'
        )
        
        if len(merged_df) == 0:
            print("No matching events between options and earnings data")
            return {}
        
        print(f"Testing correlations for {len(merged_df)} matched events")
        
        correlation_results = {}
        
        # Test various options metrics against earnings surprise
        options_metrics = [
            'total_volume', 'pcr_volume', 'anomaly_score',
            'otm_volume_ratio', 'short_term_ratio', 'avg_delta'
        ]
        
        for metric in options_metrics:
            if metric in merged_df.columns and 'eps_surprise_pct' in merged_df.columns:
                # Remove NaN values
                clean_data = merged_df[[metric, 'eps_surprise_pct']].dropna()
                
                if len(clean_data) >= 5:  # Need at least 5 data points
                    # Pearson correlation
                    pearson_r, pearson_p = pearsonr(clean_data[metric], clean_data['eps_surprise_pct'])
                    
                    # Spearman correlation (rank-based, more robust)
                    spearman_r, spearman_p = spearmanr(clean_data[metric], clean_data['eps_surprise_pct'])
                    
                    correlation_results[metric] = {
                        'sample_size': len(clean_data),
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'significant_at_05': min(pearson_p, spearman_p) < 0.05,
                        'significant_at_01': min(pearson_p, spearman_p) < 0.01
                    }
        
        return correlation_results
    
    def correlate_options_with_returns(self):
        """Test correlation between options anomalies and subsequent returns."""
        print("Testing options anomalies vs stock returns correlation...")
        
        if self.options_df.empty or self.returns_df.empty:
            print("Missing options or returns data for correlation test")
            return {}
        
        # Merge options and returns data
        merged_df = pd.merge(
            self.options_df,
            self.returns_df,
            left_on=['event_name', 'ticker'],
            right_on=['event_name', 'ticker'],
            how='inner'
        )
        
        if len(merged_df) == 0:
            print("No matching events between options and returns data")
            return {}
        
        print(f"Testing correlations for {len(merged_df)} matched events")
        
        correlation_results = {}
        
        # Test various options metrics against different return windows
        options_metrics = [
            'total_volume', 'pcr_volume', 'anomaly_score',
            'otm_volume_ratio', 'short_term_ratio', 'avg_delta'
        ]
        
        return_windows = ['return_1d', 'return_3d', 'return_5d', 'return_10d']
        
        for metric in options_metrics:
            correlation_results[metric] = {}
            
            for return_window in return_windows:
                if metric in merged_df.columns and return_window in merged_df.columns:
                    # Remove NaN values
                    clean_data = merged_df[[metric, return_window]].dropna()
                    
                    if len(clean_data) >= 5:  # Need at least 5 data points
                        # Pearson correlation
                        pearson_r, pearson_p = pearsonr(clean_data[metric], clean_data[return_window])
                        
                        # Spearman correlation
                        spearman_r, spearman_p = spearmanr(clean_data[metric], clean_data[return_window])
                        
                        correlation_results[metric][return_window] = {
                            'sample_size': len(clean_data),
                            'pearson_r': pearson_r,
                            'pearson_p': pearson_p,
                            'spearman_r': spearman_r,
                            'spearman_p': spearman_p,
                            'significant_at_05': min(pearson_p, spearman_p) < 0.05,
                            'significant_at_01': min(pearson_p, spearman_p) < 0.01
                        }
        
        return correlation_results
    
    def run_regression_analysis(self):
        """Run comprehensive regression analysis."""
        print("Running regression analysis...")
        
        try:
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import r2_score, mean_squared_error
        except ImportError:
            print("Scikit-learn not available for regression analysis")
            return {}
        
        # Prepare regression dataset
        if self.options_df.empty:
            print("No options data for regression")
            return {}
        
        # Merge with returns data
        if not hasattr(self, 'returns_df') or self.returns_df.empty:
            print("No returns data for regression")
            return {}
        
        merged_df = pd.merge(
            self.options_df,
            self.returns_df,
            left_on=['event_name', 'ticker'],
            right_on=['event_name', 'ticker'],
            how='inner'
        )
        
        if len(merged_df) < 10:
            print(f"Insufficient data for regression: {len(merged_df)} observations")
            return {}
        
        # Prepare features (X) and targets (y)
        feature_cols = ['total_volume', 'pcr_volume', 'anomaly_score', 'otm_volume_ratio', 'short_term_ratio']
        target_cols = ['return_1d', 'return_3d', 'return_5d']
        
        regression_results = {}
        
        for target in target_cols:
            if target not in merged_df.columns:
                continue
                
            # Prepare data
            feature_data = merged_df[feature_cols].fillna(0)
            target_data = merged_df[target].dropna()
            
            # Align features and targets
            common_idx = feature_data.index.intersection(target_data.index)
            X = feature_data.loc[common_idx]
            y = target_data.loc[common_idx]
            
            if len(X) < 10:
                continue
            
            regression_results[target] = {}
            
            # Test different models
            models = {
                'Linear': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=0.1),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            for model_name, model in models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                    
                    # Fit full model
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    
                    regression_results[target][model_name] = {
                        'cv_r2_mean': np.mean(cv_scores),
                        'cv_r2_std': np.std(cv_scores),
                        'full_r2': r2_score(y, y_pred),
                        'mse': mean_squared_error(y, y_pred),
                        'sample_size': len(X)
                    }
                except Exception as e:
                    print(f"Error in {model_name} for {target}: {e}")
                    continue
        
        return regression_results
    
    def run_robustness_checks(self):
        """Perform robustness checks and sensitivity analysis."""
        print("Running robustness checks...")
        
        robustness_results = {}
        
        # Bootstrap analysis
        if not self.options_df.empty and hasattr(self, 'returns_df') and not self.returns_df.empty:
            merged_df = pd.merge(
                self.options_df,
                self.returns_df,
                left_on=['event_name', 'ticker'],
                right_on=['event_name', 'ticker'],
                how='inner'
            )
            
            if len(merged_df) >= 10:
                # Bootstrap correlation analysis
                n_bootstrap = 1000
                correlations = []
                
                for _ in range(n_bootstrap):
                    # Sample with replacement
                    bootstrap_sample = merged_df.sample(n=len(merged_df), replace=True)
                    
                    if 'anomaly_score' in bootstrap_sample.columns and 'return_5d' in bootstrap_sample.columns:
                        clean_data = bootstrap_sample[['anomaly_score', 'return_5d']].dropna()
                        if len(clean_data) >= 5:
                            corr, _ = pearsonr(clean_data['anomaly_score'], clean_data['return_5d'])
                            correlations.append(corr)
                
                if correlations:
                    robustness_results['bootstrap_correlation'] = {
                        'mean': np.mean(correlations),
                        'std': np.std(correlations),
                        'ci_2.5': np.percentile(correlations, 2.5),
                        'ci_97.5': np.percentile(correlations, 97.5),
                        'n_bootstrap': len(correlations)
                    }
        
        return robustness_results
    
    def run_complete_analysis(self):
        """Run the complete Phase II statistical analysis."""
        print("="*80)
        print("PHASE II STATISTICAL ANALYSIS")
        print("Testing Core Hypotheses: Options Market Predictive Power")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Get earnings and returns data
        earnings_results = self.get_earnings_surprises()
        returns_results = self.get_post_announcement_returns()
        
        # Run correlation tests
        earnings_correlations = self.correlate_options_with_earnings()
        returns_correlations = self.correlate_options_with_returns()
        
        # Run regression analysis
        regression_results = self.run_regression_analysis()
        
        # Run robustness checks
        robustness_results = self.run_robustness_checks()
        
        # Compile final results
        final_results = {
            'data_summary': {
                'total_events': len(self.events_df),
                'options_events': len(self.options_df),
                'earnings_events': len(earnings_results) if not earnings_results.empty else 0,
                'returns_events': len(returns_results) if not returns_results.empty else 0
            },
            'earnings_correlations': earnings_correlations,
            'returns_correlations': returns_correlations,
            'regression_analysis': regression_results,
            'robustness_checks': robustness_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        output_file = self.results_dir / "phase2_statistical_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\nPhase II statistical analysis complete!")
        print(f"Results saved to: {output_file}")
        
        # Print summary
        self.print_summary(final_results)
        
        return final_results
    
    def print_summary(self, results):
        """Print analysis summary."""
        print("\n" + "="*60)
        print("PHASE II ANALYSIS SUMMARY")
        print("="*60)
        
        data_summary = results.get('data_summary', {})
        print(f"Dataset Coverage:")
        print(f"  Total Events: {data_summary.get('total_events', 0)}")
        print(f"  Options Events: {data_summary.get('options_events', 0)}")
        print(f"  Earnings Events: {data_summary.get('earnings_events', 0)}")
        print(f"  Returns Events: {data_summary.get('returns_events', 0)}")
        
        # Earnings correlations summary
        earnings_corr = results.get('earnings_correlations', {})
        if earnings_corr:
            print(f"\nEarnings Surprise Correlations:")
            significant_earnings = [k for k, v in earnings_corr.items() if v.get('significant_at_05', False)]
            print(f"  Significant correlations (p<0.05): {len(significant_earnings)}")
            for metric in significant_earnings:
                corr_data = earnings_corr[metric]
                print(f"    {metric}: r={corr_data.get('pearson_r', 0):.3f}, p={corr_data.get('pearson_p', 1):.3f}")
        
        # Returns correlations summary
        returns_corr = results.get('returns_correlations', {})
        if returns_corr:
            print(f"\nStock Returns Correlations:")
            total_significant = 0
            for metric, windows in returns_corr.items():
                for window, corr_data in windows.items():
                    if corr_data.get('significant_at_05', False):
                        total_significant += 1
            print(f"  Significant correlations (p<0.05): {total_significant}")
        
        # Regression summary
        regression = results.get('regression_analysis', {})
        if regression:
            print(f"\nRegression Analysis:")
            best_r2 = 0
            best_model = None
            for target, models in regression.items():
                for model_name, metrics in models.items():
                    r2 = metrics.get('cv_r2_mean', 0)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = f"{model_name} ({target})"
            print(f"  Best Model R²: {best_r2:.3f} ({best_model})")
        
        print("="*60)

def main():
    """Run Phase II statistical analysis."""
    analyzer = Phase2StatisticalAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()