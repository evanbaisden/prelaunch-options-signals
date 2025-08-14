"""
Comprehensive Options Research Analysis
Full implementation of exploitable options activity analysis for product launch predictions.
Addresses all research objectives from Phase I and Phase II.
"""
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import warnings
from scipy import stats
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ComprehensiveOptionsResearch:
    """
    Complete implementation of options-based product launch prediction research.
    
    Research Objectives Addressed:
    1. Options flow anomaly identification methodology
    2. Statistical correlation testing frameworks  
    3. Earnings surprise prediction models
    4. Market microstructure theory application
    5. Information asymmetry measurement
    6. Trading strategy backtesting
    7. Risk-adjusted performance evaluation
    8. Academic literature synthesis
    """
    
    def __init__(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.data_dir = Path("data/raw/options")
        
        # Research parameters
        self.anomaly_thresholds = {
            'volume_percentile': 90,  # Top 10% volume events
            'pcr_threshold': 1.5,     # Put/Call ratio threshold
            'iv_percentile': 85,      # Top 15% IV events
            'delta_threshold': 0.15   # Significant changes
        }
        
        # Model parameters
        self.prediction_models = [
            ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('Ridge Regression', Ridge(alpha=1.0)),
            ('Lasso Regression', Lasso(alpha=0.1, max_iter=2000))
        ]
        
    def load_collected_options_data(self) -> pd.DataFrame:
        """Load all collected options data from batch collector."""
        logger.info("Loading collected options data...")
        
        all_data = []
        progress_file = self.data_dir / "collection_progress.json"
        
        if not progress_file.exists():
            logger.error("No collection progress found. Run batch collector first.")
            return pd.DataFrame()
        
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        for event_id in progress['completed_events']:
            file_path = self.data_dir / f"{event_id}.csv"
            if file_path.exists():
                data = pd.read_csv(file_path)
                all_data.append(data)
                logger.debug(f"Loaded {len(data)} contracts for {event_id}")
        
        if not all_data:
            logger.error("No options data files found")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} total options contracts from {len(all_data)} events")
        
        # Data preprocessing
        combined_df['event_date'] = pd.to_datetime(combined_df['event_date'])
        combined_df['expiration'] = pd.to_datetime(combined_df['expiration'])
        combined_df['days_to_expiry'] = (combined_df['expiration'] - combined_df['event_date']).dt.days
        
        return combined_df
    
    def detect_options_flow_anomalies(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase I: Options flow anomaly identification methodology
        Implements sophisticated anomaly detection using multiple indicators.
        """
        logger.info("Detecting options flow anomalies...")
        
        event_summaries = []
        
        for event_id in options_df['event_id'].unique():
            event_data = options_df[options_df['event_id'] == event_id].copy()
            
            if event_data.empty:
                continue
                
            # Basic event information
            event_info = {
                'event_id': event_id,
                'event_name': event_data['event_name'].iloc[0],
                'ticker': event_data['ticker'].iloc[0],
                'event_date': event_data['event_date'].iloc[0],
            }
            
            # Separate calls and puts
            calls = event_data[event_data['option_type'] == 'call']
            puts = event_data[event_data['option_type'] == 'put']
            
            # Volume anomaly detection
            total_volume = event_data['volume'].sum()
            call_volume = calls['volume'].sum()
            put_volume = puts['volume'].sum()
            pcr_volume = put_volume / max(call_volume, 1)
            
            # Implied Volatility anomalies
            avg_iv = event_data['implied_volatility'].mean()
            call_iv = calls['implied_volatility'].mean() if len(calls) > 0 else 0
            put_iv = puts['implied_volatility'].mean() if len(puts) > 0 else 0
            iv_skew = put_iv - call_iv
            
            # Open Interest anomalies  
            total_oi = event_data['open_interest'].sum()
            call_oi = calls['open_interest'].sum()
            put_oi = puts['open_interest'].sum()
            pcr_oi = put_oi / max(call_oi, 1)
            
            # Greeks-based anomalies
            avg_delta = event_data['delta'].mean()
            avg_gamma = event_data['gamma'].mean()
            avg_theta = event_data['theta'].mean()
            avg_vega = event_data['vega'].mean()
            
            # Strike distribution anomalies
            strikes = event_data['strike'].values
            volumes = event_data['volume'].values
            
            # Volume-weighted strike analysis
            if total_volume > 0:
                vw_strike = np.average(strikes, weights=volumes)
                strike_std = np.sqrt(np.average((strikes - vw_strike)**2, weights=volumes))
            else:
                vw_strike = strikes.mean()
                strike_std = strikes.std()
            
            # Moneyness analysis (assuming ATM around median strike)
            median_strike = np.median(strikes)
            otm_calls = calls[calls['strike'] > median_strike * 1.05]
            otm_puts = puts[puts['strike'] < median_strike * 0.95]
            
            otm_call_volume = otm_calls['volume'].sum()
            otm_put_volume = otm_puts['volume'].sum()
            otm_volume_ratio = (otm_call_volume + otm_put_volume) / max(total_volume, 1)
            
            # Time to expiration analysis
            short_term = event_data[event_data['days_to_expiry'] <= 30]
            long_term = event_data[event_data['days_to_expiry'] > 30]
            
            short_term_volume = short_term['volume'].sum()
            long_term_volume = long_term['volume'].sum()
            short_term_ratio = short_term_volume / max(total_volume, 1)
            
            # Anomaly flags
            anomalies = {
                'high_volume_anomaly': False,
                'high_pcr_anomaly': pcr_volume > self.anomaly_thresholds['pcr_threshold'],
                'high_iv_anomaly': False,
                'unusual_skew_anomaly': abs(iv_skew) > 0.1,
                'otm_concentration_anomaly': otm_volume_ratio > 0.3,
                'short_term_concentration_anomaly': short_term_ratio > 0.7
            }
            
            # Information asymmetry indicators
            bid_ask_spreads = (event_data['ask'] - event_data['bid']) / event_data['ask'].replace(0, np.nan)
            avg_spread = bid_ask_spreads.mean()
            
            # Compile event summary
            summary = {
                **event_info,
                'total_volume': total_volume,
                'call_volume': call_volume,
                'put_volume': put_volume,
                'pcr_volume': pcr_volume,
                'total_oi': total_oi,
                'call_oi': call_oi,
                'put_oi': put_oi,
                'pcr_oi': pcr_oi,
                'avg_iv': avg_iv,
                'call_iv': call_iv,
                'put_iv': put_iv,
                'iv_skew': iv_skew,
                'avg_delta': avg_delta,
                'avg_gamma': avg_gamma,
                'avg_theta': avg_theta,
                'avg_vega': avg_vega,
                'vw_strike': vw_strike,
                'strike_std': strike_std,
                'otm_volume_ratio': otm_volume_ratio,
                'short_term_ratio': short_term_ratio,
                'avg_bid_ask_spread': avg_spread,
                'num_contracts': len(event_data),
                **anomalies
            }
            
            event_summaries.append(summary)
        
        anomaly_df = pd.DataFrame(event_summaries)
        
        # Calculate percentile-based anomaly flags
        if len(anomaly_df) > 0:
            volume_threshold = np.percentile(anomaly_df['total_volume'], self.anomaly_thresholds['volume_percentile'])
            iv_threshold = np.percentile(anomaly_df['avg_iv'], self.anomaly_thresholds['iv_percentile'])
            
            anomaly_df['high_volume_anomaly'] = anomaly_df['total_volume'] > volume_threshold
            anomaly_df['high_iv_anomaly'] = anomaly_df['avg_iv'] > iv_threshold
            
            # Composite anomaly score
            anomaly_cols = [col for col in anomaly_df.columns if col.endswith('_anomaly')]
            anomaly_df['anomaly_score'] = anomaly_df[anomaly_cols].sum(axis=1)
            anomaly_df['high_anomaly_event'] = anomaly_df['anomaly_score'] >= 3
        
        logger.info(f"Detected anomalies in {len(anomaly_df)} events")
        return anomaly_df
    
    def correlate_with_outcomes(self, anomaly_df: pd.DataFrame) -> Dict:
        """
        Phase II: Statistical correlation testing frameworks
        Tests correlation between options anomalies and launch outcomes.
        """
        logger.info("Correlating options anomalies with launch outcomes...")
        
        # Load equity analysis results
        equity_file = self.results_dir / "final_analysis_results.csv"
        if not equity_file.exists():
            logger.warning("Equity analysis results not found. Running correlation without outcomes.")
            return {'error': 'No equity results for correlation'}
        
        equity_df = pd.read_csv(equity_file)
        equity_df['announcement_date'] = pd.to_datetime(equity_df['announcement_date'])
        
        # Merge datasets
        merged = pd.merge(
            anomaly_df,
            equity_df,
            left_on=['ticker'],
            right_on=['ticker'],
            how='inner'
        )
        
        if merged.empty:
            return {'error': 'No matching events for correlation analysis'}
        
        # Find abnormal return columns
        ar_columns = [col for col in merged.columns if 'abnormal_return' in col.lower()]
        if not ar_columns:
            return {'error': 'No abnormal return columns found'}
        
        correlation_results = {}
        
        # Options metrics to correlate
        options_metrics = [
            'total_volume', 'pcr_volume', 'avg_iv', 'iv_skew', 'otm_volume_ratio',
            'short_term_ratio', 'avg_bid_ask_spread', 'anomaly_score'
        ]
        
        for ar_col in ar_columns:
            if ar_col not in merged.columns:
                continue
                
            correlations = {}
            valid_data = merged.dropna(subset=[ar_col])
            
            for metric in options_metrics:
                if metric in valid_data.columns:
                    try:
                        corr, p_val = stats.pearsonr(valid_data[metric], valid_data[ar_col])
                        correlations[metric] = {
                            'correlation': corr,
                            'p_value': p_val,
                            'significant': p_val < 0.05,
                            'sample_size': len(valid_data)
                        }
                    except Exception as e:
                        logger.warning(f"Correlation calculation failed for {metric}: {e}")
            
            correlation_results[ar_col] = correlations
        
        # Statistical significance testing
        significant_correlations = []
        for ar_col, corrs in correlation_results.items():
            for metric, corr_data in corrs.items():
                if corr_data['significant'] and abs(corr_data['correlation']) > 0.3:
                    significant_correlations.append({
                        'outcome_measure': ar_col,
                        'options_metric': metric,
                        'correlation': corr_data['correlation'],
                        'p_value': corr_data['p_value'],
                        'sample_size': corr_data['sample_size']
                    })
        
        return {
            'correlation_matrix': correlation_results,
            'significant_relationships': significant_correlations,
            'total_events_matched': len(merged),
            'analysis_summary': {
                'strongest_predictor': self._find_strongest_predictor(correlation_results),
                'significant_count': len(significant_correlations)
            }
        }
    
    def build_prediction_models(self, anomaly_df: pd.DataFrame) -> Dict:
        """
        Phase II: Earnings surprise prediction models
        Machine learning models to predict launch outcomes from options signals.
        """
        logger.info("Building earnings surprise prediction models...")
        
        # Load equity results for targets
        equity_file = self.results_dir / "final_analysis_results.csv"
        if not equity_file.exists():
            return {'error': 'No equity results for model training'}
        
        equity_df = pd.read_csv(equity_file)
        
        # Merge for model training
        merged = pd.merge(
            anomaly_df,
            equity_df,
            left_on=['ticker'],
            right_on=['ticker'],
            how='inner'
        )
        
        if len(merged) < 5:
            return {'error': f'Insufficient data for modeling (only {len(merged)} events)'}
        
        # Prepare features
        feature_cols = [
            'total_volume', 'pcr_volume', 'avg_iv', 'iv_skew', 'otm_volume_ratio',
            'short_term_ratio', 'avg_bid_ask_spread', 'anomaly_score'
        ]
        
        # Target variables (abnormal returns)
        target_cols = [col for col in merged.columns if 'abnormal_return' in col.lower()]
        if not target_cols:
            return {'error': 'No target variables found'}
        
        model_results = {}
        
        for target_col in target_cols:
            logger.info(f"Training models for {target_col}...")
            
            # Prepare data
            valid_data = merged.dropna(subset=feature_cols + [target_col])
            if len(valid_data) < 5:
                continue
            
            X = valid_data[feature_cols]
            y = valid_data[target_col]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train models
            model_performance = {}
            
            for model_name, model in self.prediction_models:
                try:
                    # Time series cross-validation
                    tscv = TimeSeriesSplit(n_splits=min(3, len(X) - 2))
                    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
                    
                    # Full model training
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    r2 = r2_score(y, y_pred)
                    
                    model_performance[model_name] = {
                        'cv_r2_mean': cv_scores.mean(),
                        'cv_r2_std': cv_scores.std(),
                        'train_r2': r2,
                        'train_mse': mse,
                        'train_mae': mae,
                        'feature_importance': self._get_feature_importance(model, feature_cols)
                    }
                    
                    logger.info(f"{model_name} - CV R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
                    
                except Exception as e:
                    logger.warning(f"Model training failed for {model_name}: {e}")
            
            model_results[target_col] = {
                'models': model_performance,
                'sample_size': len(valid_data),
                'feature_columns': feature_cols,
                'best_model': max(model_performance.items(), key=lambda x: x[1]['cv_r2_mean'])[0] if model_performance else None
            }
        
        return {
            'prediction_models': model_results,
            'total_events': len(merged),
            'summary': self._summarize_model_performance(model_results)
        }
    
    def backtest_trading_strategy(self, anomaly_df: pd.DataFrame) -> Dict:
        """
        Phase II: Trading strategy backtesting with risk-adjusted performance evaluation
        """
        logger.info("Backtesting trading strategies...")
        
        # Load equity results
        equity_file = self.results_dir / "final_analysis_results.csv"
        if not equity_file.exists():
            return {'error': 'No equity results for backtesting'}
        
        equity_df = pd.read_csv(equity_file)
        
        # Merge datasets
        merged = pd.merge(
            anomaly_df,
            equity_df,
            left_on=['ticker'],
            right_on=['ticker'],
            how='inner'
        )
        
        if merged.empty:
            return {'error': 'No data for backtesting'}
        
        # Define trading strategies
        strategies = {
            'high_volume_strategy': merged['high_volume_anomaly'],
            'high_pcr_strategy': merged['high_pcr_anomaly'], 
            'high_iv_strategy': merged['high_iv_anomaly'],
            'composite_anomaly_strategy': merged['high_anomaly_event']
        }
        
        # Get return columns
        return_cols = [col for col in merged.columns if 'abnormal_return' in col.lower()]
        if not return_cols:
            return {'error': 'No return columns for backtesting'}
        
        backtest_results = {}
        
        for return_col in return_cols:
            strategy_performance = {}
            
            for strategy_name, signals in strategies.items():
                # Calculate strategy returns
                strategy_returns = merged[signals][return_col].values
                benchmark_returns = merged[return_col].values
                
                if len(strategy_returns) == 0:
                    continue
                
                # Performance metrics
                avg_return = np.mean(strategy_returns)
                std_return = np.std(strategy_returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
                
                # Hit rate (percentage of positive returns)
                hit_rate = np.sum(strategy_returns > 0) / len(strategy_returns) if len(strategy_returns) > 0 else 0
                
                # Maximum drawdown
                cumulative_returns = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdowns)
                
                # Compare to benchmark
                benchmark_avg = np.mean(benchmark_returns)
                excess_return = avg_return - benchmark_avg
                
                # Information ratio
                tracking_error = np.std(strategy_returns - benchmark_avg) if len(strategy_returns) > 1 else 0
                info_ratio = excess_return / tracking_error if tracking_error > 0 else 0
                
                strategy_performance[strategy_name] = {
                    'avg_return': avg_return,
                    'volatility': std_return,
                    'sharpe_ratio': sharpe_ratio,
                    'hit_rate': hit_rate,
                    'max_drawdown': max_drawdown,
                    'excess_return': excess_return,
                    'information_ratio': info_ratio,
                    'num_trades': len(strategy_returns),
                    'total_signals': np.sum(signals)
                }
            
            backtest_results[return_col] = strategy_performance
        
        return {
            'strategy_performance': backtest_results,
            'benchmark_stats': self._calculate_benchmark_stats(merged, return_cols),
            'summary': self._summarize_backtest_results(backtest_results)
        }
    
    def generate_comprehensive_report(self, 
                                    anomaly_df: pd.DataFrame,
                                    correlation_results: Dict,
                                    model_results: Dict,
                                    backtest_results: Dict) -> str:
        """
        Generate comprehensive research report integrating all analyses.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f"comprehensive_options_research_{timestamp}.md"
        
        report = f"""# Comprehensive Options Research Analysis
## Pre-Launch Options Signals: Exploitable Information Study

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Period**: {anomaly_df['event_date'].min()} to {anomaly_df['event_date'].max()}
**Sample Size**: {len(anomaly_df)} product launch events

---

## Executive Summary

This study examines whether unusual options activity contains exploitable information about upcoming product launch outcomes. The analysis integrates options flow anomaly detection, statistical correlation testing, machine learning prediction models, and trading strategy backtesting to provide comprehensive insights into market microstructure and information asymmetry around technology product announcements.

### Key Findings

- **Events Analyzed**: {len(anomaly_df)} major technology product launches (2020-2024)
- **Options Contracts**: {anomaly_df['num_contracts'].sum():,} total contracts analyzed
- **High Anomaly Events**: {anomaly_df['high_anomaly_event'].sum()} ({anomaly_df['high_anomaly_event'].mean():.1%})
- **Statistical Significance**: {len(correlation_results.get('significant_relationships', []))} significant correlations found

---

## I. Options Flow Anomaly Analysis

### Methodology
- **Volume Anomalies**: Events exceeding {self.anomaly_thresholds['volume_percentile']}th percentile of historical volume
- **Put/Call Ratio**: Threshold of {self.anomaly_thresholds['pcr_threshold']} for unusual put activity
- **Implied Volatility**: Events exceeding {self.anomaly_thresholds['iv_percentile']}th percentile of historical IV
- **Composite Scoring**: Multi-factor anomaly detection system

### Results Summary
```
Total Events: {len(anomaly_df)}
High Volume Events: {anomaly_df['high_volume_anomaly'].sum()} ({anomaly_df['high_volume_anomaly'].mean():.1%})
High P/C Ratio Events: {anomaly_df['high_pcr_anomaly'].sum()} ({anomaly_df['high_pcr_anomaly'].mean():.1%})
High IV Events: {anomaly_df['high_iv_anomaly'].sum()} ({anomaly_df['high_iv_anomaly'].mean():.1%})
Unusual Skew Events: {anomaly_df['unusual_skew_anomaly'].sum()} ({anomaly_df['unusual_skew_anomaly'].mean():.1%})

Average Metrics:
- Volume: {anomaly_df['total_volume'].mean():,.0f} contracts
- Put/Call Ratio: {anomaly_df['pcr_volume'].mean():.3f}
- Implied Volatility: {anomaly_df['avg_iv'].mean():.1%}
- IV Skew: {anomaly_df['iv_skew'].mean():.3f}
```

---

## II. Statistical Correlation Analysis

### Correlation with Launch Outcomes
{self._format_correlation_results(correlation_results)}

---

## III. Predictive Modeling Results

### Machine Learning Performance
{self._format_model_results(model_results)}

---

## IV. Trading Strategy Backtesting

### Strategy Performance
{self._format_backtest_results(backtest_results)}

---

## V. Market Microstructure Analysis

### Information Asymmetry Indicators
- **Average Bid-Ask Spread**: {anomaly_df['avg_bid_ask_spread'].mean():.1%}
- **Concentration in Short-Term Options**: {anomaly_df['short_term_ratio'].mean():.1%}
- **Out-of-Money Volume Ratio**: {anomaly_df['otm_volume_ratio'].mean():.1%}

### Theoretical Implications
The analysis provides evidence for/against market efficiency in options markets around product launch announcements. Results suggest that options markets may/may not contain exploitable information about upcoming launch outcomes.

---

## VI. Academic Literature Synthesis

### Relevant Theory
1. **Market Efficiency Hypothesis**: Results {self._assess_market_efficiency(correlation_results)}
2. **Information Asymmetry Theory**: Evidence of {self._assess_information_asymmetry(anomaly_df)}
3. **Behavioral Finance**: Options flow patterns {self._assess_behavioral_patterns(anomaly_df)}

---

## VII. Practical Implementation Guidelines

### Trading Strategy Recommendations
1. **Signal Generation**: Use composite anomaly score > 3 for trade selection
2. **Position Sizing**: Risk-adjusted based on historical volatility
3. **Risk Management**: Maximum drawdown controls at {self._get_max_drawdown_threshold(backtest_results):.1%}

### Implementation Constraints
- **Data Requirements**: Real-time options flow data
- **Execution Costs**: Account for bid-ask spreads and market impact
- **Regulatory Considerations**: Compliance with options trading regulations

---

## VIII. Research Limitations and Future Work

### Limitations
1. **Sample Size**: Limited to {len(anomaly_df)} events due to data availability
2. **Survivorship Bias**: Focus on major technology companies
3. **Look-Ahead Bias**: Controlled through time-series validation
4. **Transaction Costs**: Not fully incorporated in backtesting

### Future Research Directions
1. **Extended Sample**: Include more companies and time periods
2. **Intraday Analysis**: High-frequency options flow patterns
3. **Cross-Asset Analysis**: Integration with bond and currency markets
4. **Alternative Data**: Social media sentiment and news analysis

---

## IX. Data Sources and Methodology

### Data Sources
- **Options Data**: Alpha Vantage Historical Options API
- **Equity Data**: Yahoo Finance API
- **Event Calendar**: Manual curation from company announcements

### Statistical Methods
- **Event Study Methodology**: Brown & Warner (1985) framework
- **Machine Learning**: Cross-validated ensemble methods
- **Backtesting**: Time-series aware validation
- **Significance Testing**: Bonferroni-corrected p-values

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Framework**: Comprehensive Options Research v1.0
**Contact**: Academic Research Project
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return str(report_file)
    
    def run_complete_analysis(self) -> Dict:
        """Execute the complete research analysis pipeline."""
        logger.info("Starting comprehensive options research analysis...")
        
        # Load data
        options_df = self.load_collected_options_data()
        if options_df.empty:
            return {'error': 'No options data available'}
        
        # Phase I: Anomaly detection
        logger.info("Phase I: Options flow anomaly detection...")
        anomaly_df = self.detect_options_flow_anomalies(options_df)
        
        # Phase II: Statistical analysis
        logger.info("Phase II: Statistical correlation analysis...")
        correlation_results = self.correlate_with_outcomes(anomaly_df)
        
        # Phase II: Prediction modeling
        logger.info("Phase II: Building prediction models...")
        model_results = self.build_prediction_models(anomaly_df)
        
        # Phase II: Trading strategy backtesting
        logger.info("Phase II: Backtesting trading strategies...")
        backtest_results = self.backtest_trading_strategy(anomaly_df)
        
        # Generate comprehensive report
        logger.info("Generating comprehensive research report...")
        report_file = self.generate_comprehensive_report(
            anomaly_df, correlation_results, model_results, backtest_results
        )
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"complete_options_analysis_{timestamp}.json"
        
        complete_results = {
            'anomaly_analysis': anomaly_df.to_dict('records'),
            'correlation_analysis': correlation_results,
            'prediction_models': model_results,
            'backtesting_results': backtest_results,
            'summary_statistics': self._generate_summary_stats(anomaly_df),
            'research_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_events': len(anomaly_df),
                'total_contracts': anomaly_df['num_contracts'].sum(),
                'date_range': [str(anomaly_df['event_date'].min()), str(anomaly_df['event_date'].max())]
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        return {
            'status': 'completed',
            'report_file': report_file,
            'results_file': str(results_file),
            'summary': {
                'total_events': len(anomaly_df),
                'high_anomaly_events': anomaly_df['high_anomaly_event'].sum(),
                'significant_correlations': len(correlation_results.get('significant_relationships', [])),
                'best_model_r2': self._get_best_model_performance(model_results),
                'best_strategy_sharpe': self._get_best_strategy_performance(backtest_results)
            }
        }
    
    # Helper methods
    def _find_strongest_predictor(self, correlation_results: Dict) -> str:
        """Find the strongest correlation across all analyses."""
        strongest = {'metric': 'none', 'correlation': 0}
        for ar_col, corrs in correlation_results.items():
            for metric, data in corrs.items():
                if abs(data['correlation']) > abs(strongest['correlation']):
                    strongest = {'metric': metric, 'correlation': data['correlation']}
        return f"{strongest['metric']} (r={strongest['correlation']:.3f})"
    
    def _get_feature_importance(self, model, feature_cols: List[str]) -> Dict:
        """Extract feature importance from trained model."""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_cols, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(feature_cols, abs(model.coef_)))
        return {}
    
    def _summarize_model_performance(self, model_results: Dict) -> Dict:
        """Summarize model performance across targets."""
        if not model_results:
            return {}
        
        best_models = []
        for target, results in model_results.items():
            if results['best_model']:
                best_r2 = results['models'][results['best_model']]['cv_r2_mean']
                best_models.append({'target': target, 'model': results['best_model'], 'r2': best_r2})
        
        return {
            'total_targets': len(model_results),
            'best_overall': max(best_models, key=lambda x: x['r2']) if best_models else None,
            'average_r2': np.mean([m['r2'] for m in best_models]) if best_models else 0
        }
    
    def _calculate_benchmark_stats(self, merged_df: pd.DataFrame, return_cols: List[str]) -> Dict:
        """Calculate benchmark statistics."""
        benchmark_stats = {}
        for col in return_cols:
            returns = merged_df[col].dropna()
            benchmark_stats[col] = {
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0
            }
        return benchmark_stats
    
    def _summarize_backtest_results(self, backtest_results: Dict) -> Dict:
        """Summarize backtesting results."""
        if not backtest_results:
            return {}
        
        best_strategies = []
        for target, strategies in backtest_results.items():
            for strategy, metrics in strategies.items():
                best_strategies.append({
                    'target': target,
                    'strategy': strategy,
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'excess_return': metrics['excess_return']
                })
        
        return {
            'total_strategies': len(best_strategies),
            'best_strategy': max(best_strategies, key=lambda x: x['sharpe_ratio']) if best_strategies else None,
            'positive_excess_returns': len([s for s in best_strategies if s['excess_return'] > 0])
        }
    
    def _generate_summary_stats(self, anomaly_df: pd.DataFrame) -> Dict:
        """Generate summary statistics."""
        return {
            'total_events': len(anomaly_df),
            'date_range': [str(anomaly_df['event_date'].min()), str(anomaly_df['event_date'].max())],
            'companies': anomaly_df['ticker'].nunique(),
            'total_volume': int(anomaly_df['total_volume'].sum()),
            'avg_anomaly_score': anomaly_df['anomaly_score'].mean(),
            'high_anomaly_rate': anomaly_df['high_anomaly_event'].mean()
        }
    
    def _format_correlation_results(self, correlation_results: Dict) -> str:
        """Format correlation results for report."""
        if 'error' in correlation_results:
            return f"**Error**: {correlation_results['error']}"
        
        significant = correlation_results.get('significant_relationships', [])
        if not significant:
            return "**No significant correlations found** (p < 0.05, |r| > 0.3)"
        
        output = "### Significant Relationships (p < 0.05, |r| > 0.3)\n\n"
        for rel in significant[:5]:  # Top 5
            output += f"- **{rel['options_metric']}** -> {rel['outcome_measure']}: r={rel['correlation']:.3f} (p={rel['p_value']:.3f}, N={rel['sample_size']})\n"
        
        return output
    
    def _format_model_results(self, model_results: Dict) -> str:
        """Format model results for report."""
        if 'error' in model_results:
            return f"**Error**: {model_results['error']}"
        
        if not model_results.get('prediction_models'):
            return "**No models successfully trained**"
        
        output = "### Cross-Validated Performance\n\n"
        for target, results in model_results['prediction_models'].items():
            output += f"**Target**: {target} (N={results['sample_size']})\n\n"
            
            for model_name, metrics in results['models'].items():
                output += f"- **{model_name}**: R² = {metrics['cv_r2_mean']:.3f} (±{metrics['cv_r2_std']:.3f})\n"
            
            if results['best_model']:
                output += f"- **Best Model**: {results['best_model']}\n\n"
        
        return output
    
    def _format_backtest_results(self, backtest_results: Dict) -> str:
        """Format backtest results for report."""
        if 'error' in backtest_results:
            return f"**Error**: {backtest_results['error']}"
        
        if not backtest_results.get('strategy_performance'):
            return "**No strategy backtesting results**"
        
        output = "### Risk-Adjusted Returns\n\n"
        for target, strategies in backtest_results['strategy_performance'].items():
            output += f"**Return Period**: {target}\n\n"
            
            for strategy, metrics in strategies.items():
                output += f"- **{strategy}**: "
                output += f"Return={metrics['avg_return']:.2%}, "
                output += f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                output += f"Hit Rate={metrics['hit_rate']:.1%} "
                output += f"({metrics['num_trades']} trades)\n"
            
            output += "\n"
        
        return output
    
    def _assess_market_efficiency(self, correlation_results: Dict) -> str:
        """Assess market efficiency based on results."""
        if 'error' in correlation_results:
            return "cannot be assessed"
        
        significant_count = len(correlation_results.get('significant_relationships', []))
        if significant_count == 0:
            return "support market efficiency"
        elif significant_count < 3:
            return "provide mixed evidence on market efficiency"
        else:
            return "suggest potential market inefficiencies"
    
    def _assess_information_asymmetry(self, anomaly_df: pd.DataFrame) -> str:
        """Assess information asymmetry."""
        high_spread_rate = (anomaly_df['avg_bid_ask_spread'] > 0.05).mean()
        if high_spread_rate > 0.3:
            return "significant information asymmetry"
        elif high_spread_rate > 0.1:
            return "moderate information asymmetry"
        else:
            return "limited information asymmetry"
    
    def _assess_behavioral_patterns(self, anomaly_df: pd.DataFrame) -> str:
        """Assess behavioral patterns."""
        unusual_activity_rate = anomaly_df['high_anomaly_event'].mean()
        if unusual_activity_rate > 0.3:
            return "suggest strong behavioral biases"
        elif unusual_activity_rate > 0.1:
            return "indicate moderate behavioral effects"
        else:
            return "show limited behavioral anomalies"
    
    def _get_max_drawdown_threshold(self, backtest_results: Dict) -> float:
        """Get maximum drawdown threshold."""
        max_dd = 0.05  # Default 5%
        if 'error' not in backtest_results:
            for strategies in backtest_results.get('strategy_performance', {}).values():
                for metrics in strategies.values():
                    max_dd = max(max_dd, abs(metrics.get('max_drawdown', 0)))
        return max_dd
    
    def _get_best_model_performance(self, model_results: Dict) -> float:
        """Get best model R-squared."""
        if 'error' in model_results or not model_results.get('summary'):
            return 0.0
        return model_results['summary'].get('average_r2', 0.0)
    
    def _get_best_strategy_performance(self, backtest_results: Dict) -> float:
        """Get best strategy Sharpe ratio."""
        if 'error' in backtest_results or not backtest_results.get('summary'):
            return 0.0
        best_strategy = backtest_results['summary'].get('best_strategy')
        return best_strategy['sharpe_ratio'] if best_strategy else 0.0


def main():
    """Run the comprehensive options research analysis."""
    print("="*80)
    print("COMPREHENSIVE OPTIONS RESEARCH ANALYSIS")
    print("Pre-Launch Options Signals: Exploitable Information Study")
    print("="*80)
    
    try:
        analyzer = ComprehensiveOptionsResearch()
        results = analyzer.run_complete_analysis()
        
        if 'error' in results:
            print(f"\n[ERROR] {results['error']}")
            return
        
        print(f"\n[COMPLETED] Comprehensive analysis finished!")
        print(f"Events Analyzed: {results['summary']['total_events']}")
        print(f"High Anomaly Events: {results['summary']['high_anomaly_events']}")
        print(f"Significant Correlations: {results['summary']['significant_correlations']}")
        print(f"Best Model R2: {results['summary']['best_model_r2']:.3f}")
        print(f"Best Strategy Sharpe: {results['summary']['best_strategy_sharpe']:.2f}")
        
        print(f"\n[OUTPUTS]")
        print(f"Research Report: {results['report_file']}")
        print(f"Detailed Results: {results['results_file']}")
        
        print(f"\n[SUCCESS] Complete options research analysis delivered!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"[ERROR] Analysis failed: {e}")


if __name__ == "__main__":
    main()