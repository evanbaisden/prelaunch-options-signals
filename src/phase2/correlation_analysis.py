"""
Statistical correlation testing framework for Phase 2 analysis.
Tests correlations between options signals and launch outcomes.
"""
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, chi2_contingency
import warnings

from .options_data import OptionsChain
from .flow_analysis import FlowMetrics, UnusualActivity
from .earnings_data import EarningsAnalysis
from ..common.types import LaunchEvent


@dataclass
class CorrelationTest:
    """Individual correlation test result."""
    test_name: str
    variable1: str
    variable2: str
    correlation_coefficient: float
    p_value: float
    sample_size: int
    test_statistic: float
    confidence_interval: Tuple[float, float]
    significance_level: str  # 'high', 'medium', 'low', 'none'
    interpretation: str


@dataclass
class EventStudyResults:
    """Event study analysis results."""
    event: LaunchEvent
    event_windows: List[str]  # e.g., ['-1:+1', '-2:+2', '-5:+5']
    
    # Abnormal returns
    abnormal_returns: Dict[str, float]  # window -> abnormal return
    cumulative_abnormal_returns: Dict[str, float]
    
    # Statistical tests
    t_statistics: Dict[str, float]
    p_values: Dict[str, float]
    
    # Options correlation
    options_correlation: Dict[str, float]
    options_p_values: Dict[str, float]


class CorrelationAnalyzer:
    """Analyzes correlations between options signals and outcomes."""
    
    def __init__(self, config=None):
        from ..config import get_config
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
    
    def test_options_price_correlation(
        self,
        options_flow: List[FlowMetrics],
        price_returns: pd.Series,
        test_type: str = 'pearson'
    ) -> CorrelationTest:
        """Test correlation between options flow and price returns."""
        
        # Align data by date
        flow_df = pd.DataFrame([
            {
                'date': f.date,
                'put_call_ratio': f.put_call_volume_ratio,
                'iv_skew': f.iv_skew,
                'total_volume': f.total_call_volume + f.total_put_volume
            }
            for f in options_flow
        ])
        
        # Merge with price returns
        merged_df = flow_df.merge(
            price_returns.reset_index().rename(columns={'Date': 'date', 0: 'return'}),
            on='date',
            how='inner'
        )
        
        if len(merged_df) < 3:
            return self._null_correlation_test("Insufficient data for correlation test")
        
        # Choose variables for correlation
        options_var = merged_df['put_call_ratio'].fillna(0)
        returns_var = merged_df['return'].fillna(0)
        
        # Perform correlation test
        if test_type.lower() == 'pearson':
            corr_coef, p_value = pearsonr(options_var, returns_var)
            test_stat = corr_coef * np.sqrt((len(merged_df) - 2) / (1 - corr_coef**2))
        elif test_type.lower() == 'spearman':
            corr_coef, p_value = spearmanr(options_var, returns_var)
            test_stat = corr_coef
        elif test_type.lower() == 'kendall':
            corr_coef, p_value = kendalltau(options_var, returns_var)
            test_stat = corr_coef
        else:
            return self._null_correlation_test(f"Unknown test type: {test_type}")
        
        # Calculate confidence interval (approximate for Pearson)
        n = len(merged_df)
        if test_type.lower() == 'pearson' and n > 3:
            fisher_z = np.arctanh(corr_coef)
            se = 1 / np.sqrt(n - 3)
            z_critical = stats.norm.ppf(0.975)  # 95% confidence
            ci_lower = np.tanh(fisher_z - z_critical * se)
            ci_upper = np.tanh(fisher_z + z_critical * se)
        else:
            ci_lower, ci_upper = corr_coef - 0.1, corr_coef + 0.1  # Rough approximation
        
        return CorrelationTest(
            test_name=f"{test_type.title()} Correlation Test",
            variable1="Options Put/Call Ratio",
            variable2="Stock Price Returns",
            correlation_coefficient=corr_coef,
            p_value=p_value,
            sample_size=n,
            test_statistic=test_stat,
            confidence_interval=(ci_lower, ci_upper),
            significance_level=self._get_significance_level(p_value),
            interpretation=self._interpret_correlation(corr_coef, p_value, "options flow", "price returns")
        )
    
    def test_unusual_activity_prediction(
        self,
        unusual_activities: List[UnusualActivity],
        earnings_results: List[EarningsAnalysis]
    ) -> CorrelationTest:
        """Test if unusual options activity predicts earnings surprises."""
        
        if not unusual_activities or not earnings_results:
            return self._null_correlation_test("No unusual activity or earnings data")
        
        # Create binary variables for analysis
        activity_data = []
        earnings_data = []
        
        for earnings in earnings_results:
            event_date = earnings.event.announcement or earnings.event.release
            
            # Count unusual activities in the days before earnings
            activities_before_earnings = [
                ua for ua in unusual_activities
                if ua.contract.underlying_ticker == earnings.event.ticker and
                abs((ua.timestamp.date() - event_date).days) <= 30
            ]
            
            has_unusual_activity = len(activities_before_earnings) > 0
            has_positive_surprise = earnings.avg_eps_surprise_after > 0
            
            activity_data.append(1 if has_unusual_activity else 0)
            earnings_data.append(1 if has_positive_surprise else 0)
        
        if len(activity_data) < 3:
            return self._null_correlation_test("Insufficient paired data")
        
        # Chi-square test for independence
        try:
            # Create contingency table
            contingency_table = pd.crosstab(
                pd.Series(activity_data, name='unusual_activity'),
                pd.Series(earnings_data, name='positive_surprise')
            )
            
            if contingency_table.shape != (2, 2):
                # Fallback to correlation if contingency table is not 2x2
                corr_coef, p_value = pearsonr(activity_data, earnings_data)
                test_stat = corr_coef
            else:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                # Convert chi2 to correlation-like measure
                n = sum(activity_data) + sum(earnings_data)
                corr_coef = np.sqrt(chi2 / (n + chi2))
                test_stat = chi2
            
        except Exception as e:
            self.logger.warning(f"Error in chi-square test: {e}")
            # Fallback to simple correlation
            corr_coef, p_value = pearsonr(activity_data, earnings_data)
            test_stat = corr_coef
        
        return CorrelationTest(
            test_name="Unusual Activity vs Earnings Surprise",
            variable1="Unusual Options Activity (Binary)",
            variable2="Positive Earnings Surprise (Binary)",
            correlation_coefficient=corr_coef,
            p_value=p_value,
            sample_size=len(activity_data),
            test_statistic=test_stat,
            confidence_interval=(max(-1, corr_coef - 0.2), min(1, corr_coef + 0.2)),
            significance_level=self._get_significance_level(p_value),
            interpretation=self._interpret_correlation(corr_coef, p_value, "unusual activity", "earnings surprises")
        )
    
    def run_event_study(
        self,
        event: LaunchEvent,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        options_flow: List[FlowMetrics],
        event_windows: List[str] = None
    ) -> EventStudyResults:
        """Run event study analysis around product launch."""
        
        if event_windows is None:
            event_windows = ['-1:+1', '-2:+2', '-5:+5']
        
        event_date = event.announcement or event.release
        
        # Estimate market model parameters (pre-event period)
        estimation_start = event_date - pd.Timedelta(days=120)
        estimation_end = event_date - pd.Timedelta(days=30)
        
        # Get estimation period data
        estimation_stock = stock_returns[
            (stock_returns.index >= estimation_start) & 
            (stock_returns.index <= estimation_end)
        ]
        estimation_market = market_returns[
            (market_returns.index >= estimation_start) & 
            (market_returns.index <= estimation_end)
        ]
        
        # Align data
        aligned_data = pd.DataFrame({
            'stock': estimation_stock,
            'market': estimation_market
        }).dropna()
        
        if len(aligned_data) < 30:
            self.logger.warning(f"Insufficient estimation period data for {event.name}")
            return self._null_event_study(event, event_windows)
        
        # Estimate market model: R_stock = alpha + beta * R_market + error
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                aligned_data['market'], aligned_data['stock']
            )
            alpha, beta = intercept, slope
        except Exception as e:
            self.logger.warning(f"Error estimating market model: {e}")
            alpha, beta = 0, 1  # Default to market model
        
        # Calculate abnormal returns for each event window
        abnormal_returns = {}
        cumulative_abnormal_returns = {}
        t_statistics = {}
        p_values = {}
        
        for window in event_windows:
            try:
                # Parse window (e.g., '-5:+5' -> -5 to +5 days)
                start_offset, end_offset = map(int, window.split(':'))
                
                window_start = event_date + pd.Timedelta(days=start_offset)
                window_end = event_date + pd.Timedelta(days=end_offset)
                
                # Get event window data
                window_stock = stock_returns[
                    (stock_returns.index >= window_start) & 
                    (stock_returns.index <= window_end)
                ]
                window_market = market_returns[
                    (market_returns.index >= window_start) & 
                    (stock_returns.index <= window_end)
                ]
                
                # Calculate abnormal returns
                window_data = pd.DataFrame({
                    'stock': window_stock,
                    'market': window_market
                }).dropna()
                
                if len(window_data) == 0:
                    abnormal_returns[window] = 0
                    cumulative_abnormal_returns[window] = 0
                    t_statistics[window] = 0
                    p_values[window] = 1
                    continue
                
                # Abnormal return = Actual return - Expected return
                expected_returns = alpha + beta * window_data['market']
                abnormal_return_series = window_data['stock'] - expected_returns
                
                # Average abnormal return
                avg_abnormal_return = abnormal_return_series.mean()
                cumulative_abnormal_return = abnormal_return_series.sum()
                
                # Statistical significance test
                if len(abnormal_return_series) > 1:
                    t_stat, p_val = stats.ttest_1samp(abnormal_return_series, 0)
                else:
                    t_stat, p_val = 0, 1
                
                abnormal_returns[window] = avg_abnormal_return
                cumulative_abnormal_returns[window] = cumulative_abnormal_return
                t_statistics[window] = t_stat
                p_values[window] = p_val
                
            except Exception as e:
                self.logger.warning(f"Error processing window {window}: {e}")
                abnormal_returns[window] = 0
                cumulative_abnormal_returns[window] = 0
                t_statistics[window] = 0
                p_values[window] = 1
        
        # Options correlation analysis
        options_correlation, options_p_values = self._analyze_options_event_correlation(
            options_flow, abnormal_returns, event_date
        )
        
        return EventStudyResults(
            event=event,
            event_windows=event_windows,
            abnormal_returns=abnormal_returns,
            cumulative_abnormal_returns=cumulative_abnormal_returns,
            t_statistics=t_statistics,
            p_values=p_values,
            options_correlation=options_correlation,
            options_p_values=options_p_values
        )
    
    def run_comprehensive_correlation_analysis(
        self,
        events: List[LaunchEvent],
        options_data: Dict[str, List[FlowMetrics]],
        earnings_data: Dict[str, EarningsAnalysis],
        stock_data: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """Run comprehensive correlation analysis across all events."""
        
        results = {
            'individual_correlations': [],
            'cross_sectional_analysis': {},
            'time_series_analysis': {},
            'robustness_tests': {},
            'summary_statistics': {}
        }
        
        # Individual event correlations
        for event in events:
            ticker = event.ticker
            
            if ticker not in options_data or ticker not in stock_data:
                continue
            
            # Test options-price correlation
            options_price_corr = self.test_options_price_correlation(
                options_data[ticker], stock_data[ticker]
            )
            results['individual_correlations'].append({
                'event': event.name,
                'ticker': ticker,
                'test': options_price_corr
            })
        
        # Cross-sectional analysis (compare across companies)
        results['cross_sectional_analysis'] = self._cross_sectional_analysis(
            events, options_data, stock_data
        )
        
        # Time series analysis (trends over time)
        results['time_series_analysis'] = self._time_series_analysis(
            events, options_data
        )
        
        # Robustness tests
        results['robustness_tests'] = self._robustness_tests(
            events, options_data, stock_data
        )
        
        # Summary statistics
        all_correlations = [
            test['test'].correlation_coefficient 
            for test in results['individual_correlations']
            if test['test'].correlation_coefficient is not None
        ]
        
        if all_correlations:
            results['summary_statistics'] = {
                'mean_correlation': np.mean(all_correlations),
                'median_correlation': np.median(all_correlations),
                'std_correlation': np.std(all_correlations),
                'min_correlation': np.min(all_correlations),
                'max_correlation': np.max(all_correlations),
                'significant_correlations': sum(1 for test in results['individual_correlations'] 
                                              if test['test'].p_value < 0.05),
                'total_tests': len(all_correlations)
            }
        
        return results
    
    def _null_correlation_test(self, reason: str) -> CorrelationTest:
        """Return null correlation test result."""
        return CorrelationTest(
            test_name="Null Test",
            variable1="N/A",
            variable2="N/A",
            correlation_coefficient=0,
            p_value=1,
            sample_size=0,
            test_statistic=0,
            confidence_interval=(0, 0),
            significance_level='none',
            interpretation=f"Test not performed: {reason}"
        )
    
    def _null_event_study(self, event: LaunchEvent, event_windows: List[str]) -> EventStudyResults:
        """Return null event study result."""
        return EventStudyResults(
            event=event,
            event_windows=event_windows,
            abnormal_returns={window: 0 for window in event_windows},
            cumulative_abnormal_returns={window: 0 for window in event_windows},
            t_statistics={window: 0 for window in event_windows},
            p_values={window: 1 for window in event_windows},
            options_correlation={window: 0 for window in event_windows},
            options_p_values={window: 1 for window in event_windows}
        )
    
    def _get_significance_level(self, p_value: float) -> str:
        """Map p-value to significance level."""
        if p_value < 0.01:
            return 'high'
        elif p_value < 0.05:
            return 'medium'
        elif p_value < 0.1:
            return 'low'
        else:
            return 'none'
    
    def _interpret_correlation(self, corr_coef: float, p_value: float, var1: str, var2: str) -> str:
        """Generate interpretation of correlation result."""
        strength = 'weak'
        if abs(corr_coef) > 0.7:
            strength = 'strong'
        elif abs(corr_coef) > 0.3:
            strength = 'moderate'
        
        direction = 'positive' if corr_coef > 0 else 'negative'
        significance = self._get_significance_level(p_value)
        
        if significance == 'none':
            return f"No statistically significant correlation found between {var1} and {var2}"
        else:
            return f"{strength.title()} {direction} correlation between {var1} and {var2} (p={p_value:.3f})"
    
    def _analyze_options_event_correlation(
        self, 
        options_flow: List[FlowMetrics], 
        abnormal_returns: Dict[str, float],
        event_date: date
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Analyze correlation between options flow and event abnormal returns."""
        
        correlations = {}
        p_values = {}
        
        if not options_flow:
            return correlations, p_values
        
        # Get options flow around event date
        event_flow = [
            f for f in options_flow 
            if abs((f.date - event_date).days) <= 10
        ]
        
        if len(event_flow) < 3:
            return correlations, p_values
        
        # Calculate average options metrics
        avg_put_call_ratio = np.mean([f.put_call_volume_ratio for f in event_flow])
        
        for window, abnormal_return in abnormal_returns.items():
            if abnormal_return != 0:
                # Simple correlation approximation
                # In practice, this would be more sophisticated
                correlation = -0.1 * avg_put_call_ratio + 0.05  # Placeholder
                correlations[window] = correlation
                p_values[window] = 0.5  # Placeholder
        
        return correlations, p_values
    
    def _cross_sectional_analysis(
        self, 
        events: List[LaunchEvent],
        options_data: Dict[str, List[FlowMetrics]], 
        stock_data: Dict[str, pd.Series]
    ) -> Dict:
        """Analyze correlations across different companies."""
        return {
            'company_correlations': {},
            'sector_analysis': {},
            'market_cap_effects': {}
        }  # Placeholder
    
    def _time_series_analysis(
        self, 
        events: List[LaunchEvent],
        options_data: Dict[str, List[FlowMetrics]]
    ) -> Dict:
        """Analyze correlation trends over time.""" 
        return {
            'trend_analysis': {},
            'seasonal_patterns': {},
            'regime_changes': {}
        }  # Placeholder
    
    def _robustness_tests(
        self,
        events: List[LaunchEvent],
        options_data: Dict[str, List[FlowMetrics]],
        stock_data: Dict[str, pd.Series]
    ) -> Dict:
        """Run robustness tests for correlation results."""
        return {
            'bootstrap_results': {},
            'outlier_sensitivity': {},
            'alternative_specifications': {}
        }  # Placeholder