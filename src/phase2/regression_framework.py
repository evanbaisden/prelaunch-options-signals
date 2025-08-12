"""
Regression analysis framework for Phase 2 statistical modeling.
Implements various regression models for signal prediction and hypothesis testing.
"""
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import warnings

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_white, jarque_bera
    from statsmodels.stats.stattools import durbin_watson
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from .options_data import OptionsChain
from .flow_analysis import FlowMetrics
from .earnings_data import EarningsAnalysis
from ..common.types import LaunchEvent


@dataclass
class RegressionResults:
    """Regression model results container."""
    model_name: str
    model_type: str  # 'linear', 'logistic', 'random_forest', etc.
    
    # Model fit statistics
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    f_pvalue: float
    
    # Coefficients and significance
    coefficients: Dict[str, float]
    coefficient_pvalues: Dict[str, float]
    coefficient_std_errors: Dict[str, float]
    
    # Predictions and residuals
    fitted_values: np.ndarray
    residuals: np.ndarray
    predictions: np.ndarray
    
    # Diagnostic tests
    durbin_watson_stat: Optional[float] = None
    jarque_bera_stat: Optional[float] = None
    white_test_stat: Optional[float] = None
    
    # Cross-validation results
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Feature importance (for tree-based models)
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class FeatureSet:
    """Feature engineering results."""
    features: pd.DataFrame
    target: pd.Series
    feature_names: List[str]
    target_name: str
    feature_descriptions: Dict[str, str]


class FeatureEngineer:
    """Creates features from options and stock data for regression analysis."""
    
    def __init__(self, config=None):
        from ..config import get_config
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
    
    def create_options_features(
        self, 
        options_flow: List[FlowMetrics],
        stock_returns: pd.Series,
        earnings_data: Optional[EarningsAnalysis] = None,
        lookback_days: int = 30
    ) -> FeatureSet:
        """Create comprehensive feature set from options and stock data."""
        
        # Convert options flow to DataFrame
        flow_df = pd.DataFrame([
            {
                'date': f.date,
                'ticker': f.ticker,
                'put_call_volume_ratio': f.put_call_volume_ratio,
                'put_call_oi_ratio': f.put_call_oi_ratio,
                'total_volume': f.total_call_volume + f.total_put_volume,
                'iv_skew': f.iv_skew,
                'average_iv': (f.average_call_iv + f.average_put_iv) / 2,
                'volume_weighted_strike': f.volume_weighted_strike,
                'max_pain_strike': f.max_pain_strike,
                'net_flow_direction': f.net_flow_direction
            }
            for f in options_flow
        ])
        
        if flow_df.empty:
            return self._empty_feature_set()
        
        # Merge with stock returns
        returns_df = stock_returns.reset_index()
        returns_df.columns = ['date', 'stock_return']
        
        merged_df = flow_df.merge(returns_df, on='date', how='inner')
        
        if len(merged_df) < 10:  # Need minimum data
            return self._empty_feature_set()
        
        # Sort by date for time series features
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        # Create lagged and technical features
        features_df = self._create_technical_features(merged_df, lookback_days)
        
        # Add earnings features if available
        if earnings_data:
            features_df = self._add_earnings_features(features_df, earnings_data)
        
        # Create target variable (forward-looking returns)
        target_series = self._create_target_variable(merged_df)
        
        # Align features and target
        min_length = min(len(features_df), len(target_series))
        features_df = features_df.iloc[:min_length]
        target_series = target_series.iloc[:min_length]
        
        feature_names = list(features_df.columns)
        
        return FeatureSet(
            features=features_df,
            target=target_series,
            feature_names=feature_names,
            target_name='forward_return_5d',
            feature_descriptions=self._get_feature_descriptions(feature_names)
        )
    
    def _create_technical_features(self, df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
        """Create technical and lagged features."""
        
        features_df = df.copy()
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features_df[f'put_call_ratio_lag{lag}'] = features_df['put_call_volume_ratio'].shift(lag)
            features_df[f'iv_skew_lag{lag}'] = features_df['iv_skew'].shift(lag)
            features_df[f'stock_return_lag{lag}'] = features_df['stock_return'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features_df[f'put_call_ratio_ma{window}'] = features_df['put_call_volume_ratio'].rolling(window).mean()
            features_df[f'put_call_ratio_std{window}'] = features_df['put_call_volume_ratio'].rolling(window).std()
            features_df[f'iv_skew_ma{window}'] = features_df['iv_skew'].rolling(window).mean()
            features_df[f'volume_ma{window}'] = features_df['total_volume'].rolling(window).mean()
        
        # Momentum features
        features_df['put_call_momentum_5d'] = (
            features_df['put_call_volume_ratio'] - features_df['put_call_volume_ratio'].shift(5)
        )
        features_df['iv_skew_momentum_5d'] = (
            features_df['iv_skew'] - features_df['iv_skew'].shift(5)
        )
        
        # Volatility features
        features_df['stock_return_volatility_10d'] = features_df['stock_return'].rolling(10).std()
        features_df['stock_return_volatility_20d'] = features_df['stock_return'].rolling(20).std()
        
        # Extreme values indicators
        features_df['extreme_put_call_ratio'] = (
            (features_df['put_call_volume_ratio'] > 2.0) | 
            (features_df['put_call_volume_ratio'] < 0.5)
        ).astype(int)
        
        features_df['extreme_iv_skew'] = (
            abs(features_df['iv_skew']) > 0.1
        ).astype(int)
        
        # Interaction terms
        features_df['put_call_x_iv_skew'] = (
            features_df['put_call_volume_ratio'] * features_df['iv_skew']
        )
        
        # Binary flow direction
        flow_direction_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
        features_df['flow_direction_numeric'] = features_df['net_flow_direction'].map(flow_direction_map)
        
        # Remove non-numeric columns
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_columns]
        
        # Drop date and target-like columns
        columns_to_drop = ['date', 'stock_return'] if 'date' in features_df.columns else ['stock_return']
        features_df = features_df.drop(columns=[col for col in columns_to_drop if col in features_df.columns])
        
        return features_df
    
    def _add_earnings_features(self, features_df: pd.DataFrame, earnings_data: EarningsAnalysis) -> pd.DataFrame:
        """Add earnings-related features."""
        
        # Add earnings metrics as static features
        features_df['avg_eps_surprise_before'] = earnings_data.avg_eps_surprise_before
        features_df['avg_eps_surprise_after'] = earnings_data.avg_eps_surprise_after
        features_df['earnings_improvement'] = int(earnings_data.surprise_improvement)
        features_df['earnings_significance'] = earnings_data.surprise_significance
        
        return features_df
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """Create forward-looking target variable."""
        
        # 5-day forward return
        target = df['stock_return'].rolling(5).sum().shift(-5)
        
        return target.fillna(0)
    
    def _get_feature_descriptions(self, feature_names: List[str]) -> Dict[str, str]:
        """Get human-readable descriptions of features."""
        
        descriptions = {
            'put_call_volume_ratio': 'Put/Call volume ratio',
            'put_call_oi_ratio': 'Put/Call open interest ratio',
            'total_volume': 'Total options volume',
            'iv_skew': 'Implied volatility skew',
            'average_iv': 'Average implied volatility',
            'volume_weighted_strike': 'Volume-weighted average strike',
            'max_pain_strike': 'Max pain strike level',
            'flow_direction_numeric': 'Options flow direction (-1=bearish, 0=neutral, 1=bullish)',
        }
        
        # Add descriptions for engineered features
        for name in feature_names:
            if name not in descriptions:
                if 'lag' in name:
                    descriptions[name] = f"Lagged version of {name.split('_lag')[0]}"
                elif 'ma' in name:
                    descriptions[name] = f"Moving average of {name.split('_ma')[0]}"
                elif 'std' in name:
                    descriptions[name] = f"Rolling standard deviation of {name.split('_std')[0]}"
                elif 'momentum' in name:
                    descriptions[name] = f"Momentum indicator for {name.split('_momentum')[0]}"
                elif 'volatility' in name:
                    descriptions[name] = f"Volatility measure for {name.split('_volatility')[0]}"
                else:
                    descriptions[name] = name.replace('_', ' ').title()
        
        return descriptions
    
    def _empty_feature_set(self) -> FeatureSet:
        """Return empty feature set."""
        return FeatureSet(
            features=pd.DataFrame(),
            target=pd.Series(dtype=float),
            feature_names=[],
            target_name='',
            feature_descriptions={}
        )


class RegressionAnalyzer:
    """Performs regression analysis for options signals prediction."""
    
    def __init__(self, config=None):
        from ..config import get_config
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available. Some regression methods will not work.")
        
        if not STATSMODELS_AVAILABLE:
            self.logger.warning("statsmodels not available. Some diagnostic tests will not work.")
    
    def fit_linear_regression(
        self, 
        feature_set: FeatureSet,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> RegressionResults:
        """Fit linear regression model with diagnostics."""
        
        if feature_set.features.empty:
            return self._empty_regression_results("No features available")
        
        # Remove rows with NaN values
        clean_data = pd.concat([feature_set.features, feature_set.target], axis=1).dropna()
        
        if len(clean_data) < 10:
            return self._empty_regression_results("Insufficient clean data")
        
        X = clean_data.iloc[:, :-1]
        y = clean_data.iloc[:, -1]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.seed
        )
        
        results = RegressionResults(
            model_name="Linear Regression",
            model_type="linear",
            r_squared=0,
            adjusted_r_squared=0,
            f_statistic=0,
            f_pvalue=1,
            coefficients={},
            coefficient_pvalues={},
            coefficient_std_errors={},
            fitted_values=np.array([]),
            residuals=np.array([]),
            predictions=np.array([])
        )
        
        try:
            if SKLEARN_AVAILABLE:
                # Scikit-learn model for basic fitting
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                results.fitted_values = y_pred_train
                results.predictions = y_pred_test
                results.residuals = y_train - y_pred_train
                results.r_squared = r2_score(y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=cv_folds)
                results.cv_scores = cv_scores.tolist()
                results.cv_mean = cv_scores.mean()
                results.cv_std = cv_scores.std()
                
                # Basic coefficients
                coef_dict = dict(zip(feature_set.feature_names, model.coef_))
                results.coefficients = coef_dict
            
            if STATSMODELS_AVAILABLE:
                # Statsmodels for detailed statistics
                X_sm = sm.add_constant(X_train)
                model_sm = sm.OLS(y_train, X_sm).fit()
                
                # Detailed statistics
                results.r_squared = model_sm.rsquared
                results.adjusted_r_squared = model_sm.rsquared_adj
                results.f_statistic = model_sm.fvalue
                results.f_pvalue = model_sm.f_pvalue
                
                # Coefficients with statistics
                coef_names = ['const'] + feature_set.feature_names
                results.coefficients = dict(zip(coef_names, model_sm.params))
                results.coefficient_pvalues = dict(zip(coef_names, model_sm.pvalues))
                results.coefficient_std_errors = dict(zip(coef_names, model_sm.bse))
                
                # Diagnostic tests
                results.durbin_watson_stat = durbin_watson(model_sm.resid)
                
                try:
                    jb_stat, jb_pvalue = jarque_bera(model_sm.resid)
                    results.jarque_bera_stat = jb_stat
                except:
                    pass
                
                try:
                    white_stat, white_pvalue, _, _ = het_white(model_sm.resid, model_sm.model.exog)
                    results.white_test_stat = white_stat
                except:
                    pass
            
        except Exception as e:
            self.logger.error(f"Error fitting linear regression: {e}")
        
        return results
    
    def fit_logistic_regression(
        self, 
        feature_set: FeatureSet,
        threshold: float = 0.0,
        test_size: float = 0.2
    ) -> RegressionResults:
        """Fit logistic regression for binary classification."""
        
        if not SKLEARN_AVAILABLE:
            return self._empty_regression_results("scikit-learn not available")
        
        if feature_set.features.empty:
            return self._empty_regression_results("No features available")
        
        # Convert target to binary
        y_binary = (feature_set.target > threshold).astype(int)
        
        # Remove rows with NaN values
        clean_data = pd.concat([feature_set.features, y_binary], axis=1).dropna()
        
        if len(clean_data) < 10:
            return self._empty_regression_results("Insufficient clean data")
        
        X = clean_data.iloc[:, :-1]
        y = clean_data.iloc[:, -1]
        
        # Check if we have both classes
        if len(y.unique()) < 2:
            return self._empty_regression_results("Need both positive and negative examples")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.seed, stratify=y
        )
        
        try:
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Fit model
            model = LogisticRegression(random_state=self.config.seed)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = model.predict_proba(X_train_scaled)[:, 1]
            y_pred_test = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate pseudo R-squared (McFadden's)
            log_likelihood = model.score(X_train_scaled, y_train) * len(y_train)
            null_log_likelihood = -len(y_train) * (
                np.mean(y_train) * np.log(np.mean(y_train)) +
                (1 - np.mean(y_train)) * np.log(1 - np.mean(y_train))
            )
            pseudo_r_squared = 1 - (log_likelihood / null_log_likelihood)
            
            return RegressionResults(
                model_name="Logistic Regression",
                model_type="logistic",
                r_squared=pseudo_r_squared,
                adjusted_r_squared=pseudo_r_squared,  # Approximation
                f_statistic=0,  # Not applicable
                f_pvalue=1,     # Not applicable
                coefficients=dict(zip(feature_set.feature_names, model.coef_[0])),
                coefficient_pvalues={},  # Would need additional computation
                coefficient_std_errors={},  # Would need additional computation
                fitted_values=y_pred_train,
                residuals=y_train - y_pred_train,
                predictions=y_pred_test
            )
            
        except Exception as e:
            self.logger.error(f"Error fitting logistic regression: {e}")
            return self._empty_regression_results(f"Logistic regression failed: {e}")
    
    def fit_random_forest(
        self, 
        feature_set: FeatureSet,
        n_estimators: int = 100,
        test_size: float = 0.2
    ) -> RegressionResults:
        """Fit random forest regression model."""
        
        if not SKLEARN_AVAILABLE:
            return self._empty_regression_results("scikit-learn not available")
        
        if feature_set.features.empty:
            return self._empty_regression_results("No features available")
        
        # Remove rows with NaN values
        clean_data = pd.concat([feature_set.features, feature_set.target], axis=1).dropna()
        
        if len(clean_data) < 10:
            return self._empty_regression_results("Insufficient clean data")
        
        X = clean_data.iloc[:, :-1]
        y = clean_data.iloc[:, -1]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.seed
        )
        
        try:
            # Fit model
            model = RandomForestRegressor(
                n_estimators=n_estimators, 
                random_state=self.config.seed
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Feature importance
            feature_importance = dict(zip(feature_set.feature_names, model.feature_importances_))
            
            return RegressionResults(
                model_name="Random Forest",
                model_type="random_forest",
                r_squared=r2_score(y_test, y_pred_test),
                adjusted_r_squared=0,  # Not directly applicable
                f_statistic=0,  # Not applicable
                f_pvalue=1,     # Not applicable
                coefficients={},  # Not applicable for tree-based models
                coefficient_pvalues={},
                coefficient_std_errors={},
                fitted_values=y_pred_train,
                residuals=y_train - y_pred_train,
                predictions=y_pred_test,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            self.logger.error(f"Error fitting random forest: {e}")
            return self._empty_regression_results(f"Random forest failed: {e}")
    
    def compare_models(self, feature_set: FeatureSet) -> Dict[str, RegressionResults]:
        """Compare multiple regression models."""
        
        models = {}
        
        # Fit different models
        models['linear'] = self.fit_linear_regression(feature_set)
        models['logistic'] = self.fit_logistic_regression(feature_set)
        models['random_forest'] = self.fit_random_forest(feature_set)
        
        # Log comparison results
        self.logger.info("Model Comparison Results:")
        for name, result in models.items():
            self.logger.info(f"{name}: R² = {result.r_squared:.3f}")
        
        return models
    
    def _empty_regression_results(self, reason: str) -> RegressionResults:
        """Return empty regression results."""
        self.logger.warning(f"Returning empty regression results: {reason}")
        
        return RegressionResults(
            model_name="Empty Model",
            model_type="none",
            r_squared=0,
            adjusted_r_squared=0,
            f_statistic=0,
            f_pvalue=1,
            coefficients={},
            coefficient_pvalues={},
            coefficient_std_errors={},
            fitted_values=np.array([]),
            residuals=np.array([]),
            predictions=np.array([])
        )