"""
Options flow analysis and unusual activity detection.
Analyzes options volume, open interest, and sentiment patterns.
"""
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .options_data import OptionsChain, OptionContract


@dataclass
class FlowMetrics:
    """Options flow analysis metrics."""
    date: date
    ticker: str
    
    # Volume metrics
    total_call_volume: int
    total_put_volume: int
    put_call_volume_ratio: float
    
    # Open interest metrics
    total_call_oi: int
    total_put_oi: int
    put_call_oi_ratio: float
    
    # Activity metrics
    unusual_call_activity: int
    unusual_put_activity: int
    net_flow_direction: str  # 'bullish', 'bearish', 'neutral'
    
    # IV metrics
    average_call_iv: float
    average_put_iv: float
    iv_skew: float  # put_iv - call_iv
    
    # Price metrics
    volume_weighted_strike: float
    max_pain_strike: float


@dataclass  
class UnusualActivity:
    """Individual unusual options activity alert."""
    contract: OptionContract
    activity_type: str  # 'volume_spike', 'oi_spike', 'iv_spike'
    z_score: float
    baseline_value: float
    current_value: float
    significance_level: str  # 'low', 'medium', 'high'
    timestamp: datetime


class OptionsFlowAnalyzer:
    """Analyzes options flow patterns and unusual activity."""
    
    def __init__(self, config=None):
        from ..config import get_config
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
    
    def analyze_flow(self, options_chain: OptionsChain, baseline_days: int = 20) -> FlowMetrics:
        """Analyze options flow for a single chain."""
        calls = options_chain.calls
        puts = options_chain.puts
        
        # Volume metrics
        total_call_volume = sum(opt.volume for opt in calls)
        total_put_volume = sum(opt.volume for opt in puts)
        put_call_volume_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else np.inf
        
        # Open interest metrics
        total_call_oi = sum(opt.open_interest for opt in calls)
        total_put_oi = sum(opt.open_interest for opt in puts)
        put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else np.inf
        
        # IV metrics
        call_ivs = [opt.implied_volatility for opt in calls if opt.implied_volatility > 0]
        put_ivs = [opt.implied_volatility for opt in puts if opt.implied_volatility > 0]
        
        average_call_iv = np.mean(call_ivs) if call_ivs else 0
        average_put_iv = np.mean(put_ivs) if put_ivs else 0
        iv_skew = average_put_iv - average_call_iv
        
        # Flow direction analysis
        net_flow_direction = self._determine_flow_direction(
            put_call_volume_ratio, put_call_oi_ratio, iv_skew
        )
        
        # Volume-weighted strike
        volume_weighted_strike = self._calculate_volume_weighted_strike(options_chain.options)
        
        # Max pain calculation
        max_pain_strike = self._calculate_max_pain(options_chain)
        
        return FlowMetrics(
            date=options_chain.timestamp.date(),
            ticker=options_chain.underlying_ticker,
            total_call_volume=total_call_volume,
            total_put_volume=total_put_volume,
            put_call_volume_ratio=put_call_volume_ratio,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            put_call_oi_ratio=put_call_oi_ratio,
            unusual_call_activity=0,  # TODO: implement
            unusual_put_activity=0,   # TODO: implement
            net_flow_direction=net_flow_direction,
            average_call_iv=average_call_iv,
            average_put_iv=average_put_iv,
            iv_skew=iv_skew,
            volume_weighted_strike=volume_weighted_strike,
            max_pain_strike=max_pain_strike
        )
    
    def detect_unusual_activity(
        self, 
        options_chain: OptionsChain, 
        baseline_metrics: Optional[List[FlowMetrics]] = None,
        z_threshold: float = 2.0
    ) -> List[UnusualActivity]:
        """Detect unusual options activity."""
        unusual_activities = []
        
        if not baseline_metrics:
            # Can't detect unusual activity without baseline
            return unusual_activities
        
        # Calculate baseline statistics
        baseline_df = pd.DataFrame([
            {
                'total_volume': m.total_call_volume + m.total_put_volume,
                'call_volume': m.total_call_volume,
                'put_volume': m.total_put_volume,
                'avg_iv': (m.average_call_iv + m.average_put_iv) / 2
            }
            for m in baseline_metrics
        ])
        
        if len(baseline_df) < 5:  # Need minimum baseline data
            return unusual_activities
        
        # Current metrics
        current_flow = self.analyze_flow(options_chain)
        current_total_volume = current_flow.total_call_volume + current_flow.total_put_volume
        
        # Volume spike detection
        volume_mean = baseline_df['total_volume'].mean()
        volume_std = baseline_df['total_volume'].std()
        
        if volume_std > 0:
            volume_z_score = (current_total_volume - volume_mean) / volume_std
            
            if abs(volume_z_score) >= z_threshold:
                # Find the most unusual contracts
                for contract in options_chain.options:
                    if contract.volume > 0:  # Has activity
                        # Simple heuristic: contracts with high volume relative to open interest
                        if contract.open_interest > 0:
                            volume_oi_ratio = contract.volume / contract.open_interest
                            if volume_oi_ratio > 1.0:  # Volume exceeds open interest
                                activity = UnusualActivity(
                                    contract=contract,
                                    activity_type='volume_spike',
                                    z_score=volume_z_score,
                                    baseline_value=volume_mean,
                                    current_value=current_total_volume,
                                    significance_level=self._get_significance_level(volume_z_score, z_threshold),
                                    timestamp=options_chain.timestamp
                                )
                                unusual_activities.append(activity)
        
        return unusual_activities
    
    def calculate_put_call_ratios(self, options_chains: List[OptionsChain]) -> pd.DataFrame:
        """Calculate put/call ratios over time."""
        ratios_data = []
        
        for chain in options_chains:
            flow = self.analyze_flow(chain)
            ratios_data.append({
                'date': flow.date,
                'ticker': flow.ticker,
                'put_call_volume_ratio': flow.put_call_volume_ratio,
                'put_call_oi_ratio': flow.put_call_oi_ratio,
                'iv_skew': flow.iv_skew,
                'net_flow_direction': flow.net_flow_direction
            })
        
        return pd.DataFrame(ratios_data)
    
    def _determine_flow_direction(self, pc_volume_ratio: float, pc_oi_ratio: float, iv_skew: float) -> str:
        """Determine overall flow direction from multiple indicators."""
        bullish_signals = 0
        bearish_signals = 0
        
        # Put/call volume ratio analysis
        if pc_volume_ratio < 0.8:  # More calls than puts
            bullish_signals += 1
        elif pc_volume_ratio > 1.2:  # More puts than calls  
            bearish_signals += 1
        
        # Put/call OI ratio analysis
        if pc_oi_ratio < 0.8:
            bullish_signals += 1
        elif pc_oi_ratio > 1.2:
            bearish_signals += 1
        
        # IV skew analysis (puts typically have higher IV)
        if iv_skew < 0.02:  # Unusually low put IV premium
            bullish_signals += 1
        elif iv_skew > 0.08:  # Very high put IV premium
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'bullish'
        elif bearish_signals > bullish_signals:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_volume_weighted_strike(self, options: List[OptionContract]) -> float:
        """Calculate volume-weighted average strike price."""
        total_volume = sum(opt.volume for opt in options if opt.volume > 0)
        
        if total_volume == 0:
            return 0.0
        
        weighted_sum = sum(opt.strike * opt.volume for opt in options if opt.volume > 0)
        return weighted_sum / total_volume
    
    def _calculate_max_pain(self, options_chain: OptionsChain) -> float:
        """Calculate max pain strike (strike with maximum option value loss)."""
        strikes = options_chain.strikes
        
        if not strikes:
            return options_chain.underlying_price
        
        max_pain_values = []
        
        for strike in strikes:
            total_pain = 0
            
            # Calculate pain for all options at this strike
            for option in options_chain.options:
                if option.open_interest > 0:
                    if option.option_type == 'call':
                        # Call pain: max(0, underlying - strike) * OI  
                        pain = max(0, options_chain.underlying_price - option.strike) * option.open_interest
                    else:  # put
                        # Put pain: max(0, strike - underlying) * OI
                        pain = max(0, option.strike - options_chain.underlying_price) * option.open_interest
                    
                    total_pain += pain
            
            max_pain_values.append((strike, total_pain))
        
        # Return strike with minimum total pain (maximum loss for option holders)
        if max_pain_values:
            return min(max_pain_values, key=lambda x: x[1])[0]
        
        return options_chain.underlying_price
    
    def _get_significance_level(self, z_score: float, threshold: float) -> str:
        """Map Z-score to significance level."""
        abs_z = abs(z_score)
        
        if abs_z >= threshold * 2:
            return 'high'
        elif abs_z >= threshold * 1.5:
            return 'medium'
        elif abs_z >= threshold:
            return 'low'
        else:
            return 'none'
    
    def generate_flow_report(self, flow_metrics: List[FlowMetrics]) -> Dict:
        """Generate summary report of options flow analysis."""
        if not flow_metrics:
            return {'error': 'No flow metrics provided'}
        
        df = pd.DataFrame([
            {
                'date': m.date,
                'ticker': m.ticker,
                'put_call_volume_ratio': m.put_call_volume_ratio,
                'put_call_oi_ratio': m.put_call_oi_ratio,
                'iv_skew': m.iv_skew,
                'net_flow_direction': m.net_flow_direction,
                'total_volume': m.total_call_volume + m.total_put_volume
            }
            for m in flow_metrics
        ])
        
        return {
            'summary': {
                'total_days': len(df),
                'avg_put_call_volume_ratio': df['put_call_volume_ratio'].mean(),
                'avg_put_call_oi_ratio': df['put_call_oi_ratio'].mean(),
                'avg_iv_skew': df['iv_skew'].mean(),
                'avg_daily_volume': df['total_volume'].mean(),
                'bullish_days': len(df[df['net_flow_direction'] == 'bullish']),
                'bearish_days': len(df[df['net_flow_direction'] == 'bearish']),
                'neutral_days': len(df[df['net_flow_direction'] == 'neutral'])
            },
            'trend_analysis': {
                'volume_trend': 'increasing' if df['total_volume'].iloc[-1] > df['total_volume'].iloc[0] else 'decreasing',
                'iv_skew_trend': 'increasing' if df['iv_skew'].iloc[-1] > df['iv_skew'].iloc[0] else 'decreasing',
                'dominant_direction': df['net_flow_direction'].mode().iloc[0] if len(df['net_flow_direction'].mode()) > 0 else 'neutral'
            },
            'risk_indicators': {
                'high_iv_skew_days': len(df[df['iv_skew'] > 0.08]),
                'extreme_put_call_ratio_days': len(df[(df['put_call_volume_ratio'] > 2.0) | (df['put_call_volume_ratio'] < 0.5)]),
                'low_volume_days': len(df[df['total_volume'] < df['total_volume'].quantile(0.25)])
            }
        }