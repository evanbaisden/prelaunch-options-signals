"""
Data types and schemas for prelaunch options signals analysis.
Defines core data structures used throughout the pipeline.
"""
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Dict, Any
import pandas as pd


@dataclass(frozen=True)
class LaunchEvent:
    """Represents a technology product launch event."""
    name: str
    company: str
    ticker: str
    announcement: date
    release: date
    next_earnings: Optional[date] = None
    category: str = "Technology"
    
    def __post_init__(self):
        """Validate event dates."""
        if self.announcement > self.release:
            raise ValueError(f"Announcement date cannot be after release date for {self.name}")


@dataclass
class BaselineMetrics:
    """Statistical baseline metrics for a stock during the pre-announcement period."""
    avg_daily_return: float
    avg_volume: float
    volume_std: float
    price_volatility: float
    trading_days: int
    start_date: date
    end_date: date
    
    @property
    def volume_cv(self) -> float:
        """Coefficient of variation for volume."""
        return self.volume_std / self.avg_volume if self.avg_volume > 0 else 0.0


@dataclass
class SignalMetrics:
    """Signal detection metrics around announcement/release events."""
    event_date: date
    event_type: str  # 'announcement' or 'release'
    
    # Price metrics
    price_5day_return: float
    price_1day_return: float
    
    # Volume metrics
    volume_spike_pct: float
    volume_z_score: float
    volume_anomaly_detected: bool
    
    # Optional price metrics
    abnormal_return: Optional[float] = None
    
    # Statistical significance
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None
    significance_level: Optional[str] = None  # 'low', 'med', 'high', 'extreme'


@dataclass
class AnalysisResults:
    """Complete analysis results for a product launch."""
    event: LaunchEvent
    baseline: BaselineMetrics
    announcement_signal: Optional[SignalMetrics] = None
    release_signal: Optional[SignalMetrics] = None
    
    # Period averages
    pre_announce_avg_return: float = 0.0
    announce_to_release_avg_return: float = 0.0
    post_release_avg_return: float = 0.0
    
    # Volume patterns
    pre_announce_avg_volume: float = 0.0
    announce_to_release_avg_volume: float = 0.0
    post_release_avg_volume: float = 0.0
    
    # Analysis metadata
    analysis_timestamp: Optional[datetime] = None
    data_quality_score: float = 1.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        return {
            'product': self.event.name,
            'company': self.event.company,
            'ticker': self.event.ticker,
            'announcement_date': self.event.announcement.isoformat(),
            'release_date': self.event.release.isoformat(),
            'category': self.event.category,
            
            # Baseline metrics
            'baseline_avg_return': self.baseline.avg_daily_return,
            'baseline_avg_volume': self.baseline.avg_volume,
            'baseline_volatility': self.baseline.price_volatility,
            
            # Signal metrics
            'announcement_5day_return': self.announcement_signal.price_5day_return if self.announcement_signal else None,
            'announcement_volume_spike': self.announcement_signal.volume_spike_pct if self.announcement_signal else None,
            'announcement_z_score': self.announcement_signal.volume_z_score if self.announcement_signal else None,
            
            'release_5day_return': self.release_signal.price_5day_return if self.release_signal else None,
            'release_volume_spike': self.release_signal.volume_spike_pct if self.release_signal else None,
            'release_z_score': self.release_signal.volume_z_score if self.release_signal else None,
            
            # Period averages
            'pre_announce_avg_return': self.pre_announce_avg_return,
            'announce_to_release_avg_return': self.announce_to_release_avg_return,
            'post_release_avg_return': self.post_release_avg_return,
            
            'pre_announce_avg_volume': self.pre_announce_avg_volume,
            'announce_to_release_avg_volume': self.announce_to_release_avg_volume,
            'post_release_avg_volume': self.post_release_avg_volume,
            
            # Quality metrics
            'data_quality_score': self.data_quality_score,
            'analysis_timestamp': self.analysis_timestamp.isoformat() if self.analysis_timestamp else None,
            'notes': self.notes
        }


@dataclass
class StockData:
    """Container for cleaned stock price/volume data."""
    df: pd.DataFrame
    ticker: str
    date_range: tuple[date, date]
    data_quality_checks: Dict[str, bool]
    
    def __post_init__(self):
        """Validate required columns."""
        required_cols = ['Date', 'Adj Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    @property
    def trading_days(self) -> int:
        """Number of trading days in dataset."""
        return len(self.df)
    
    @property
    def avg_daily_volume(self) -> float:
        """Average daily trading volume."""
        return self.df['Volume'].mean()