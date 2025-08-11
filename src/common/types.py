from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple, Dict, List, TypedDict

# ---- Core domain objects ----
@dataclass(frozen=True)
class LaunchEvent:
    name: str
    company: str
    ticker: str
    announcement: date
    release: date
    next_earnings: Optional[date]
    category: str

@dataclass(frozen=True)
class AnalysisParams:
    baseline_days: int = 60
    signal_window_announce: Tuple[int, int] = (5, 20)
    signal_window_release: Tuple[int, int] = (5, 10)
    z_thresholds: Tuple[float, ...] = (1.645, 2.326, 2.576, 5.0)

# ---- TypedDicts used across steps (lean subsets of the keys you use) ----
class DataQualityReport(TypedDict, total=False):
    valid: bool
    reason: str
    total_rows: int
    date_range: str
    metrics: Dict[str, float]
    avg_daily_volume: float
    volume_std: float
    price_range: str
    academic_grade: float
    meets_academic_standards: bool
    failed_academic_criteria: List[str]
    academic_assessment: str

class BaselineMetrics(TypedDict, total=False):
    period_start: str
    period_end: str
    trading_days: int
    volume_mean: float
    volume_std: float
    volume_median: float
    coefficient_of_variation: float
    volume_q90: float
    volume_q95: float
    volume_q99: float
    price_volatility: float  # daily return stdev (decimal)

class Thresholds(TypedDict, total=False):
    screening_5pct: float
    statistical_1pct: float
    conservative_05pct: float
    extreme_trading: float
    unusual_std: float
    spike_std: float
    extreme_std: float
    p90_threshold: float
    p95_threshold: float
    p99_threshold: float
    ratio_2x: float

class AnomalyRecord(TypedDict, total=False):
    date: str
    volume: float
    z_score: float
    ratio_to_baseline: float
    anomaly_type: str
    severity_rank: int
    threshold_exceeded: str
    threshold_value: float
    confidence_level: str
    days_before_announcement: int
    in_optimal_window: bool
    timing_assessment: str
    open_price: float
    close_price: float
    price_change_from_open_pct: float
    intraday_range_pct: float
    company: str
    product_category: str

class PatternSummary(TypedDict, total=False):
    total_anomalies: int
    severity_distribution: Dict[str, int]
    timing_stats: Dict[str, float]
    clustering: Dict[str, object]
    intensity_stats: Dict[str, float]

class SignalMetrics(TypedDict, total=False):
    total_anomaly_days: int
    weighted_avg_z_score: float
    max_volume_ratio: float
    earliest_signal_days: int
    latest_signal_days: int
    composite_signal_score: float
    signal_confidence: float
    extreme_signals: int
    highly_significant_signals: int
    notable_signals: int

class LeakageAssessment(TypedDict, total=False):
    leakage_likelihood: str
    confidence_level: str
    evidence_strength: str
    leakage_score: int
    max_possible_score: int
    evidence_factors: List[str]
    academic_factors: List[str]
    academic_assessment: str

class OutcomeMetrics(TypedDict, total=False):
    announcement_price: float
    release_price: float
    earnings_price: float
    announcement_to_release_return: float
    release_to_earnings_return: float
    full_period_return: float
    release_day_volume: int
    release_volume_ratio: float

class RiskAdjustedMetrics(TypedDict, total=False):
    return_volatility_ratio: float
    information_ratio: float
    max_drawdown: float
    calmar_ratio: float
