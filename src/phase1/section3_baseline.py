# =========================
# Phase 1: Volume Information Leakage Analysis
# Section 3: Baseline Analysis & Thresholds
# =========================

from datetime import timedelta
from typing import Dict, Any, Optional
import re
import logging
import pandas as pd
import numpy as np

from .section1_setup import LaunchEvent

def normalize_ohlcv_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize OHLCV columns to standard names: 
    ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'].
    Handles MultiIndex, suffixed column names, and varied cases.
    """
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        levels0 = set(map(str, out.columns.get_level_values(0)))
        levels1 = set(map(str, out.columns.get_level_values(1)))
        if ticker in levels0:
            out = out.xs(ticker, axis=1, level=0)
        elif ticker.upper() in levels0:
            out = out.xs(ticker.upper(), axis=1, level=0)
        elif ticker in levels1:
            out = out.xs(ticker, axis=1, level=1)
        elif ticker.upper() in levels1:
            out = out.xs(ticker.upper(), axis=1, level=1)
        else:
            out.columns = ["_".join(map(str, tup)) for tup in out.columns.to_list()]

    def normalize_name(col: str) -> str:
        c = str(col)
        c = re.sub(rf"(?i)[_\-\.]{ticker}$", "", c)
        c = re.sub(r"(?i)[_\-\.][A-Z]{1,5}$", "", c)
        c = c.strip().lower()
        if "adj" in c and "close" in c: return "Adj Close"
        if "close" in c:                return "Close"
        if "open" in c:                 return "Open"
        if "high" in c:                 return "High"
        if "low" in c:                  return "Low"
        if "volume" in c or c == "vol": return "Volume"
        return col

    out.columns = [normalize_name(c) for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")]
    return out

def get_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col] if col in df.columns else pd.Series(index=df.index, dtype=float)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")

def establish_baseline_metrics(
    data: pd.DataFrame,
    event: LaunchEvent,
    baseline_days: int,
    signal_window_announce: tuple
) -> Optional[Dict[str, Any]]:
    if data is None or data.empty:
        logging.warning("No data passed to establish_baseline_metrics.")
        return None

    df = normalize_ohlcv_columns(data, event.ticker)

    for req in ["Volume", "Close"]:
        if req not in df.columns:
            raise KeyError(f"Required column '{req}' not found. Columns: {list(df.columns)}")

    vol = get_numeric_series(df, "Volume")
    close = get_numeric_series(df, "Close")

    a = event.announcement
    min_scan = int(signal_window_announce[0])
    baseline_end = a - timedelta(days=min_scan)
    baseline_start = baseline_end - timedelta(days=int(baseline_days))

    df.index = pd.to_datetime(df.index)
    mask = (df.index >= pd.Timestamp(baseline_start)) & (df.index < pd.Timestamp(baseline_end))
    baseline_data = df.loc[mask]

    if len(baseline_data) < 30:
        logging.warning(f"Insufficient baseline data ({len(baseline_data)} trading days). Need ≥ 30.")
        return None

    b_vol = get_numeric_series(baseline_data, "Volume")
    b_close = get_numeric_series(baseline_data, "Close")

    td_count = len(baseline_data)
    mean_v = float(b_vol.mean())
    std_v  = float(b_vol.std(ddof=1)) if td_count > 1 else 0.0
    median_v = float(b_vol.median())
    mode_v = float(b_vol.mode().iloc[0]) if not b_vol.mode().empty else median_v
    cv_v = (std_v / mean_v) if mean_v > 0 else float("nan")

    return {
        "period_start": pd.Timestamp(baseline_start),
        "period_end": pd.Timestamp(baseline_end),
        "trading_days": td_count,
        "volume_mean": mean_v,
        "volume_median": median_v,
        "volume_mode": mode_v,
        "volume_std": std_v,
        "volume_variance": float(b_vol.var(ddof=1)) if td_count > 1 else 0.0,
        "volume_range": float(b_vol.max() - b_vol.min()),
        "volume_iqr": float(b_vol.quantile(0.75) - b_vol.quantile(0.25)),
        "volume_skewness": float(b_vol.skew()),
        "volume_kurtosis": float(b_vol.kurtosis()),
        "coefficient_of_variation": float(cv_v),
        "volume_q10": float(b_vol.quantile(0.10)),
        "volume_q25": float(b_vol.quantile(0.25)),
        "volume_q75": float(b_vol.quantile(0.75)),
        "volume_q90": float(b_vol.quantile(0.90)),
        "volume_q95": float(b_vol.quantile(0.95)),
        "volume_q99": float(b_vol.quantile(0.99)),
        "avg_close_price": float(b_close.mean()),
        "price_volatility": float(b_close.pct_change().std(ddof=1)),
        "avg_daily_return": float(b_close.pct_change().mean()),
        "high_volume_days": int((b_vol > b_vol.quantile(0.90)).sum()),
        "low_volume_days": int((b_vol < b_vol.quantile(0.10)).sum()),
    }

def calculate_volume_thresholds(
    baseline_metrics: Dict[str, Any],
    academic_validation: bool = True
) -> Optional[Dict[str, float]]:
    if not baseline_metrics:
        return None

    mean_v = baseline_metrics["volume_mean"]
    std_v  = baseline_metrics["volume_std"]

    thresholds: Dict[str, float] = {}

    if academic_validation:
        academic_mults = {
            "screening_5pct": 1.645,
            "statistical_1pct": 2.326,
            "conservative_05pct": 2.576,
            "extreme_trading": 5.0
        }
        for k, m in academic_mults.items():
            thresholds[k] = mean_v + m * std_v

    for name, m in zip(["moderate_std", "unusual_std", "spike_std", "extreme_std"], [1.5, 2.0, 2.5, 3.0]):
        thresholds[name] = mean_v + m * std_v

    thresholds.update({
        "p90_threshold": baseline_metrics["volume_q90"],
        "p95_threshold": baseline_metrics["volume_q95"],
        "p99_threshold": baseline_metrics["volume_q99"],
        "ratio_1_5x": mean_v * 1.5,
        "ratio_2x":   mean_v * 2.0,
        "ratio_3x":   mean_v * 3.0,
        "ratio_4x":   mean_v * 4.0,
    })

    logging.info(
        f"Thresholds — 1.645σ: {thresholds['screening_5pct']:.0f}, "
        f"2.326σ: {thresholds['statistical_1pct']:.0f}, "
        f"2.576σ: {thresholds['conservative_05pct']:.0f}, "
        f"p95: {thresholds['p95_threshold']:.0f}"
    )

    return thresholds

def log_baseline_summary(event: LaunchEvent, baseline_metrics: Dict[str, Any], thresholds: Dict[str, float]) -> None:
    if not baseline_metrics:
        logging.error(f"{event.name}: Baseline analysis failed.")
        return
    bm = baseline_metrics
    logging.info(
        f"Baseline Analysis — {event.company} {event.name} | Period: "
        f"{bm['period_start'].date()} → {bm['period_end'].date()} | Trading days: {bm['trading_days']}"
    )
    logging.info(
        f"Volume Mean: {bm['volume_mean']:.0f}, Median: {bm['volume_median']:.0f}, Std: {bm['volume_std']:.0f}, "
        f"CV: {bm['coefficient_of_variation']:.3f}"
    )

# Smoke test
if 'analyzer' in locals():
    ev = next((e for e in analyzer.events if e.name == "iPhone 14"), analyzer.events[0])
    if 'data_ev' not in locals() or data_ev is None:
        data_ev = collect_event_data(ev, analyzer.baseline_days, analyzer.signal_window_announce, 30)
    if data_ev is not None and not data_ev.empty:
        bm = establish_baseline_metrics(data_ev, ev, analyzer.baseline_days, analyzer.signal_window_announce)
        if bm:
            th = calculate_volume_thresholds(bm)
            log_baseline_summary(ev, bm, th)
