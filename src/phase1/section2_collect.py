# Phase 1: Volume Information Leakage Analysis
# Section 2: Data Collection & Quality Checks (Clean, Professional Version)

from typing import Optional, Dict, Any
from datetime import timedelta
import traceback
import logging
import pandas as pd
import numpy as np
import yfinance as yf

# Import from section1_setup
from .section1_setup import LaunchEvent

logger = logging.getLogger(__name__)

def _event_to_dict(e: LaunchEvent) -> Dict[str, Any]:
    """Serialize LaunchEvent to a plain dict for metadata logging and saving."""
    return {
        "name": e.name,
        "company": e.company,
        "ticker": e.ticker,
        "announcement": e.announcement.isoformat(),
        "release": e.release.isoformat(),
        "next_earnings": e.next_earnings.isoformat() if e.next_earnings else None,
        "category": e.category,
    }


def collect_event_data(
    event: LaunchEvent,
    baseline_days: Optional[int] = None,
    signal_window_announce: Optional[tuple] = None,
    post_event_buffer_days: int = 30,
    provider: str = "yfinance",
) -> Optional[pd.DataFrame]:
    """
    Collect OHLCV data around a single launch event with academic-grade windowing.
    Anchor: Announcement date (primary). Release and earnings dates pad right tail.
    """
    assert provider == "yfinance", "Only yfinance is implemented in Section 2."

    baseline = baseline_days if baseline_days is not None else analyzer.baseline_days
    sw = signal_window_announce if signal_window_announce is not None else analyzer.signal_window_announce
    max_scan = max(sw) if isinstance(sw, (tuple, list)) else int(sw)

    a, r, ne = event.announcement, event.release, event.next_earnings

    # Build analysis window
    left_buffer = 10
    start_date = a - timedelta(days=baseline + max_scan + left_buffer)
    right_anchor = max([d for d in [r, ne] if d is not None])
    end_date = right_anchor + timedelta(days=post_event_buffer_days)

    logger.info(f"Collecting data for {event.company} — {event.name} ({event.category})")
    logger.info(f"Announcement: {a} | Release: {r} | Earnings: {ne}")
    logger.info(f"Window: {start_date} → {end_date} (baseline={baseline}d; scan={sw}; buffer={post_event_buffer_days}d)")

    try:
        df = yf.download(
            event.ticker,
            start=start_date,
            end=end_date + timedelta(days=1),  # yfinance end is exclusive
            progress=False,
            auto_adjust=False,
        )

        if df is None or df.empty:
            logger.warning("No data returned. Check ticker or network.")
            return None

        # Normalize and check required columns
        df = df.rename(columns=str.title)
        required_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"Missing expected columns: {missing}")

        df.index = pd.to_datetime(df.index)

        # Coverage metrics
        expected_days = len(pd.bdate_range(df.index.min(), df.index.max()))
        coverage_ratio = len(df) / expected_days if expected_days else 0.0
        vol_pos_ratio = float((df["Volume"] > 0).sum()) / len(df)

        logger.info(f"Downloaded {len(df)} rows | Coverage: {coverage_ratio:.1%} | Positive-volume days: {vol_pos_ratio:.1%}")

        # Save raw data
        company_clean = event.company.lower().replace(" ", "_")
        name_clean = event.name.lower().replace(" ", "_").replace("/", "-")
        raw_out = f"data/raw/{company_clean}_{name_clean}_raw.csv"
        df.to_csv(raw_out)
        logger.info(f"Saved raw file: {raw_out}")

        # Attach metadata
        df.attrs["event_info"] = _event_to_dict(event)
        df.attrs["collection_params"] = {
            "baseline_days": baseline,
            "signal_window_announce": sw,
            "post_event_buffer_days": post_event_buffer_days,
            "coverage_ratio": coverage_ratio,
            "positive_volume_ratio": vol_pos_ratio,
            "provider": provider,
        }

        return df

    except Exception:
        logger.error("Data collection error:\n" + traceback.format_exc())
        return None


def validate_data_quality(
    data: Optional[pd.DataFrame],
    event: LaunchEvent,
    academic_standards: bool = True
) -> Dict[str, Any]:
    """Validate data quality using academic thresholds."""
    if data is None or data.empty:
        return {
            "valid": False,
            "reason": "No data available",
            "academic_assessment": "Insufficient for event-study analysis",
        }

    logger.info(f"Data Quality — {event.company} {event.name}")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required if c not in data.columns]
    if missing_cols:
        return {
            "valid": False,
            "reason": f"Missing columns: {missing_cols}",
            "academic_assessment": "Critical fields missing",
        }

    # Ensure numeric Series
    def _to_numeric_series(s):
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")

    open_s  = _to_numeric_series(data["Open"])
    high_s  = _to_numeric_series(data["High"])
    low_s   = _to_numeric_series(data["Low"])
    close_s = _to_numeric_series(data["Close"])
    vol_s   = _to_numeric_series(data["Volume"])

    n = len(data)
    metrics = {
        "data_completeness": 1.0 - float(data.isnull().sum().max()) / n,
        "volume_quality": 1.0 - (int((vol_s.fillna(0) == 0).sum()) / n),
        "price_continuity": 1.0 - (int((close_s.pct_change().abs() > 0.5).sum()) / n),
        "temporal_coverage": (n / len(pd.bdate_range(data.index.min(), data.index.max()))) if n else 0.0
    }

    if academic_standards:
        thresholds = {
            "data_completeness": 0.90,
            "volume_quality": 0.95,
            "price_continuity": 0.98,
            "temporal_coverage": 0.80,
        }
        failed = [f"{k}: {metrics[k]:.2%} < {t:.0%}" for k, t in thresholds.items() if metrics[k] < t]
        grade = sum(metrics[k] >= t for k, t in thresholds.items()) / len(thresholds)
        logger.info(
            f"Completeness: {metrics['data_completeness']:.1%} | "
            f"Volume: {metrics['volume_quality']:.1%} | "
            f"Continuity: {metrics['price_continuity']:.1%} | "
            f"Coverage: {metrics['temporal_coverage']:.1%}"
        )
    else:
        failed, grade = [], None

    is_valid = (
        metrics["data_completeness"] >= 0.80 and
        metrics["volume_quality"] >= 0.90 and
        metrics["price_continuity"] >= 0.95 and
        metrics["temporal_coverage"] >= 0.70
    )

    result = {
        "valid": is_valid,
        "total_rows": n,
        "date_range": f"{data.index.min().date()} → {data.index.max().date()}",
        "metrics": metrics,
        "avg_daily_volume": float(np.nanmean(vol_s.values)),
        "volume_std": float(np.nanstd(vol_s.values, ddof=1)) if n > 1 else 0.0,
        "price_range": f"${float(np.nanmin(close_s.values)):.2f}–${float(np.nanmax(close_s.values)):.2f}"
    }

    if academic_standards:
        result.update({
            "academic_grade": grade,
            "meets_academic_standards": grade is not None and grade >= 0.75,
            "failed_academic_criteria": failed,
            "academic_assessment": (
                "Meets academic standards" if grade and grade >= 0.75
                else "Marginal — note limitations" if grade and grade >= 0.50
                else "Below academic standards"
            )
        })

    return result


