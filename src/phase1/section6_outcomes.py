# =========================
# Phase 1: Volume Information Leakage Analysis
# Section 6: Outcome Measurement (Clean Print Version)
# =========================

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from .section1_setup import LaunchEvent
from .section3_baseline import normalize_ohlcv_columns, get_numeric_series

def _nearest_trading_idx(idx: pd.DatetimeIndex, target: pd.Timestamp) -> int:
    """Return the index position of the nearest trading day to target."""
    try:
        pos = idx.get_loc(target)
        if isinstance(pos, slice):
            pos = pos.start
        if isinstance(pos, (list, np.ndarray)) and len(pos):
            pos = pos[0]
        return int(pos)
    except KeyError:
        return int(idx.get_indexer([target], method="nearest")[0])


def measure_stock_performance_outcomes(
    data: pd.DataFrame,
    event: LaunchEvent,
    short_term_windows_td: List[int] = [1, 3, 5, 10, 20]
) -> Optional[Dict[str, Any]]:
    """
    Measure performance for:
      Announcement -> Release
      Release -> Earnings
      Announcement -> Earnings
    Includes short-term post-release returns, volume context, and volatility.
    """
    if data is None or data.empty:
        return None

    df = normalize_ohlcv_columns(data, event.ticker).copy()
    for req in ["Close", "Volume", "Open", "High", "Low"]:
        if req not in df.columns:
            print(f"    Missing column '{req}' for outcome calculation.")
            return None
    df.index = pd.to_datetime(df.index)

    close = get_numeric_series(df, "Close")
    vol = get_numeric_series(df, "Volume")

    a = pd.Timestamp(event.announcement)
    r = pd.Timestamp(event.release)
    ne = pd.Timestamp(event.next_earnings) if event.next_earnings else None
    if ne is None:
        print("    next_earnings is missing; cannot compute release->earnings outcomes.")
        return None

    try:
        idx = df.index
        ai = _nearest_trading_idx(idx, a)
        ri = _nearest_trading_idx(idx, r)
        ei = _nearest_trading_idx(idx, ne)

        ann_px = float(close.iloc[ai])
        rel_px = float(close.iloc[ri])
        earn_px = float(close.iloc[ei])

        ann_to_rel_ret = (rel_px / ann_px - 1.0) * 100.0
        rel_to_earn_ret = (earn_px / rel_px - 1.0) * 100.0
        full_ret = (earn_px / ann_px - 1.0) * 100.0

        # Short-term returns after release (trading-day offsets)
        short_terms = {}
        for k in short_term_windows_td:
            tgt_i = min(ri + k, len(df) - 1)
            tgt_px = float(close.iloc[tgt_i])
            short_terms[f"return_{k}d"] = (tgt_px / rel_px - 1.0) * 100.0

        # Volume context (5 trading days pre/post release)
        pre_avg_vol = float(df.iloc[max(0, ri - 5):ri]["Volume"].mean()) if ri > 0 else None
        post_avg_vol = float(df.iloc[ri + 1:min(len(df), ri + 6)]["Volume"].mean()) if ri < len(df) - 1 else None
        rel_vol = float(vol.iloc[ri])
        vol_ratio = (rel_vol / pre_avg_vol) if pre_avg_vol and pre_avg_vol > 0 else None

        # Volatility over release -> earnings window
        lo, hi = min(ri, ei), max(ri, ei)
        win = df.iloc[lo:hi + 1]
        if len(win) > 1:
            px_vol = float(get_numeric_series(win, "Close").pct_change().std(ddof=1))  # decimal
            vol_vol = float(get_numeric_series(win, "Volume").std(ddof=1))
        else:
            px_vol, vol_vol = None, None

        return {
            "announcement_date": a,
            "release_date": r,
            "earnings_date": ne,
            "announcement_price": ann_px,
            "release_price": rel_px,
            "earnings_price": earn_px,
            "announcement_to_release_return": ann_to_rel_ret,
            "release_to_earnings_return": rel_to_earn_ret,
            "full_period_return": full_ret,
            **short_terms,
            "release_day_volume": int(rel_vol),
            "pre_release_avg_volume": pre_avg_vol,
            "post_release_avg_volume": post_avg_vol,
            "release_volume_ratio": vol_ratio,
            "price_volatility_release_period": px_vol,   # decimal
            "volume_volatility_release_period": vol_vol,
            "announcement_to_release_days": int((r.date() - a.date()).days),
            "release_to_earnings_days": int((ne.date() - r.date()).days),
            "total_analysis_days": int((ne.date() - a.date()).days),
        }

    except Exception as e:
        print(f"    ERROR: outcome calculation failed: {e}")
        return None


def calculate_risk_adjusted_returns(
    outcome_metrics: Optional[Dict[str, Any]],
    baseline_metrics: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Compute simple risk-adjusted figures with available volatilities."""
    if not outcome_metrics:
        return None

    ra: Dict[str, Optional[float]] = {}
    px_vol = outcome_metrics.get("price_volatility_release_period")   # decimal
    rel_to_earn = outcome_metrics.get("release_to_earnings_return")   # percent

    # Return/volatility ratio using release-period volatility
    if px_vol is not None and rel_to_earn is not None:
        denom = px_vol * 100.0 if px_vol > 0 else None  # convert decimal -> percent
        ra["return_volatility_ratio"] = (abs(rel_to_earn) / denom) if denom and denom > 0 else None
    else:
        ra["return_volatility_ratio"] = None

    # Information-ratio-style vs baseline volatility (baseline price_volatility is decimal)
    if baseline_metrics and ("price_volatility" in baseline_metrics) and rel_to_earn is not None:
        base_vol_pct = float(baseline_metrics["price_volatility"]) * 100.0
        ra["information_ratio"] = (rel_to_earn / base_vol_pct) if base_vol_pct > 0 else None
    else:
        ra["information_ratio"] = None

    # Max drawdown across short-term window returns (percentage points)
    returns = [v for k, v in outcome_metrics.items() if k.startswith("return_") and v is not None]
    if returns:
        cum_max, max_dd = -float("inf"), 0.0
        for r_ in returns:
            cum_max = max(cum_max, r_)
            max_dd = max(max_dd, cum_max - r_)
        ra["max_drawdown"] = float(max_dd)
        ra["calmar_ratio"] = (rel_to_earn / max_dd) if max_dd > 0 else None
    else:
        ra["max_drawdown"] = None
        ra["calmar_ratio"] = None

    return ra


def print_outcome_summary(
    event: LaunchEvent,
    outcome_metrics: Optional[Dict[str, Any]],
    risk_adjusted_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Print concise outcome and risk-adjusted metrics."""
    print(f"\nOutcome Analysis — {event.company} {event.name}")
    if not outcome_metrics:
        print("    No outcome metrics available.")
        return

    om = outcome_metrics
    print("    Dates and Prices")
    print(f"      Announcement: {om['announcement_date'].date()}  Price: ${om['announcement_price']:.2f}")
    print(f"      Release     : {om['release_date'].date()}  Price: ${om['release_price']:.2f}")
    print(f"      Earnings    : {om['earnings_date'].date()}  Price: ${om['earnings_price']:.2f}")

    print("\n    Returns")
    print(f"      Announcement -> Release : {om['announcement_to_release_return']:+.2f}%")
    print(f"      Release -> Earnings     : {om['release_to_earnings_return']:+.2f}%")
    print(f"      Full Period             : {om['full_period_return']:+.2f}%")

    print("\n    Short-term Returns after Release")
    for k in [1, 3, 5, 10, 20]:
        key = f"return_{k}d"
        if key in om and om[key] is not None:
            print(f"      {k:>2}d: {om[key]:+.2f}%")

    print("\n    Volume")
    print(f"      Release-day volume          : {om['release_day_volume']:,}")
    if om.get("release_volume_ratio") is not None:
        print(f"      Release / Pre-release avg   : {om['release_volume_ratio']:.2f}x")

    if risk_adjusted_metrics:
        ra = risk_adjusted_metrics
        print("\n    Risk-Adjusted")
        if ra.get("return_volatility_ratio") is not None:
            print(f"      Return/Volatility ratio     : {ra['return_volatility_ratio']:.2f}")
        if ra.get("information_ratio") is not None:
            print(f"      Information ratio           : {ra['information_ratio']:.2f}")
        if ra.get("max_drawdown") is not None:
            print(f"      Max drawdown (short windows): {ra['max_drawdown']:.2f} pp")
        if ra.get("calmar_ratio") is not None:
            print(f"      Calmar-like ratio           : {ra['calmar_ratio']:.2f}")


