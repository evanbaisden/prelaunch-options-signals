# =========================
# Phase 1: Volume Information Leakage Analysis
# Section 4: Anomaly Detection (Clean Log Format)
# =========================

from datetime import timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

from .section1_setup import LaunchEvent
from .section3_baseline import normalize_ohlcv_columns, get_numeric_series

def detect_volume_anomalies(
    data: pd.DataFrame,
    event: LaunchEvent,
    baseline_metrics: Dict[str, Any],
    thresholds: Dict[str, float],
    signal_window_announce: tuple = (5, 20)
) -> List[Dict[str, Any]]:
    if not (isinstance(baseline_metrics, dict) and isinstance(thresholds, dict)) or data is None or data.empty:
        return []

    df = normalize_ohlcv_columns(data, event.ticker)
    for req in ["Open", "High", "Low", "Close", "Volume"]:
        if req not in df.columns:
            logging.warning(f"Missing column '{req}' after normalization. Columns: {list(df.columns)}")
            return []

    vol = get_numeric_series(df, "Volume")
    open_s = get_numeric_series(df, "Open")
    close = get_numeric_series(df, "Close")
    high  = get_numeric_series(df, "High")
    low   = get_numeric_series(df, "Low")

    a = event.announcement
    w_min, w_max = int(signal_window_announce[0]), int(signal_window_announce[1])
    signal_start = a - timedelta(days=w_max)
    signal_end   = a  # exclusive

    mask = (df.index >= pd.Timestamp(signal_start)) & (df.index < pd.Timestamp(signal_end))
    signal_data = df.loc[mask].copy()
    if signal_data.empty:
        logging.warning(f"No pre-announcement signal data for {event.name}")
        return []

    start_str = pd.Timestamp(signal_start).date()
    end_str   = pd.Timestamp(a - timedelta(days=1)).date()
    logging.info(f"Signal detection for {event.company} {event.name} ({event.category})")
    logging.info(f"Window: {start_str} → {end_str} ({len(signal_data)} trading days) "
                 f"| Target band: {w_min}–{w_max} days pre-announcement")

    mean_v = float(baseline_metrics["volume_mean"])
    std_v  = float(baseline_metrics.get("volume_std", 0.0)) or 0.0
    std_v  = std_v if std_v > 0 else 1e-9

    vulnerability_factors = {
        "Consumer Hardware":       {"leak_timing": "early",   "spread": "broad"},
        "Semiconductor Hardware":  {"leak_timing": "later",   "spread": "contained"},
        "Gaming Hardware":         {"leak_timing": "moderate","spread": "moderate"},
    }
    vuln = vulnerability_factors.get(event.category, {"leak_timing": "unknown", "spread": "unknown"})

    anomalies: List[Dict[str, Any]] = []

    for idx, row in signal_data.iterrows():
        v = float(row["Volume"])
        o = float(row["Open"])
        c = float(row["Close"])
        h = float(row["High"])
        l = float(row["Low"])

        idx_date = pd.Timestamp(idx).date()
        days_before_announce = (a - idx_date).days

        z = (v - mean_v) / std_v
        ratio = v / mean_v if mean_v > 0 else float("inf")

        anomaly_type = "normal"
        threshold_key = None
        confidence = None
        severity_rank = 0
        if "extreme_trading" in thresholds and v > thresholds["extreme_trading"]:
            anomaly_type, threshold_key, confidence, severity_rank = "extreme", "extreme_trading", "sys_5σ", 5
        elif "conservative_05pct" in thresholds and v > thresholds["conservative_05pct"]:
            anomaly_type, threshold_key, confidence, severity_rank = "highly_significant", "conservative_05pct", "99.5%", 4
        elif "statistical_1pct" in thresholds and v > thresholds["statistical_1pct"]:
            anomaly_type, threshold_key, confidence, severity_rank = "significant", "statistical_1pct", "99%", 3
        elif "screening_5pct" in thresholds and v > thresholds["screening_5pct"]:
            anomaly_type, threshold_key, confidence, severity_rank = "notable", "screening_5pct", "95%", 2
        elif v > thresholds.get("extreme_std", float("inf")):
            anomaly_type, threshold_key, confidence, severity_rank = "extreme", "extreme_std", "3σ", 5
        elif v > thresholds.get("spike_std", float("inf")):
            anomaly_type, threshold_key, confidence, severity_rank = "spike", "spike_std", "2.5σ", 3
        elif v > thresholds.get("unusual_std", float("inf")):
            anomaly_type, threshold_key, confidence, severity_rank = "unusual", "unusual_std", "2σ", 2

        if anomaly_type != "normal":
            in_optimal = (w_min <= days_before_announce <= w_max)
            timing_flag = "optimal" if in_optimal else ("early" if days_before_announce > w_max else "late")
            anomalies.append({
                "date": idx,
                "volume": v,
                "z_score": z,
                "ratio_to_baseline": ratio,
                "anomaly_type": anomaly_type,
                "severity_rank": severity_rank,
                "threshold_exceeded": threshold_key,
                "threshold_value": thresholds.get(threshold_key, float("nan")),
                "confidence_level": confidence,
                "days_before_announcement": days_before_announce,
                "in_optimal_window": in_optimal,
                "timing_assessment": timing_flag,
                "open_price": o,
                "close_price": c,
                "price_change_from_open_pct": ((c - o) / o) * 100 if o else float("nan"),
                "intraday_range_pct": ((h - l) / l) * 100 if l else float("nan"),
                "company": event.company,
                "product_category": event.category,
                "vulnerability_profile": vuln
            })

    anomalies.sort(key=lambda a: (-a["severity_rank"], a["days_before_announcement"]))
    n_opt = sum(1 for a in anomalies if a["in_optimal_window"])
    logging.info(f"Anomalies in {w_min}–{w_max}d window: {n_opt} | Total anomalies: {len(anomalies)}")
    logging.info(f"Category vulnerability: {vuln['leak_timing']} timing, {vuln['spread']} spread")
    return anomalies


def classify_anomaly_patterns(anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not anomalies:
        return {"total_anomalies": 0, "severity_distribution": {}, "timing_stats": {}, "clustering": {}, "intensity_stats": {}}

    sev_counts: Dict[str, int] = {}
    for a in anomalies:
        sev_counts[a["anomaly_type"]] = sev_counts.get(a["anomaly_type"], 0) + 1

    timing_vals = [a["days_before_announcement"] for a in anomalies]
    earliest = int(max(timing_vals)) if timing_vals else 0
    latest   = int(min(timing_vals)) if timing_vals else 0
    avg_t    = float(np.mean(timing_vals)) if timing_vals else 0.0
    std_t    = float(np.std(timing_vals)) if timing_vals else 0.0

    dts = sorted([pd.Timestamp(a["date"]).normalize() for a in anomalies])
    clusters = []
    if dts:
        cur = [dts[0]]
        for i in range(1, len(dts)):
            if (dts[i] - dts[i-1]).days <= 2:
                cur.append(dts[i])
            else:
                if len(cur) > 1: clusters.append(cur)
                cur = [dts[i]]
        if len(cur) > 1: clusters.append(cur)

    z_vals = [a["z_score"] for a in anomalies]
    r_vals = [a["ratio_to_baseline"] for a in anomalies]

    return {
        "total_anomalies": len(anomalies),
        "severity_distribution": sev_counts,
        "timing_stats": {
            "earliest_days_before": earliest,
            "latest_days_before": latest,
            "avg_days_before": avg_t,
            "timing_std": std_t,
        },
        "clustering": {
            "total_clusters": len(clusters),
            "cluster_sizes": [len(c) for c in clusters],
            "isolated_events": len(anomalies) - sum(len(c) for c in clusters),
        },
        "intensity_stats": {
            "avg_z_score": float(np.mean(z_vals)) if z_vals else 0.0,
            "max_z_score": float(np.max(z_vals)) if z_vals else 0.0,
            "avg_volume_ratio": float(np.mean(r_vals)) if r_vals else 0.0,
            "max_volume_ratio": float(np.max(r_vals)) if r_vals else 0.0,
        }
    }


def print_anomaly_summary(event: LaunchEvent, anomalies: List[Dict[str, Any]], patterns: Dict[str, Any]) -> None:
    logging.info(f"Anomaly Detection — {event.company} {event.name}")
    if not anomalies:
        logging.info("No volume anomalies detected in the pre-announcement window.")
        return

    logging.info(f"Total anomalies: {len(anomalies)}")
    logging.info("Severity breakdown:")
    for sev, cnt in sorted(patterns["severity_distribution"].items(), key=lambda x: (-x[1], x[0])):
        logging.info(f"{sev}: {cnt}")

    t = patterns["timing_stats"]
    logging.info(f"Earliest signal: {t['earliest_days_before']}d before announcement")
    logging.info(f"Latest signal: {t['latest_days_before']}d before announcement")
    logging.info(f"Average timing: {t['avg_days_before']:.1f}d (σ={t['timing_std']:.1f})")

    i = patterns["intensity_stats"]
    logging.info(f"Avg Z-score: {i['avg_z_score']:.2f} | Max Z-score: {i['max_z_score']:.2f}")
    logging.info(f"Avg vol ratio: {i['avg_volume_ratio']:.2f}x | Max vol ratio: {i['max_volume_ratio']:.2f}x")

    logging.info("Top anomalies (max 10):")
    for k, a in enumerate(anomalies[:10], 1):
        when = a["date"].strftime("%Y-%m-%d")
        logging.info(f"{k:>2}. {when} — {int(a['volume']):,} vol "
                     f"({a['z_score']:.1f}σ, {a['ratio_to_baseline']:.1f}x) "
                     f"[{a['anomaly_type']}] · {a['days_before_announcement']}d before")


# -------------------------
# Section 4 smoke test
# -------------------------
def smoke_test():
    """Run smoke test for section 4 anomaly detection"""
    if 'analyzer' not in locals():
        logging.error("Analyzer not found. Run Section 1 first.")
        return
    
    if 'ev' not in locals():
        ev = next(e for e in analyzer.events if e.name == "iPhone 14")
    if 'data_ev' not in locals() or data_ev is None or data_ev.empty:
        from .section2_collect import collect_event_data
        data_ev = collect_event_data(ev, analyzer.baseline_days, analyzer.signal_window_announce, 30)
    if 'bm' not in locals() or not bm:
        from .section3_baseline import establish_baseline_metrics
        bm = establish_baseline_metrics(data_ev, ev, analyzer.baseline_days, analyzer.signal_window_announce)
    if 'th' not in locals() or not th:
        from .section3_baseline import calculate_volume_thresholds
        th = calculate_volume_thresholds(bm, academic_validation=True)

    if data_ev is None or not bm or not th:
        logging.error("Section 4 test skipped — prerequisites missing.")
    else:
        logging.info("Running anomaly detection…")
        anomalies = detect_volume_anomalies(data_ev, ev, bm, th, signal_window_announce=analyzer.signal_window_announce)
        patterns = classify_anomaly_patterns(anomalies)
        print_anomaly_summary(ev, anomalies, patterns)
        logging.info("Section 4 test complete.")

if __name__ == "__main__":
    smoke_test()
