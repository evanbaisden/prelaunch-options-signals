# =========================
# Phase 1: Volume Information Leakage Analysis
# Section 7: Main Analysis Workflow
# =========================

from typing import Dict, Any, Optional
from math import erf, sqrt
import numpy as np
import pandas as pd
from datetime import datetime

# Import required functions from other sections
from .section1_setup import LaunchEvent
from .section2_collect import collect_event_data, validate_data_quality
from .section3_baseline import establish_baseline_metrics, calculate_volume_thresholds, log_baseline_summary as print_baseline_summary
from .section4_anomalies import detect_volume_anomalies, classify_anomaly_patterns, print_anomaly_summary
from .section5_signal import calculate_signal_strength_metrics, assess_information_leakage_likelihood, log_signal_analysis_summary as print_signal_analysis_summary
from .section6_outcomes import measure_stock_performance_outcomes, calculate_risk_adjusted_returns, print_outcome_summary

def analyze_single_event_complete(event: LaunchEvent, analyzer) -> Optional[Dict[str, Any]]:
    """
    Full pipeline for one event:
    1) Collect data  2) Quality check  3) Baseline  4) Thresholds
    5) Anomalies     6) Signal analysis   7) Outcomes & risk-adjusted
    Returns a dict of all artifacts or None on failure.
    """
    print("\n" + "="*90)
    print(f"COMPREHENSIVE ANALYSIS: {event.company} — {event.name}")
    print(f"Announcement: {event.announcement} | Release: {event.release} | Earnings: {event.next_earnings}")
    print("="*90)

    # ---- Step 1: Data Collection (announcement-anchored windows) ----
    print("Step 1: Data Collection")
    data = collect_event_data(
        event,
        baseline_days=analyzer.baseline_days,
        signal_window_announce=analyzer.signal_window_announce,
        post_event_buffer_days=30
    )
    if data is None or data.empty:
        print("  ANALYSIS FAILED: Data collection unsuccessful")
        return None

    # ---- Step 2: Data Quality ----
    print("\nStep 2: Data Quality Validation")
    quality = validate_data_quality(data, event, academic_standards=True)
    if not quality.get("valid", False):
        print(f"  ANALYSIS FAILED: {quality.get('reason', 'Unknown quality failure')}")
        return None

    # ---- Step 3: Baseline ----
    print("\nStep 3: Baseline Analysis")
    bm = establish_baseline_metrics(
        data,
        event,
        baseline_days=analyzer.baseline_days,
        signal_window_announce=analyzer.signal_window_announce
    )
    if not bm:
        print("  ANALYSIS FAILED: Baseline analysis unsuccessful")
        return None

    th = calculate_volume_thresholds(bm, academic_validation=True)
    print_baseline_summary(event, bm, th)

    # ---- Step 4: Anomaly Detection ----
    print("\nStep 4: Anomaly Detection")
    anomalies = detect_volume_anomalies(
        data, event, bm, th, signal_window_announce=analyzer.signal_window_announce
    )
    patterns = classify_anomaly_patterns(anomalies)
    print_anomaly_summary(event, anomalies, patterns)

    # ---- Step 5: Signal Analysis ----
    print("\nStep 5: Signal Analysis")
    sig = calculate_signal_strength_metrics(
        anomalies, patterns, max_window_days=analyzer.signal_window_announce[1]
    )
    assess = assess_information_leakage_likelihood(
        sig, anomalies, event=event, max_window_days=analyzer.signal_window_announce[1]
    )
    print_signal_analysis_summary(event, sig, assess)

    # ---- Step 6: Outcomes ----
    print("\nStep 6: Outcome Measurement")
    om = measure_stock_performance_outcomes(data, event)
    if om:
        ra = calculate_risk_adjusted_returns(om, baseline_metrics=bm)
        print_outcome_summary(event, om, ra)
    else:
        ra = None
        print("  WARNING: Outcome measurement failed")

    # ---- Step 7: Package results ----
    results = {
        "event_info": {
            "name": event.name,
            "company": event.company,
            "ticker": event.ticker,
            "category": event.category,
            "announcement": str(event.announcement),
            "release": str(event.release),
            "next_earnings": str(event.next_earnings) if event.next_earnings else None,
        },
        "analysis_parameters": {
            "baseline_days": analyzer.baseline_days,
            "signal_window_announce": analyzer.signal_window_announce,
            "volume_threshold": getattr(analyzer, "volume_threshold", None),
        },
        "data_quality": quality,
        "baseline_metrics": bm,
        "volume_thresholds": th,
        "anomalies": anomalies,
        "pattern_analysis": patterns,
        "signal_metrics": sig,
        "leakage_assessment": assess,
        "outcome_metrics": om,
        "risk_adjusted_metrics": ra,
        "analysis_timestamp": datetime.now().isoformat(),
    }

    print(f"\n  ✅ ANALYSIS COMPLETE: {event.company} {event.name}")
    return results


def run_multi_event_analysis(analyzer) -> Dict[str, Any]:
    """
    Run the full pipeline across all events in analyzer.events (LaunchEvent list).
    Returns a dict keyed by event name → results.
    """
    print("PHASE 1: MULTI-EVENT VOLUME INFORMATION LEAKAGE ANALYSIS")
    print("="*96)
    print(f"Total events to analyze: {len(analyzer.events)}")

    all_results: Dict[str, Any] = {}
    ok, fail = 0, 0

    for i, event in enumerate(analyzer.events, 1):
        print(f"\n[EVENT {i}/{len(analyzer.events)}] {event.company} — {event.name}")
        try:
            res = analyze_single_event_complete(event, analyzer)
            if res:
                all_results[event.name] = res
                ok += 1
                print("  STATUS: SUCCESS")
            else:
                fail += 1
                print("  STATUS: FAILED")
        except Exception as e:
            fail += 1
            print(f"  STATUS: ERROR — {e}")

    print("\n" + "="*96)
    print("MULTI-EVENT ANALYSIS SUMMARY")
    print("="*96)
    print(f"Processed: {len(analyzer.events)}  |  Success: {ok}  |  Failed: {fail}")
    rate = (ok / len(analyzer.events) * 100) if analyzer.events else 0.0
    print(f"Success rate: {rate:.1f}%")
    return all_results


def generate_cross_event_summary(all_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Cross-event rollup with small-sample methods:
    - Bootstrap CIs for correlation (no SciPy needed)
    - Corrado-style rank Z test vs. 0 using error function
    - Company-level cross-sectional view
    """
    if not all_results:
        print("No results available for cross-event summary.")
        return None

    print("\nCROSS-EVENT SUMMARY ANALYSIS")
    print("="*36)
    print("Small-sample methods: Bootstrap CIs + nonparametric rank Z test")

    total_events = len(all_results)
    events_with_signals = sum(1 for r in all_results.values() if r.get("signal_metrics"))
    total_anomalies = sum(len(r.get("anomalies", [])) for r in all_results.values())

    signal_strengths, volume_ratios, signal_timings, stock_returns = [], [], [], []
    companies, categories = [], []

    for ev_name, r in all_results.items():
        sig = r.get("signal_metrics")
        om = r.get("outcome_metrics")
        ev = r.get("event_info", {})

        if sig:
            signal_strengths.append(sig.get("weighted_avg_z_score", np.nan))
            volume_ratios.append(sig.get("max_volume_ratio", np.nan))
            signal_timings.append(sig.get("avg_signal_timing", np.nan))
            companies.append(ev.get("company", ""))
            categories.append(ev.get("category", ""))

        if om and (om.get("release_to_earnings_return") is not None):
            stock_returns.append(om["release_to_earnings_return"])

    # Clean NaNs
    signal_strengths = [x for x in signal_strengths if np.isfinite(x)]
    volume_ratios = [x for x in volume_ratios if np.isfinite(x)]
    signal_timings = [x for x in signal_timings if np.isfinite(x)]
    stock_returns = [x for x in stock_returns if np.isfinite(x)]

    summary = {
        "total_events_analyzed": total_events,
        "events_with_significant_signals": events_with_signals,
        "signal_detection_rate_pct": (events_with_signals / total_events * 100.0) if total_events else 0.0,
        "total_anomalies_detected": total_anomalies,
        "avg_anomalies_per_event": (total_anomalies / total_events) if total_events else 0.0,
        "signal_strength_stats": {
            "mean": float(np.mean(signal_strengths)) if signal_strengths else 0.0,
            "median": float(np.median(signal_strengths)) if signal_strengths else 0.0,
            "max": float(np.max(signal_strengths)) if signal_strengths else 0.0,
            "std": float(np.std(signal_strengths)) if signal_strengths else 0.0,
        },
        "volume_ratio_stats": {
            "mean": float(np.mean(volume_ratios)) if volume_ratios else 1.0,
            "median": float(np.median(volume_ratios)) if volume_ratios else 1.0,
            "max": float(np.max(volume_ratios)) if volume_ratios else 1.0,
        },
        "timing_stats": {
            "mean": float(np.mean(signal_timings)) if signal_timings else 0.0,
            "earliest": float(np.max(signal_timings)) if signal_timings else 0.0,
            "latest": float(np.min(signal_timings)) if signal_timings else 0.0,
        },
        "return_stats": {
            "mean": float(np.mean(stock_returns)) if stock_returns else 0.0,
            "median": float(np.median(stock_returns)) if stock_returns else 0.0,
            "std": float(np.std(stock_returns)) if stock_returns else 0.0,
            "positive_rate_pct": (sum(1 for r in stock_returns if r > 0) / len(stock_returns) * 100.0) if stock_returns else 0.0,
        },
    }

    # ---- Bootstrap correlation (signal_strengths vs stock_returns) ----
    def bootstrap_corr(x, y, n=1000, cl=0.95):
        if len(x) != len(y) or len(x) < 3:
            return None
        x, y = np.array(x), np.array(y)
        corrs = []
        for _ in range(n):
            idx = np.random.randint(0, len(x), len(x))
            xs, ys = x[idx], y[idx]
            if xs.std() > 0 and ys.std() > 0:
                c = np.corrcoef(xs, ys)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        if not corrs:
            return None
        alpha = 1 - cl
        return {
            "correlation": float(np.mean(corrs)),
            "ci_lower": float(np.percentile(corrs, (alpha/2)*100)),
            "ci_upper": float(np.percentile(corrs, (1-alpha/2)*100)),
            "significant": not (np.percentile(corrs, (alpha/2)*100) <= 0 <= np.percentile(corrs, (1-alpha/2)*100)),
            "replications": n,
        }

    paired = min(len(signal_strengths), len(stock_returns))
    if paired >= 3:
        # Align by truncating to the shortest list
        bs = bootstrap_corr(signal_strengths[:paired], stock_returns[:paired])
        summary["bootstrap_correlation"] = bs
        if bs:
            print("\nBootstrap correlation (signal strength ↔ returns):")
            print(f"  r ≈ {bs['correlation']:.3f} | 95% CI [{bs['ci_lower']:.3f}, {bs['ci_upper']:.3f}] | significant: {bs['significant']}")
    else:
        summary["bootstrap_correlation"] = None
        print("\nBootstrap correlation: insufficient paired observations.")

    # ---- Corrado-style rank Z test vs. 0 (no SciPy) ----
    def corrado_rank_z(returns):
        # Rank each return among the sample; test mean rank vs midpoint
        n = len(returns)
        if n < 3:
            return None
        arr = np.array(returns)
        ranks = pd.Series(arr).rank(method="average").to_numpy()
        expected = (n + 1) / 2.0
        var = (n + 1) * (n - 1) / 12.0
        z = (ranks.mean() - expected) / sqrt(var / n)
        # two-tailed p using error function
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2))))
        return {"z": float(z), "p_value": float(p), "sig_5pct": p < 0.05, "sig_1pct": p < 0.01}

    if len(stock_returns) >= 3:
        cz = corrado_rank_z(stock_returns)
        summary["corrado_rank_test"] = cz
        if cz:
            print("Corrado rank Z (returns vs 0):")
            print(f"  Z = {cz['z']:.3f}, p = {cz['p_value']:.4f} | 5%: {cz['sig_5pct']}  1%: {cz['sig_1pct']}")
    else:
        summary["corrado_rank_test"] = None
        print("Corrado rank test: insufficient returns.")

    # ---- Company cross-section (means) ----
    if companies:
        cross = {}
        uniq = sorted(set(companies))
        for comp in uniq:
            idxs = [i for i, c in enumerate(companies) if c == comp]
            vals = [signal_strengths[i] for i in idxs if i < len(signal_strengths)]
            if len(vals) >= 2:
                cross[comp] = {
                    "event_count": len(vals),
                    "mean_signal_strength": float(np.mean(vals)),
                    "category": categories[idxs[0]] if idxs and idxs[0] < len(categories) else "",
                }
        summary["cross_section_company"] = cross if cross else None
        if cross:
            print("\nCross-section (company):")
            for comp, stats in cross.items():
                print(f"  {comp}: {stats['event_count']} events | mean signal {stats['mean_signal_strength']:.2f}σ")
    else:
        summary["cross_section_company"] = None

    print("\nAcademic validation checklist:")
    print("  • Bootstrap CIs applied")
    print("  • Nonparametric rank Z used")
    print("  • Cross-sectional lens included")

    return summary


# -------------------------
# Section 7 smoke test
# -------------------------
def smoke_test():
    """Run smoke test if analyzer is available"""
    if 'analyzer' not in locals():
        print("Analyzer not found. Run Section 1 first.")
        return
    
    print("\nSection 7 smoke test (single event)...")
    ev = next((e for e in analyzer.events if e.name == "iPhone 14"), analyzer.events[0])
    single_results = analyze_single_event_complete(ev, analyzer)
    if single_results:
        print("\nSection 7 smoke test (multi-event)...")
        all_results = run_multi_event_analysis(analyzer)
        _ = generate_cross_event_summary(all_results)
        print("\nSection 7 ready.")
    else:
        print("Single-event analysis failed — fix earlier sections and retry.")

if __name__ == "__main__":
    smoke_test()
