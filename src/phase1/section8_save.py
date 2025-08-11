# Phase 1: Volume Information Leakage Analysis
# Section 8: Execute Analysis and Save Results

import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def _py(obj):
    """Convert numpy scalars/arrays to pure Python for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_py(v) for v in obj]
    return obj

def save_analysis_results(all_results: Dict[str, Any],
                          summary_stats: Optional[Dict[str, Any]],
                          analyzer) -> Dict[str, Any]:
    """
    Save comprehensive analysis artifacts (pickle, CSV, JSON).
    Compatible with LaunchEvent-based pipeline.
    """
    _ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------- Build complete package (pickle) ----------
    # Pull analyzer event metadata safely (LaunchEvent objects)
    events_analyzed = [e.name for e in getattr(analyzer, "events", [])]
    companies_covered = sorted(set(e.company for e in getattr(analyzer, "events", [])))
    product_categories = sorted(set(e.category for e in getattr(analyzer, "events", [])))

    complete_package = {
        "analysis_results": all_results,
        "summary_statistics": summary_stats,
        "analysis_parameters": {
            "baseline_days": getattr(analyzer, "baseline_days", None),
            "signal_window_announce": getattr(analyzer, "signal_window_announce", None),
            "thresholds_per_event": {k: v.get("volume_thresholds") for k, v in all_results.items()},
            "events_analyzed": events_analyzed,
            "companies_covered": companies_covered,
            "product_categories": product_categories,
        },
        "analysis_timestamp": datetime.now().isoformat(),
    }

    pkl_path = f"results/phase1_complete_results_{ts}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(complete_package, f)

    # ---------- Build summary DataFrame (CSV) ----------
    rows = []
    for ev_name, res in all_results.items():
        ev = res.get("event_info", {})
        bm = res.get("baseline_metrics", {}) or {}
        sig = res.get("signal_metrics", {}) or {}
        om = res.get("outcome_metrics", {}) or {}
        assess = res.get("leakage_assessment", {}) or {}

        row = {
            # Event info
            "event_name": ev_name,
            "company": ev.get("company"),
            "product_category": ev.get("category"),
            "announcement_date": ev.get("announcement"),
            "release_date": ev.get("release"),
            "earnings_date": ev.get("next_earnings"),

            # Baseline
            "baseline_volume_mean": bm.get("volume_mean"),
            "baseline_volume_std": bm.get("volume_std"),
            "baseline_cv": bm.get("coefficient_of_variation"),

            # Anomaly counts
            "total_anomalies": len(res.get("anomalies", [])),
        }

        # Signal metrics (guard for None)
        if sig:
            row.update({
                "signal_strength_composite": sig.get("composite_signal_score", 0.0),
                "signal_confidence": sig.get("signal_confidence", 0.0),
                "avg_z_score": sig.get("avg_z_score", 0.0),
                "weighted_avg_z_score": sig.get("weighted_avg_z_score", 0.0),
                "max_z_score": sig.get("max_z_score", 0.0),
                "max_volume_ratio": sig.get("max_volume_ratio", 1.0),
                "avg_signal_timing": sig.get("avg_signal_timing", 0.0),
                "signal_persistence": sig.get("signal_persistence", 0.0),
                "extreme_signals": sig.get("extreme_signals", 0),
                "spike_signals": sig.get("spike_signals", 0),
                "unusual_signals": sig.get("unusual_signals", 0),
                "exceeds_screening_threshold": sig.get("weighted_avg_z_score", 0) > 1.645,
                "exceeds_statistical_threshold": sig.get("weighted_avg_z_score", 0) > 2.326,
                "exceeds_conservative_threshold": sig.get("weighted_avg_z_score", 0) > 2.576,
                "optimal_window_anomalies": sum(1 for a in res.get("anomalies", []) if a.get("in_optimal_window")),
            })
        else:
            row.update({
                "signal_strength_composite": 0.0,
                "signal_confidence": 0.0,
                "avg_z_score": 0.0,
                "weighted_avg_z_score": 0.0,
                "max_z_score": 0.0,
                "max_volume_ratio": 1.0,
                "avg_signal_timing": 0.0,
                "signal_persistence": 0.0,
                "extreme_signals": 0,
                "spike_signals": 0,
                "unusual_signals": 0,
                "exceeds_screening_threshold": False,
                "exceeds_statistical_threshold": False,
                "exceeds_conservative_threshold": False,
                "optimal_window_anomalies": 0,
            })

        # Leakage assessment
        if assess:
            row.update({
                "leakage_likelihood": assess.get("leakage_likelihood", "None"),
                "leakage_score": assess.get("leakage_score", 0),
                "evidence_strength": assess.get("evidence_strength", "No Evidence"),
            })
        else:
            row.update({
                "leakage_likelihood": "None",
                "leakage_score": 0,
                "evidence_strength": "No Evidence",
            })

        # Outcomes
        if om:
            row.update({
                "announcement_price": om.get("announcement_price"),
                "release_price": om.get("release_price"),
                "earnings_price": om.get("earnings_price"),
                "announcement_to_release_return": om.get("announcement_to_release_return"),
                "release_to_earnings_return": om.get("release_to_earnings_return"),
                "full_period_return": om.get("full_period_return"),
                "return_1d": om.get("return_1d"),
                "return_5d": om.get("return_5d"),
                "release_volume_ratio": om.get("release_volume_ratio"),
            })
        else:
            row.update({
                "announcement_price": None,
                "release_price": None,
                "earnings_price": None,
                "announcement_to_release_return": None,
                "release_to_earnings_return": None,
                "full_period_return": None,
                "return_1d": None,
                "return_5d": None,
                "release_volume_ratio": None,
            })

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = f"results/phase1_summary_{ts}.csv"
    summary_df.to_csv(csv_path, index=False)

    # ---------- Summary stats JSON (matches Section 7 keys) ----------
    json_path = f"results/phase1_summary_stats_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(_py(summary_stats) if summary_stats else {}, f, indent=2)

    print("\nResults saved:")
    print(f"  Complete pickle : {pkl_path}")
    print(f"  Summary CSV     : {csv_path}")
    print(f"  Summary JSON    : {json_path}")

    return {
        "pickle_file": pkl_path,
        "csv_file": csv_path,
        "json_file": json_path,
        "summary_dataframe": summary_df,
    }


def create_summary_visualization(summary_df: pd.DataFrame, save_plots: bool = True) -> Optional[str]:
    """
    Simple 2x2 dashboard: signal scores, anomaly counts, signal vs returns, leakage scores.
    """
    if summary_df is None or summary_df.empty:
        print("No data available for visualization.")
        return None

    _ensure_dirs()
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 1: Volume Information Leakage — Summary", fontsize=15, fontweight="bold")

    events = summary_df["event_name"].astype(str)

    # 1) Signal Strength by Event
    ax1 = axes[0, 0]
    sig = summary_df.get("signal_strength_composite", pd.Series([0]*len(summary_df)))
    ax1.bar(events, sig, alpha=0.8)
    ax1.set_title("Signal Strength by Event", fontweight="bold")
    ax1.set_ylabel("Composite Signal Score")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    # 2) Volume Anomalies by Event
    ax2 = axes[0, 1]
    anoms = summary_df.get("total_anomalies", pd.Series([0]*len(summary_df)))
    ax2.bar(events, anoms, alpha=0.8)
    ax2.set_title("Volume Anomalies Detected", fontweight="bold")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    # 3) Signal Strength vs Returns
    ax3 = axes[1, 0]
    rets = summary_df.get("release_to_earnings_return", pd.Series([np.nan]*len(summary_df)))
    mask = sig.notna() & rets.notna()
    if mask.sum() >= 1:
        ax3.scatter(sig[mask], rets[mask], s=80, alpha=0.7)
        if mask.sum() >= 3:
            z = np.polyfit(sig[mask], rets[mask], 1)
            p = np.poly1d(z)
            xs = np.linspace(sig[mask].min(), sig[mask].max(), 100)
            ax3.plot(xs, p(xs), "--", linewidth=2)
    ax3.set_title("Signal Strength vs Returns", fontweight="bold")
    ax3.set_xlabel("Composite Signal Score")
    ax3.set_ylabel("Release→Earnings Return (%)")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color="k", linewidth=1, alpha=0.25)
    ax3.axvline(0, color="k", linewidth=1, alpha=0.25)

    # 4) Leakage Scores
    ax4 = axes[1, 1]
    lks = summary_df.get("leakage_score", pd.Series([0]*len(summary_df)))
    ax4.bar(events, lks, alpha=0.7)
    ax4.set_title("Information Leakage Assessment Scores", fontweight="bold")
    ax4.set_ylabel("Score")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = None
    if save_plots:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"results/phase1_analysis_plots_{ts}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Visualization: {out_path}")
    plt.show()
    return out_path


# -------------------------
# Optional: one-cell runner
# -------------------------
if __name__ == "__main__":
    print("EXECUTING PHASE 1 (Section 8)")
    print("="*60)

    # Run full multi-event analysis
    all_results = run_multi_event_analysis(analyzer)

    if all_results:
        # Cross-event summary (Section 7)
        summary_stats = generate_cross_event_summary(all_results)

        # Save artifacts (pickle/CSV/JSON)
        saved = save_analysis_results(all_results, summary_stats, analyzer)

        # Plots
        print("\nRendering visual summary…")
        viz_path = create_summary_visualization(saved["summary_dataframe"])

        # Paper-friendly recap (key numbers)
        if summary_stats:
            print("\nPHASE 1 — Key Figures")
            print("---------------------")
            print(f"Events analyzed: {summary_stats.get('total_events_analyzed', 0)}")
            print(f"Detection rate: {summary_stats.get('signal_detection_rate_pct', 0):.1f}%")
            bc = summary_stats.get("bootstrap_correlation")
            if bc:
                print(f"Bootstrap corr: r≈{bc.get('correlation', 0):.3f} "
                      f"[{bc.get('ci_lower', 0):.3f}, {bc.get('ci_upper', 0):.3f}] "
                      f"sig={bc.get('significant', False)}")
            cz = summary_stats.get("corrado_rank_test")
            if cz:
                print(f"Corrado rank Z: Z={cz.get('z', 0):.3f}, p={cz.get('p_value', 1.0):.4f} "
                      f"5%={cz.get('sig_5pct', False)} 1%={cz.get('sig_1pct', False)}")

        print("\n✅ Section 8 complete.")
    else:
        print("❗No successful analyses completed; nothing to save.")
