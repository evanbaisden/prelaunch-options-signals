# =========================
# Phase 1: Volume Information Leakage Analysis
# Section 5: Signal Analysis (Clean Log Format)
# =========================

from typing import Dict, Any, Optional, List
import numpy as np
import logging

from .section1_setup import LaunchEvent

def _safe_get(d: Dict[str, Any], key: str, default=0):
    return d.get(key, default) if isinstance(d, dict) else default

def calculate_signal_strength_metrics(
    anomalies: List[Dict[str, Any]],
    pattern_analysis: Dict[str, Any],
    max_window_days: int = 20
) -> Optional[Dict[str, Any]]:
    """
    Aggregate anomaly-level features into signal strength metrics and a composite score.
    Weights anomalies closer to the announcement more heavily.
    """
    if not anomalies or _safe_get(pattern_analysis, "total_anomalies", 0) == 0:
        return None

    z_scores = [float(a.get("z_score", np.nan)) for a in anomalies]
    z_scores = [z for z in z_scores if np.isfinite(z)]
    vol_ratios = [float(a.get("ratio_to_baseline", np.nan)) for a in anomalies]
    vol_ratios = [r for r in vol_ratios if np.isfinite(r)]
    timing_vals = [int(a.get("days_before_announcement", 0)) for a in anomalies]

    if not z_scores or not vol_ratios or not timing_vals:
        return None

    max_days = max(int(max_window_days), 1)
    time_weights = [max((max_days - d + 1), 1) / max_days for d in timing_vals]

    weighted_avg_z = float(np.average(z_scores, weights=time_weights))
    weighted_avg_ratio = float(np.average(vol_ratios, weights=time_weights))

    t_max, t_min = max(timing_vals), min(timing_vals)
    timing_range = t_max - t_min if len(timing_vals) > 1 else 0
    signal_persistence = (timing_range / t_max) if t_max > 0 else 0.0
    total_signal_days = len(anomalies)
    total_possible_days = max(timing_range + 1, 1)
    signal_concentration = total_signal_days / total_possible_days

    # Composite score (bounded in [0,1])
    intensity_score = min(weighted_avg_z / 3.0, 1.0) * 0.4
    frequency_score = min(total_signal_days / 10.0, 1.0) * 0.3
    closest = float(min(timing_vals))
    timing_score = (1.0 - min(closest / max_days, 1.0)) * 0.3
    composite_score = float(intensity_score + frequency_score + timing_score)

    sev = _safe_get(pattern_analysis, "severity_distribution", {})
    extreme_cnt      = int(sev.get("extreme", 0))
    spike_cnt        = int(sev.get("spike", 0))
    unusual_cnt      = int(sev.get("unusual", 0))
    moderate_cnt     = int(sev.get("moderate", 0))
    notable_cnt      = int(sev.get("notable", 0))
    significant_cnt  = int(sev.get("significant", 0))
    highly_sig_cnt   = int(sev.get("highly_significant", 0))

    cluster_info = _safe_get(pattern_analysis, "clustering", {})
    sizes = cluster_info.get("cluster_sizes", [])
    max_cluster_size = int(max(sizes)) if sizes else 0

    return {
        "total_anomaly_days": total_signal_days,

        "avg_z_score": float(np.mean(z_scores)),
        "weighted_avg_z_score": weighted_avg_z,
        "max_z_score": float(np.max(z_scores)),
        "median_z_score": float(np.median(z_scores)),

        "avg_volume_ratio": float(np.mean(vol_ratios)),
        "weighted_avg_volume_ratio": weighted_avg_ratio,
        "max_volume_ratio": float(np.max(vol_ratios)),
        "median_volume_ratio": float(np.median(vol_ratios)),

        "earliest_signal_days": int(t_max),
        "latest_signal_days": int(t_min),
        "avg_signal_timing": float(np.mean(timing_vals)),
        "signal_timing_std": float(np.std(timing_vals)),
        "signal_persistence": float(signal_persistence),
        "signal_concentration": float(signal_concentration),

        "extreme_signals": extreme_cnt,
        "spike_signals": spike_cnt,
        "unusual_signals": unusual_cnt,
        "moderate_signals": moderate_cnt,
        "notable_signals": notable_cnt,
        "significant_signals": significant_cnt,
        "highly_significant_signals": highly_sig_cnt,

        "composite_signal_score": composite_score,
        "signal_confidence": float(min(composite_score * 1.2, 1.0)),

        "clustering_present": bool(cluster_info.get("total_clusters", 0) > 0),
        "max_cluster_size": max_cluster_size,
    }

def assess_information_leakage_likelihood(
    signal_metrics: Optional[Dict[str, Any]],
    anomalies: List[Dict[str, Any]],
    event: Optional[LaunchEvent] = None,
    max_window_days: int = 20
) -> Dict[str, Any]:
    """
    Score leakage likelihood using academic thresholds and simple contextual priors.
    """
    if not signal_metrics:
        return {
            "leakage_likelihood": "None",
            "confidence_level": "N/A",
            "evidence_strength": "No Evidence",
            "leakage_score": 0,
            "max_possible_score": 15,
            "evidence_factors": [],
            "academic_factors": [],
            "regulatory_context": {},
            "academic_assessment": "No anomalies",
            "assessment_summary": "No volume anomalies detected",
        }

    reg_ctx = {
        "Apple": {
            "precedent": "SEC v. Gene Levoff",
            "vulnerability": "High — consumer product secrecy; broad access",
            "pattern": "Annual cadence expands leak window",
        },
        "NVIDIA": {
            "precedent": "SEC investigations involving employees/partners",
            "vulnerability": "High — supplier/manufacturer touchpoints",
            "pattern": "Specs leak closer to event via technical channels",
        },
        "Microsoft": {
            "precedent": "SEC v. Brian Jorgenson",
            "vulnerability": "Medium — partner networks/platform deals",
            "pattern": "More contained dissemination vs consumer hardware",
        },
    }
    company = event.company if event else "Unknown"
    reg = reg_ctx.get(company, {
        "precedent": "General SEC enforcement in tech",
        "vulnerability": "Sector-standard risks",
        "pattern": "Leaks often cluster 1–3 weeks pre-event",
    })

    score = 0
    evidence_factors, academic_factors = [], []

    waz = signal_metrics["weighted_avg_z_score"]
    if waz > 2.576:
        score += 4
        evidence_factors.append("Weighted avg Z > 2.576 (99.5%)")
        academic_factors.append("Exceeds conservative academic threshold (2.576σ)")
    elif waz > 2.326:
        score += 3
        evidence_factors.append("Weighted avg Z > 2.326 (99%)")
        academic_factors.append("Exceeds primary academic threshold (2.326σ)")
    elif waz > 1.645:
        score += 2
        evidence_factors.append("Weighted avg Z > 1.645 (95%)")
        academic_factors.append("Exceeds screening threshold (1.645σ)")

    in_optimal = [a for a in anomalies if a.get("in_optimal_window", False)]
    if len(in_optimal) >= 3:
        score += 3
        evidence_factors.append("≥3 signals in 5–20d window")
        academic_factors.append("Matches pre-announcement leak band (5–20d)")
    elif len(in_optimal) >= 1:
        score += 2
        evidence_factors.append("Signals in 5–20d window")

    if signal_metrics["total_anomaly_days"] >= 5:
        score += 2
        evidence_factors.append("Frequent signals (≥5 days)")
    elif signal_metrics["total_anomaly_days"] >= 3:
        score += 1
        evidence_factors.append("Multiple anomaly days (≥3)")

    if signal_metrics.get("extreme_signals", 0) > 0 or signal_metrics.get("highly_significant_signals", 0) > 0:
        score += 3
        evidence_factors.append("Extreme/highly-significant spikes present")
        academic_factors.append("Consistent with informed trading patterns")

    cat = event.category if event else ""
    if "Consumer" in cat or "Semiconductor" in cat:
        score += 1
        evidence_factors.append(f"Category vulnerability: {cat}")

    if company in reg_ctx:
        score += 1
        evidence_factors.append(f"SEC precedent exists for {company}")

    if signal_metrics.get("signal_persistence", 0) > 0.5:
        score += 1
        evidence_factors.append("Signals persisted across the window")

    if score >= 10:
        likelihood, conf, strength, acad = "Very High", "High", "Strong Evidence", "Pattern aligns with documented cases"
    elif score >= 8:
        likelihood, conf, strength, acad = "High", "High", "Moderate–Strong Evidence", "Multiple academic indicators"
    elif score >= 6:
        likelihood, conf, strength, acad = "Moderate", "Medium", "Moderate Evidence", "Some academic indicators"
    elif score >= 3:
        likelihood, conf, strength, acad = "Low", "Medium", "Weak Evidence", "Limited academic validation"
    else:
        likelihood, conf, strength, acad = "Very Low", "Low", "Minimal Evidence", "Insufficient academic indicators"

    return {
        "leakage_likelihood": likelihood,
        "confidence_level": conf,
        "evidence_strength": strength,
        "leakage_score": int(score),
        "max_possible_score": 15,
        "evidence_factors": evidence_factors,
        "academic_factors": academic_factors,
        "regulatory_context": reg,
        "academic_assessment": acad,
        "assessment_summary": f"{strength} of leakage based on {len(evidence_factors)} factors (pre-announcement window)",
    }

def log_signal_analysis_summary(
    event: LaunchEvent,
    signal_metrics: Optional[Dict[str, Any]],
    leakage_assessment: Dict[str, Any]
) -> None:
    if not signal_metrics:
        logging.info(f"Signal Analysis — {event.company} {event.name}")
        logging.info("No significant signals detected.")
        logging.info(f"Information leakage likelihood: {leakage_assessment['leakage_likelihood']}")
        return

    logging.info(f"Signal Analysis — {event.company} {event.name}")
    logging.info(
        "Signal Strength | Composite: %.3f | Confidence: %.3f | Weighted avg Z: %.2f | Max vol ratio: %.2fx | "
        "Timing range: %dd–%dd before announcement",
        signal_metrics["composite_signal_score"],
        signal_metrics["signal_confidence"],
        signal_metrics["weighted_avg_z_score"],
        signal_metrics["max_volume_ratio"],
        signal_metrics["latest_signal_days"],
        signal_metrics["earliest_signal_days"],
    )

    logging.info(
        "Leakage Assessment | Likelihood: %s | Confidence: %s | Evidence: %s | Score: %d/%d",
        leakage_assessment["leakage_likelihood"],
        leakage_assessment["confidence_level"],
        leakage_assessment["evidence_strength"],
        leakage_assessment["leakage_score"],
        leakage_assessment["max_possible_score"],
    )
    if leakage_assessment.get("evidence_factors"):
        for f in leakage_assessment["evidence_factors"]:
            logging.info("Factor: %s", f)

