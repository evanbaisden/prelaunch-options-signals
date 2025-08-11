"""
Outcome compilation and results formatting for prelaunch options analysis.
Combines signals, baseline metrics, and anomalies into final analysis results.
"""
import pandas as pd
import json
from datetime import datetime, date
from typing import List, Dict, Optional
from pathlib import Path

from ..common.types import LaunchEvent, AnalysisResults, BaselineMetrics, SignalMetrics, StockData
from ..common.utils import validate_stock_data, ensure_directory, format_percentage
from ..config import Config


def compile_analysis_results(
    event: LaunchEvent,
    stock_data: StockData,
    baseline: BaselineMetrics,
    announcement_signal: Optional[SignalMetrics] = None,
    release_signal: Optional[SignalMetrics] = None,
    period_averages: Optional[Dict[str, float]] = None
) -> AnalysisResults:
    """
    Compile all analysis components into final results object.
    
    Args:
        event: LaunchEvent with product details
        stock_data: StockData with quality metrics
        baseline: BaselineMetrics from pre-announcement period
        announcement_signal: SignalMetrics for announcement
        release_signal: SignalMetrics for release  
        period_averages: Period-wise return/volume averages
        
    Returns:
        AnalysisResults object with complete analysis
    """
    # Calculate data quality score
    quality_checks = stock_data.data_quality_checks
    quality_score = sum(quality_checks.values()) / len(quality_checks)
    
    # Create results object
    results = AnalysisResults(
        event=event,
        baseline=baseline,
        announcement_signal=announcement_signal,
        release_signal=release_signal,
        analysis_timestamp=datetime.now(),
        data_quality_score=quality_score,
        notes=""
    )
    
    # Add period averages if provided
    if period_averages:
        results.pre_announce_avg_return = period_averages.get('pre_announce_avg_return', 0.0)
        results.announce_to_release_avg_return = period_averages.get('announce_to_release_avg_return', 0.0)  
        results.post_release_avg_return = period_averages.get('post_release_avg_return', 0.0)
        results.pre_announce_avg_volume = period_averages.get('pre_announce_avg_volume', 0.0)
        results.announce_to_release_avg_volume = period_averages.get('announce_to_release_avg_volume', 0.0)
        results.post_release_avg_volume = period_averages.get('post_release_avg_volume', 0.0)
    
    # Add quality notes
    failed_checks = [check for check, passed in quality_checks.items() if not passed]
    if failed_checks:
        results.notes = f"Data quality issues: {', '.join(failed_checks)}"
    
    return results


def export_results_to_csv(
    results_list: List[AnalysisResults],
    output_path: Path,
    include_metadata: bool = True
) -> Tuple[Path, Optional[Path]]:
    """
    Export analysis results to CSV format with optional metadata.
    
    Args:
        results_list: List of AnalysisResults objects
        output_path: Output file path (without extension)
        include_metadata: Whether to create metadata JSON file
        
    Returns:
        Tuple of (CSV path, metadata JSON path)
    """
    config = Config()
    
    # Convert results to dictionary format
    rows = []
    for result in results_list:
        row_dict = result.to_dict()
        rows.append(row_dict)
    
    # Create DataFrame and save CSV
    df = pd.DataFrame(rows)
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    
    metadata_path = None
    if include_metadata:
        # Create metadata
        metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_products_analyzed': len(results_list),
            'companies': list(set(r.event.company for r in results_list)),
            'date_range': {
                'earliest_announcement': min(r.event.announcement for r in results_list).isoformat(),
                'latest_release': max(r.event.release for r in results_list).isoformat()
            },
            'parameters': {
                'baseline_days': config.baseline_days,
                'signal_window_announce': config.signal_window_announce,
                'signal_window_release': config.signal_window_release,
                'z_thresholds': {
                    'low': config.z_threshold_low,
                    'med': config.z_threshold_med,
                    'high': config.z_threshold_high,
                    'extreme': config.z_threshold_extreme
                }
            },
            'data_quality': {
                'avg_quality_score': sum(r.data_quality_score for r in results_list) / len(results_list),
                'products_with_issues': len([r for r in results_list if r.notes])
            },
            'calculation_definitions': {
                'announcement_5day_return': 'Cumulative return from close[t-5] to close[t+0] on announcement day',
                'release_5day_return': 'Cumulative return from close[t-5] to close[t+0] on release day',
                'volume_spike_pct': '(volume[period] / baseline_mean) - 1, where baseline = 20-day MA',
                'pre_announce_return': 'Average daily return 60 trading days before announcement',
                'announce_to_release_return': 'Average daily return from announcement to release day',
                'post_release_return': 'Average daily return 30 trading days after release'
            }
        }
        
        # Save metadata JSON
        metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return csv_path, metadata_path


def generate_summary_statistics(results_list: List[AnalysisResults]) -> Dict[str, any]:
    """
    Generate summary statistics across all analyzed products.
    
    Args:
        results_list: List of AnalysisResults objects
        
    Returns:
        Dictionary with summary statistics
    """
    config = Config()
    
    summary = {
        'total_products': len(results_list),
        'companies': {},
        'signal_statistics': {},
        'data_quality': {},
        'time_distribution': {}
    }
    
    # Company breakdown
    for result in results_list:
        company = result.event.company
        if company not in summary['companies']:
            summary['companies'][company] = 0
        summary['companies'][company] += 1
    
    # Signal statistics
    announcement_returns = [r.announcement_signal.price_5day_return 
                          for r in results_list if r.announcement_signal]
    release_returns = [r.release_signal.price_5day_return 
                      for r in results_list if r.release_signal]
    
    announcement_z_scores = [r.announcement_signal.volume_z_score 
                           for r in results_list if r.announcement_signal]
    release_z_scores = [r.release_signal.volume_z_score 
                       for r in results_list if r.release_signal]
    
    summary['signal_statistics'] = {
        'announcement_signals': {
            'count': len(announcement_returns),
            'avg_return': sum(announcement_returns) / len(announcement_returns) if announcement_returns else 0,
            'avg_z_score': sum(announcement_z_scores) / len(announcement_z_scores) if announcement_z_scores else 0,
            'significant_volume_spikes': len([z for z in announcement_z_scores if abs(z) >= config.z_threshold_low])
        },
        'release_signals': {
            'count': len(release_returns),
            'avg_return': sum(release_returns) / len(release_returns) if release_returns else 0,
            'avg_z_score': sum(release_z_scores) / len(release_z_scores) if release_z_scores else 0,
            'significant_volume_spikes': len([z for z in release_z_scores if abs(z) >= config.z_threshold_low])
        }
    }
    
    # Data quality
    quality_scores = [r.data_quality_score for r in results_list]
    summary['data_quality'] = {
        'avg_score': sum(quality_scores) / len(quality_scores),
        'min_score': min(quality_scores),
        'products_with_issues': len([r for r in results_list if r.notes]),
        'high_quality_products': len([s for s in quality_scores if s >= 0.9])
    }
    
    # Time distribution
    years = [r.event.announcement.year for r in results_list]
    year_counts = {}
    for year in years:
        year_counts[year] = year_counts.get(year, 0) + 1
    
    summary['time_distribution'] = {
        'year_counts': year_counts,
        'earliest_year': min(years),
        'latest_year': max(years)
    }
    
    return summary


def create_analysis_report(
    results_list: List[AnalysisResults],
    summary_stats: Dict[str, any],
    output_path: Path
) -> Path:
    """
    Create a markdown analysis report with key findings.
    
    Args:
        results_list: List of AnalysisResults objects
        summary_stats: Summary statistics dictionary
        output_path: Output path for report
        
    Returns:
        Path to created report file
    """
    report_lines = [
        "# Prelaunch Options Signals - Phase 1 Analysis Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Products Analyzed: {summary_stats['total_products']}",
        "\n## Executive Summary",
        "\nThis report presents the analysis of stock price and volume patterns around major technology product launches.",
        f"The analysis covers {summary_stats['total_products']} product launches across "
        f"{len(summary_stats['companies'])} companies from {summary_stats['time_distribution']['earliest_year']} "
        f"to {summary_stats['time_distribution']['latest_year']}.",
        "\n## Company Breakdown"
    ]
    
    for company, count in summary_stats['companies'].items():
        report_lines.append(f"- **{company}**: {count} products")
    
    report_lines.extend([
        "\n## Signal Analysis Results",
        "\n### Announcement Signals",
        f"- Products with announcement data: {summary_stats['signal_statistics']['announcement_signals']['count']}",
        f"- Average 5-day return: {format_percentage(summary_stats['signal_statistics']['announcement_signals']['avg_return'])}",
        f"- Average volume Z-score: {summary_stats['signal_statistics']['announcement_signals']['avg_z_score']:.2f}",
        f"- Significant volume spikes (Z ≥ 1.645): {summary_stats['signal_statistics']['announcement_signals']['significant_volume_spikes']}",
        "\n### Release Signals",
        f"- Products with release data: {summary_stats['signal_statistics']['release_signals']['count']}",  
        f"- Average 5-day return: {format_percentage(summary_stats['signal_statistics']['release_signals']['avg_return'])}",
        f"- Average volume Z-score: {summary_stats['signal_statistics']['release_signals']['avg_z_score']:.2f}",
        f"- Significant volume spikes (Z ≥ 1.645): {summary_stats['signal_statistics']['release_signals']['significant_volume_spikes']}",
        "\n## Data Quality Assessment",
        f"- Average data quality score: {summary_stats['data_quality']['avg_score']:.2f}/1.00",
        f"- Products with quality issues: {summary_stats['data_quality']['products_with_issues']}",
        f"- High-quality datasets (≥0.9): {summary_stats['data_quality']['high_quality_products']}",
        "\n## Individual Product Results",
        "\n| Product | Company | Ann. Return | Ann. Z-Score | Release Return | Release Z-Score |",
        "|---------|---------|-------------|--------------|----------------|-----------------|"
    ])
    
    # Add individual results table
    for result in results_list:
        ann_return = format_percentage(result.announcement_signal.price_5day_return) if result.announcement_signal else "N/A"
        ann_z = f"{result.announcement_signal.volume_z_score:.2f}" if result.announcement_signal else "N/A"
        rel_return = format_percentage(result.release_signal.price_5day_return) if result.release_signal else "N/A" 
        rel_z = f"{result.release_signal.volume_z_score:.2f}" if result.release_signal else "N/A"
        
        report_lines.append(f"| {result.event.name} | {result.event.company} | {ann_return} | {ann_z} | {rel_return} | {rel_z} |")
    
    report_lines.extend([
        "\n## Methodology Notes",
        "- **Baseline Period**: 60 trading days before announcement",
        "- **Signal Windows**: 5 days before event for price returns, 5-20 days for volume analysis",
        "- **Volume Anomalies**: Detected using Z-score method with 20-day rolling baseline",
        "- **Statistical Significance**: Z-scores ≥1.645 (95th percentile, one-tailed test)",
        "\n## Limitations",
        "- Analysis limited to daily OHLCV data; intraday patterns not captured",
        "- Volume analysis uses simple rolling mean baseline; more sophisticated methods could improve accuracy",
        "- Sample size may be insufficient for strong statistical conclusions",
        "- No adjustment for overall market conditions or sector performance",
        "\n---",
        f"*Report generated by Phase 1 Analysis Engine v1.0*"
    ])
    
    # Write report
    report_path = output_path.with_suffix('.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return report_path