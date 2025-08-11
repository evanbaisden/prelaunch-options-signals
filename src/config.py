"""
Configuration management for prelaunch options signals analysis.
Centralizes environment variables and CLI configuration.
"""
from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class Config:
    """Central configuration for the analysis pipeline."""
    
    # Analysis parameters
    baseline_days: int = int(os.getenv("BASELINE_DAYS", "60"))
    signal_window_announce: tuple[int, int] = tuple(
        map(int, os.getenv("SIGNAL_WINDOW_ANNOUNCE", "5,20").split(","))
    )
    signal_window_release: tuple[int, int] = tuple(
        map(int, os.getenv("SIGNAL_WINDOW_RELEASE", "5,10").split(","))
    )
    
    # Statistical thresholds
    z_threshold_low: float = float(os.getenv("Z_THRESHOLD_LOW", "1.645"))
    z_threshold_med: float = float(os.getenv("Z_THRESHOLD_MED", "2.326"))
    z_threshold_high: float = float(os.getenv("Z_THRESHOLD_HIGH", "2.576"))
    z_threshold_extreme: float = float(os.getenv("Z_THRESHOLD_EXTREME", "5.0"))
    
    # Directories
    data_dir: str = os.getenv("DATA_DIR", "data")
    data_raw_dir: str = os.getenv("DATA_RAW_DIR", "data/raw")
    data_processed_dir: str = os.getenv("DATA_PROCESSED_DIR", "data/processed")
    results_dir: str = os.getenv("RESULTS_DIR", "results")
    
    # Output settings
    save_plots: bool = os.getenv("SAVE_PLOTS", "true").lower() == "true"
    plot_dpi: int = int(os.getenv("PLOT_DPI", "300"))
    plot_format: str = os.getenv("PLOT_FORMAT", "png")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(message)s")
    
    # External APIs (Phase 2)
    polygon_api_key: Optional[str] = os.getenv("POLYGON_API_KEY")
    alpha_vantage_api_key: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    yahoo_finance_cookies: Optional[str] = os.getenv("YAHOO_FINANCE_COOKIES")
    
    @property
    def z_thresholds(self) -> dict[str, float]:
        """Z-score thresholds for statistical significance testing."""
        return {
            'low': self.z_threshold_low,      # 95th percentile (one-tailed)
            'med': self.z_threshold_med,      # 99th percentile (one-tailed) 
            'high': self.z_threshold_high,    # 99% two-tailed / 99.5% one-tailed
            'extreme': self.z_threshold_extreme  # Extreme outliers
        }


# Global configuration instance
cfg = Config()