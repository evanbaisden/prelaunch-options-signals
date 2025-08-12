import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Config:
    events_csv: str
    raw_data_dir: str
    results_dir: str
    baseline_days: int
    z_thresholds: List[float]
    event_windows: List[str]
    seed: int
    log_level: str


def get_config() -> Config:
    """Load configuration from environment variables with defaults."""
    
    # Helper function to parse list of floats
    def parse_float_list(value: str) -> List[float]:
        return [float(x.strip()) for x in value.strip('[]').split(',')]
    
    # Helper function to parse list of strings
    def parse_string_list(value: str) -> List[str]:
        return [x.strip().strip('"\'') for x in value.strip('[]').split(',')]
    
    return Config(
        events_csv=os.getenv('EVENTS_CSV', 'data/processed/events_master.csv'),
        raw_data_dir=os.getenv('RAW_DATA_DIR', 'data/raw'),
        results_dir=os.getenv('RESULTS_DIR', 'results'),
        baseline_days=int(os.getenv('BASELINE_DAYS', '60')),
        z_thresholds=parse_float_list(os.getenv('Z_THRESHOLDS', '[1.645,2.326,2.576]')),
        event_windows=parse_string_list(os.getenv('EVENT_WINDOWS', '["-1:+1","-2:+2","-5:+5"]')),
        seed=int(os.getenv('SEED', '42')),
        log_level=os.getenv('LOG_LEVEL', 'INFO')
    )