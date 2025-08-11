from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Dict, Any, List

from config.settings import DATA_PROVIDER
from data.providers.polygon_client import PolygonOptionsClient

@dataclass(frozen=True)
class OptionsPipelineConfig:
    vol_baseline_days: int = 60        # rolling baseline for chain-level volume z-scores
    chain_expiry_window_days: int = 45 # only include expiries within +/- this window
    max_contracts: int = 2000          # safety cap

class OptionsPipeline:
    def __init__(self, cfg: OptionsPipelineConfig = OptionsPipelineConfig()):
        self.cfg = cfg
        if DATA_PROVIDER != "polygon":
            raise ValueError("Phase 2 requires DATA_PROVIDER=polygon for now.")
        self.client = PolygonOptionsClient()

    def chain_snapshot_for_event_day(
        self, underlying: str, on: date
    ) -> Dict[str, Any]:
        exp_min = on
        exp_max = on + timedelta(days=self.cfg.chain_expiry_window_days)
        stats = self.client.chain_stats_for_day(underlying, on, exp_min=exp_min, exp_max=exp_max)
        return stats

    # Extend: compute 60d baseline of chain volumes and return z-score for `on`.
    # (We'll wire this once we confirm your API limits.)