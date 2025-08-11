# Phase 1: Volume Information Leakage Analysis
# Section 1: Setup and Configuration (Refined)

from typing import Tuple, List, Optional
from datetime import date
import logging
import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Import centralized types
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common.types import LaunchEvent, AnalysisParams

# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
plt.style.use("default")  # Neutral, minimal plotting defaults

# ---------------------------------------------------------------------
# Analyzer class
# ---------------------------------------------------------------------
class VolumeInformationLeakageAnalyzer:
    """
    Analyze volume patterns around technology product launches to detect potential
    information leakage. The primary anchor is the announcement date. Release dates
    are retained for robustness checks and outcome measurement.
    """

    def __init__(
        self,
        params: AnalysisParams = None,
    ) -> None:
        
        if params is None:
            params = AnalysisParams()
            
        self.baseline_days = params.baseline_days
        self.signal_window_announce = params.signal_window_announce
        self.signal_window_release = params.signal_window_release
        self.z_thresholds = params.z_thresholds

        # Events (ISO date strings parsed to date objects)
        raw_events = [
            # Apple iPhone
            {"name": "iPhone 12", "company": "Apple", "ticker": "AAPL",
             "announcement": "2020-10-13", "release": "2020-10-23", "next_earnings": "2020-10-29",
             "category": "Consumer Hardware"},
            {"name": "iPhone 13", "company": "Apple", "ticker": "AAPL",
             "announcement": "2021-09-14", "release": "2021-09-24", "next_earnings": "2021-10-28",
             "category": "Consumer Hardware"},
            {"name": "iPhone 14", "company": "Apple", "ticker": "AAPL",
             "announcement": "2022-09-07", "release": "2022-09-16", "next_earnings": "2022-10-27",
             "category": "Consumer Hardware"},
            {"name": "iPhone 15", "company": "Apple", "ticker": "AAPL",
             "announcement": "2023-09-12", "release": "2023-09-22", "next_earnings": "2023-11-02",
             "category": "Consumer Hardware"},

            # NVIDIA GeForce RTX
            {"name": "RTX 30 Series", "company": "NVIDIA", "ticker": "NVDA",
             "announcement": "2020-09-01", "release": "2020-09-17", "next_earnings": "2020-11-18",
             "category": "Semiconductor Hardware"},
            {"name": "RTX 40 Series", "company": "NVIDIA", "ticker": "NVDA",
             "announcement": "2022-09-20", "release": "2022-10-12", "next_earnings": "2022-11-16",
             "category": "Semiconductor Hardware"},
            {"name": "RTX 40 SUPER", "company": "NVIDIA", "ticker": "NVDA",
             "announcement": "2024-01-08", "release": "2024-01-17", "next_earnings": "2024-02-21",
             "category": "Semiconductor Hardware"},

            # Microsoft Xbox
            {"name": "Xbox Series X/S", "company": "Microsoft", "ticker": "MSFT",
             "announcement": "2020-09-09", "release": "2020-11-10", "next_earnings": "2020-10-27",
             "category": "Gaming Hardware"},
        ]

        def _d(s: Optional[str]) -> Optional[date]:
            return date.fromisoformat(s) if s else None

        events: List[LaunchEvent] = []
        for e in raw_events:
            a = _d(e["announcement"])
            r = _d(e["release"])
            ne = _d(e.get("next_earnings"))
            if a is None or r is None:
                raise ValueError(f"Invalid dates for event: {e}")
            if a > r:
                raise ValueError(f"Announcement after release for {e['name']}")
            events.append(
                LaunchEvent(
                    name=e["name"],
                    company=e["company"],
                    ticker=e["ticker"],
                    announcement=a,
                    release=r,
                    next_earnings=ne,
                    category=e["category"],
                )
            )
        self.events = events

        # Storage and output directories
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # Placeholder for results aggregation
        self.results = {}

        # Init summary
        logging.info("Volume Information Leakage Analyzer initialized.")
        logging.info(
            "Baseline=%sd | AnnounceWindow=%s | ReleaseWindow=%s | Z=%s",
            self.baseline_days,
            self.signal_window_announce,
            self.signal_window_release,
            self.z_thresholds,
        )
        tickers = sorted({e.ticker for e in self.events})
        logging.info("Events loaded: %s | Tickers: %s", len(self.events), ", ".join(tickers))

# Instantiate analyzer
analyzer = VolumeInformationLeakageAnalyzer()

# Academic references (metadata only)
ACADEMIC_CITATIONS = {
    "core_methodology": [
        "Back, Crotty & Li (2018) RFS — Identifying Information Asymmetry",
        "Barbon, Di Maggio, Franzoni & Landier (2019) JF — Brokers and Order Flow Leakage",
        "Muravyev, Pearson & Pollet (forthcoming JFE; WP 2022) — Why Does Options Info Predict Returns?"
    ],
    "event_studies": [
        "MacKinlay (1997) JEL — Event Studies in Economics and Finance"
    ],
    "statistical_methods": [
        "UT Austin (2024) — Cross-sectional correlation pitfalls in event studies",
        "MIT — Bootstrap inference for small samples"
    ],
    "regulatory_context": [
        "SEC v. NVIDIA (circa 2000)",
        "SEC v. Gene Levoff (Apple, 2011–2016; penalty 2024)",
        "SEC/DOJ v. Microsoft manager (2013)"
    ],
}

# Preview and persist master events table
_events_df = pd.DataFrame(
    {
        "name": [e.name for e in analyzer.events],
        "company": [e.company for e in analyzer.events],
        "ticker": [e.ticker for e in analyzer.events],
        "announcement": [e.announcement.isoformat() for e in analyzer.events],
        "release": [e.release.isoformat() for e in analyzer.events],
        "next_earnings": [e.next_earnings.isoformat() if e.next_earnings else None for e in analyzer.events],
        "category": [e.category for e in analyzer.events],
    }
)

_events_df.to_csv("data/processed/events_master.csv", index=False)
logging.info("Saved events master CSV: data/processed/events_master.csv")
