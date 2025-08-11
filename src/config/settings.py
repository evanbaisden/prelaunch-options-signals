"""
Configuration settings for the prelaunch options signals project
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# API Configuration
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "polygon")

# Data provider settings
PROVIDERS = {
    "polygon": {
        "base_url": "https://api.polygon.io",
        "rate_limit": 5,  # requests per minute for free tier
        "timeout": 30
    }
}

# Analysis parameters
DEFAULT_ANALYSIS_PARAMS = {
    "lookback_days": 60,
    "confidence_level": 0.95,
    "min_volume_threshold": 1000
}

# Ensure required directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(exist_ok=True)