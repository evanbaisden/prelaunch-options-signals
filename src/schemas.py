"""
Data validation schemas for prelaunch options signals analysis.
Provides DataFrame validation for events_master.csv and other data inputs.
"""
import pandas as pd
from datetime import date
from typing import Optional

try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False

if not PANDERA_AVAILABLE:
    try:
        from pydantic import BaseModel, validator
        PYDANTIC_AVAILABLE = True
    except ImportError:
        PYDANTIC_AVAILABLE = False


def validate_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate events_master.csv DataFrame structure and data types.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Validated DataFrame with correct types
        
    Raises:
        ValidationError: If validation fails
    """
    if PANDERA_AVAILABLE:
        return _validate_with_pandera(df)
    elif PYDANTIC_AVAILABLE:
        return _validate_with_pydantic(df)
    else:
        return _validate_basic(df)


if PANDERA_AVAILABLE:
    # Pandera schema definition
    events_schema = DataFrameSchema({
        "company": Column(str, Check.isin(["AAPL", "NVDA", "MSFT"]), nullable=False),
        "event_id": Column(str, nullable=False),
        "name": Column(str, nullable=False),
        "announcement_date": Column("datetime64[ns]", nullable=False),
        "release_date": Column("datetime64[ns]", nullable=False),
        "next_earnings_date": Column("datetime64[ns]", nullable=False),
        "source": Column(str, nullable=True, required=False),
    }, strict=False)  # Allow additional columns
    
    def _validate_with_pandera(df: pd.DataFrame) -> pd.DataFrame:
        """Validate using Pandera schema."""
        # Convert date columns to datetime if they're strings
        date_cols = ["announcement_date", "release_date", "next_earnings_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return events_schema.validate(df)


if PYDANTIC_AVAILABLE and not PANDERA_AVAILABLE:
    class EventRecord(BaseModel):
        """Pydantic model for individual event records."""
        company: str
        event_id: str
        name: str
        announcement_date: date
        release_date: date
        next_earnings_date: date
        source: Optional[str] = None
        
        @validator('company')
        def validate_company(cls, v):
            if v not in ["AAPL", "NVDA", "MSFT"]:
                raise ValueError(f"Company must be one of AAPL, NVDA, MSFT, got: {v}")
            return v
        
        @validator('release_date')
        def validate_dates(cls, v, values):
            if 'announcement_date' in values and v < values['announcement_date']:
                raise ValueError("Release date cannot be before announcement date")
            return v
    
    def _validate_with_pydantic(df: pd.DataFrame) -> pd.DataFrame:
        """Validate using Pydantic models."""
        # Convert date columns to datetime
        date_cols = ["announcement_date", "release_date", "next_earnings_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
        
        # Validate each row
        for idx, row in df.iterrows():
            try:
                EventRecord(**row.to_dict())
            except Exception as e:
                raise ValueError(f"Validation failed for row {idx}: {e}")
        
        return df


def _validate_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Basic validation without external libraries."""
    required_columns = ["company", "ticker", "name", "announcement", "release", "next_earnings"]
    
    # Check required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check ticker values (updated to include all tickers in our data)
    valid_tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "SNE", "VALVE"]
    invalid_tickers = df[~df["ticker"].isin(valid_tickers)]["ticker"].unique()
    if len(invalid_tickers) > 0:
        raise ValueError(f"Invalid tickers found: {invalid_tickers}. Must be one of {valid_tickers}")
    
    # Convert date columns
    date_cols = ["announcement", "release", "next_earnings"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        if df[col].isna().any():
            raise ValueError(f"Invalid dates found in column: {col}")
    
    # Check date logic
    invalid_dates = df[df["release"] < df["announcement"]]
    if len(invalid_dates) > 0:
        raise ValueError(f"Found {len(invalid_dates)} records where release is before announcement")
    
    return df