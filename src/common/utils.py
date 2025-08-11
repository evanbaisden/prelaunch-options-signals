import json, pickle, os, time, logging, re
from typing import Any, Optional, Dict, Iterable
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- Paths & logging ----------
RESULTS_DIR = "results"
RAW_DIR = "data/raw"
PROC_DIR = "data/processed"

def ensure_dirs() -> None:
    for d in (RESULTS_DIR, RAW_DIR, PROC_DIR):
        os.makedirs(d, exist_ok=True)

def setup_logging(level=logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

def set_plot_style() -> None:
    plt.style.use("default")

# ---------- IO helpers ----------
def to_python_scalars(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)):  return obj.tolist()
    if isinstance(obj, dict):           return {k: to_python_scalars(v) for k, v in obj.items()}
    if isinstance(obj, list):           return [to_python_scalars(v) for v in obj]
    return obj

def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dirs()
    with open(path, "w") as f:
        json.dump(to_python_scalars(data), f, indent=2)

def save_pickle(path: str, obj: Any) -> None:
    ensure_dirs()
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def save_df(path: str, df: pd.DataFrame) -> None:
    ensure_dirs()
    df.to_csv(path, index=False)

# ---------- Market data ----------
def download_ohlcv_yf(ticker: str, start, end, tries: int = 3, pause_s: float = 1.0) -> pd.DataFrame:
    """Small retry wrapper around yfinance.download."""
    last_err: Optional[Exception] = None
    for _ in range(max(1, tries)):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                df = df.rename(columns=str.title)
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            last_err = e
        time.sleep(pause_s)
    if last_err:
        logging.error(f"yfinance failed for {ticker}: {last_err}")
    return pd.DataFrame()

# ---------- Data wrangling ----------
def normalize_ohlcv_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        levels0 = set(map(str, out.columns.get_level_values(0)))
        levels1 = set(map(str, out.columns.get_level_values(1)))
        if ticker in levels0:         out = out.xs(ticker, axis=1, level=0)
        elif ticker.upper() in levels0: out = out.xs(ticker.upper(), axis=1, level=0)
        elif ticker in levels1:         out = out.xs(ticker, axis=1, level=1)
        elif ticker.upper() in levels1: out = out.xs(ticker.upper(), axis=1, level=1)
        else:
            out.columns = ["_".join(map(str, tup)) for tup in out.columns.to_list()]

    def _norm(col: str) -> str:
        c = str(col)
        c = re.sub(rf"(?i)[_\-\.]{ticker}$", "", c)
        c = re.sub(r"(?i)[_\-\.][A-Z]{1,5}$", "", c)
        c = c.strip().lower()
        if "adj" in c and "close" in c: return "Adj Close"
        if "close" in c:                return "Close"
        if "open" in c:                 return "Open"
        if "high" in c:                 return "High"
        if "low" in c:                  return "Low"
        if "volume" in c or c == "vol": return "Volume"
        return col

    out.columns = [_norm(c) for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")]
    return out

def get_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col] if col in df.columns else pd.Series(index=df.index, dtype=float)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")

def nearest_trading_idx(idx: pd.DatetimeIndex, target: pd.Timestamp) -> int:
    try:
        pos = idx.get_loc(target)
        if isinstance(pos, slice): return int(pos.start)
        if isinstance(pos, (list, np.ndarray)) and len(pos): return int(pos[0])
        return int(pos)
    except KeyError:
        return int(idx.get_indexer([target], method="nearest")[0])

# ---------- Small-sample stats ----------
def bootstrap_corr(x: Iterable[float], y: Iterable[float], n: int = 1000, cl: float = 0.95):
    x = np.asarray(list(x)); y = np.asarray(list(y))
    if len(x) != len(y) or len(x) < 3: return None
    corrs = []
    for _ in range(n):
        idx = np.random.randint(0, len(x), len(x))
        xs, ys = x[idx], y[idx]
        if xs.std() > 0 and ys.std() > 0:
            c = np.corrcoef(xs, ys)[0, 1]
            if np.isfinite(c): corrs.append(c)
    if not corrs: return None
    alpha = 1 - cl
    lo, hi = np.percentile(corrs, (alpha/2)*100), np.percentile(corrs, (1-alpha/2)*100)
    return {"correlation": float(np.mean(corrs)), "ci_lower": float(lo), "ci_upper": float(hi),
            "significant": not (lo <= 0 <= hi), "replications": n}

def corrado_rank_z(returns: Iterable[float]):
    arr = np.asarray(list(returns))
    n = len(arr)
    if n < 3: return None
    ranks = pd.Series(arr).rank(method="average").to_numpy()
    expected = (n + 1) / 2.0
    var = (n + 1) * (n - 1) / 12.0
    z = (ranks.mean() - expected) / np.sqrt(var / n)
    # two-tailed p via error function
    p = 2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(z) / np.sqrt(2))))
    return {"z": float(z), "p_value": float(p), "sig_5pct": p < 0.05, "sig_1pct": p < 0.01}
