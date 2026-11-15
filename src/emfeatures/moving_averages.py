"""Simple and exponential moving averages of closing prices.

These columns are consumed downstream by ``distance_from_ma`` (which turns
them into scale-free deviations), by ``breadth`` (percent of indices above
their averages), by ``market_regime`` (bull / bear classification) and by
``technical_indicators`` (Bollinger and Keltner both build on an average that
already exists here rather than recomputing one).
"""

import pandas as pd

from .config import MA_WINDOWS
from .utils import attach, ema, get_index_names


def generate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``{INDEX}_SMA_{w}`` and ``{INDEX}_EMA_{w}`` for every index.

    Both averages are trailing.  ``rolling`` defaults to ``center=False`` and
    the exponential average is seeded with ``min_periods``, so neither reads a
    future row and neither reports a value before its window has filled.
    """
    features: dict[str, pd.Series] = {}

    for name in get_index_names(df):
        close = df[f"{name}_Close"]
        for window in MA_WINDOWS:
            features[f"{name}_SMA_{window}"] = close.rolling(window).mean()
            features[f"{name}_EMA_{window}"] = ema(close, window)

    return attach(df, features, module="moving_averages")
