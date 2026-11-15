"""Simple percentage returns over a range of horizons.

This is the base layer of the library: volatility, rolling statistics and
relative strength all read the one-day return produced here rather than
recomputing it.
"""

import pandas as pd

from .config import RETURN_WINDOWS
from .utils import attach, get_index_names


def generate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``{INDEX}_Return_{w}D`` for every index and every return horizon.

    ``pct_change(w)`` compares row ``t`` against row ``t - w``, so the value is
    causal: it uses no information beyond the current row.
    """
    features: dict[str, pd.Series] = {}

    for name in get_index_names(df):
        close = df[f"{name}_Close"]
        for window in RETURN_WINDOWS:
            features[f"{name}_Return_{window}D"] = close.pct_change(window)

    return attach(df, features, module="returns")
