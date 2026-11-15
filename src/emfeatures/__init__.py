"""Causal market feature library for multi-index emerging-market panels.

Each module exposes one ``generate_*(df) -> pd.DataFrame`` function that adds
columns. Index names are discovered from the ``*_Close`` columns, and market
specifics (benchmark, volatility index, availability rules) come from a
``MarketSpec``, so the same code serves different markets.
"""

from .config import BRAZIL, INDIA, MARKETS, MarketSpec
from .merge import merge_frames
from .pipeline import build_features, feature_columns, target_return
from .synthetic import make_synthetic_frames

__all__ = [
    "BRAZIL", "INDIA", "MARKETS", "MarketSpec", "build_features", "feature_columns",
    "make_synthetic_frames", "merge_frames", "target_return",
]
__version__ = "0.1.0"
