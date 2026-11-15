"""Regime indicators: what kind of market is this, per index and in aggregate.

Two scopes are produced.

**Per index** -- trend direction, volatility state and trend quality for every
index in the dataset.

**Market wide** -- cross-sectional statistics computed across the universe at
each date.  These describe how the market is built up from its members rather than its
level: whether indices are moving together or apart, and whether the average
constituent is calm or stressed.

Volatility regimes are classified against each index's *own* trailing history
rather than a fixed threshold, because a level that is calm for a small-cap
index is turbulent for a large-cap one.  The ranking window is rolling and
backward-looking, so no future information enters the classification -- this
is the detail that most commonly introduces leakage in regime labelling, where
it is tempting to rank against the full sample.
"""

import pandas as pd

from .config import (
    MARKET_VOL_WINDOWS,
    MARKET_WINDOWS,
    REGIME_HIGH_QUANTILE,
    REGIME_LOOKBACK,
    REGIME_LOW_QUANTILE,
    REGIME_VOL_WINDOW,
    TREND_WINDOWS,
    MarketSpec,
)
from .utils import (
    attach,
    efficiency_ratio,
    get_index_names,
    get_universe,
    matrix,
    require_columns,
)


def generate_market_regime(df: pd.DataFrame, spec: MarketSpec) -> pd.DataFrame:
    """Add per-index regime flags and market-wide cross-sectional statistics.

    Depends on :func:`~.returns.generate_returns`,
    :func:`~.moving_averages.generate_moving_averages` and
    :func:`~.volatility.generate_volatility`.
    """
    names = get_index_names(df)
    features: dict[str, pd.Series] = {}

    _add_per_index_regimes(df, names, features)
    _add_cross_sectional(df, features, spec)

    return attach(df, features, module="market_regime")


def _add_per_index_regimes(
    df: pd.DataFrame, names: list[str], features: dict[str, pd.Series]
) -> None:
    """Bull/bear, volatility state and trend strength for each index."""
    require_columns(
        df,
        [f"{name}_SMA_50" for name in names] + [f"{name}_SMA_200" for name in names],
        produced_by="moving_averages",
    )
    require_columns(
        df,
        [f"{name}_Volatility_{REGIME_VOL_WINDOW}" for name in names],
        produced_by="volatility",
    )

    for name in names:
        close = df[f"{name}_Close"]
        fast = df[f"{name}_SMA_50"]
        slow = df[f"{name}_SMA_200"]

        # Bull requires both agreement between the averages and price holding
        # above the slower one -- a composite, not merely the sign of a
        # distance feature the library already carries.
        bull = (fast > slow) & (close > slow)
        features[f"{name}_Regime_Bull"] = (
            bull.astype("float64").where(fast.notna() & slow.notna() & close.notna())
        )

        volatility = df[f"{name}_Volatility_{REGIME_VOL_WINDOW}"]

        # Rank today's volatility against this index's own trailing year.  The
        # quantile is rolling, so the threshold at time t is built only from
        # data available at time t.
        high_threshold = volatility.rolling(REGIME_LOOKBACK).quantile(
            REGIME_HIGH_QUANTILE
        )
        low_threshold = volatility.rolling(REGIME_LOOKBACK).quantile(
            REGIME_LOW_QUANTILE
        )

        features[f"{name}_Regime_HighVol"] = (
            (volatility > high_threshold)
            .astype("float64")
            .where(volatility.notna() & high_threshold.notna())
        )
        features[f"{name}_Regime_LowVol"] = (
            (volatility < low_threshold)
            .astype("float64")
            .where(volatility.notna() & low_threshold.notna())
        )

        # Trend *quality* rather than trend size: how directly the index
        # travelled from where it was to where it is.
        for window in TREND_WINDOWS:
            features[f"{name}_TrendStrength_{window}"] = efficiency_ratio(close, window)


def _add_cross_sectional(
    df: pd.DataFrame, features: dict[str, pd.Series], spec: MarketSpec
) -> None:
    """Market-wide dispersion, cross-sectional volatility and momentum.

    Each of these reads a single row across many indices, so it is
    same-timestamp by construction and cannot leak.
    """
    universe = get_universe(df, exclude=spec.exclude_from_breadth)

    for window in MARKET_WINDOWS:
        returns = matrix(df, universe, f"{{name}}_Return_{window}D")

        # Dispersion: how far apart the indices are. Wide dispersion means
        # stock selection matters; narrow means everything moves as one block.
        features[f"Market_Dispersion_{window}"] = returns.std(axis=1, skipna=True)

        # Cross-sectional momentum: the average index's trailing return, which
        # differs from the cap-weighted benchmark whenever leadership narrows.
        features[f"Market_CrossSectional_Momentum_{window}"] = returns.mean(
            axis=1, skipna=True
        )

    for window in MARKET_VOL_WINDOWS:
        volatilities = matrix(df, universe, f"{{name}}_Volatility_{window}")

        # The typical constituent's own volatility -- distinct from dispersion,
        # which measures disagreement between constituents.
        features[f"Market_CrossSectional_Vol_{window}"] = volatilities.mean(
            axis=1, skipna=True
        )

    vix = spec.vol_index
    if vix and vix in get_index_names(df):
        vix_close = df[f"{vix}_Close"]
        threshold = vix_close.rolling(REGIME_LOOKBACK).quantile(REGIME_HIGH_QUANTILE)
        features["Market_VIX_Regime_High"] = (
            (vix_close > threshold)
            .astype("float64")
            .where(vix_close.notna() & threshold.notna())
        )
