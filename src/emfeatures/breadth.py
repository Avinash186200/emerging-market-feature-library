"""Market breadth -- how broadly a move is shared across the index universe.

Breadth answers a question a single benchmark cannot: is the market rising
because most of it is rising, or because a handful of heavyweights are
dragging an otherwise flat market upward?  Divergence between a rising
benchmark and deteriorating breadth is one of the more durable warning signals
in equity markets.

Every statistic here is a *market-level* column: one series, not one per
index.  The volatility index is excluded from the universe (see
``MarketSpec.exclude_from_breadth``) because "percent of indices above their
200-day average" stops being a breadth statistic if a volatility index is one
of the members.

All aggregation is done frame-wise via ``utils.matrix``: comparing 30 closes
against 30 moving averages is a single DataFrame comparison, never a loop over
rows.
"""

import pandas as pd

from .config import (
    BREADTH_MA_WINDOWS,
    BREADTH_MOMENTUM_WINDOWS,
    BREADTH_WINDOW,
    MarketSpec,
)
from .utils import attach, get_universe, masked_compare, matrix, percent_true


def generate_breadth(df: pd.DataFrame, spec: MarketSpec) -> pd.DataFrame:
    """Add market-wide breadth statistics.

    Depends on :func:`~.returns.generate_returns`,
    :func:`~.moving_averages.generate_moving_averages` and
    :func:`~.momentum.generate_momentum`.
    """
    universe = get_universe(df, exclude=spec.exclude_from_breadth)
    features: dict[str, pd.Series] = {}

    closes = matrix(df, universe, "{name}_Close")
    daily_returns = matrix(df, universe, "{name}_Return_1D")

    # --- participation: how much of the market is in an uptrend ------------
    for window in BREADTH_MA_WINDOWS:
        averages = matrix(df, universe, f"{{name}}_SMA_{window}")
        features[f"Market_Pct_Above_SMA{window}"] = percent_true(
            masked_compare(closes, averages, "gt")
        )

    # --- participation: how much of the market is advancing ----------------
    window_returns = matrix(df, universe, f"{{name}}_Return_{BREADTH_WINDOW}D")
    features[f"Market_Pct_Positive_Return_{BREADTH_WINDOW}D"] = percent_true(
        (window_returns > 0).astype("float64").where(window_returns.notna())
    )

    for window in BREADTH_MOMENTUM_WINDOWS:
        momentum = matrix(df, universe, f"{{name}}_Momentum_{window}")
        features[f"Market_Pct_Positive_Momentum_{window}"] = percent_true(
            (momentum > 0).astype("float64").where(momentum.notna())
        )

    # --- extremes: new highs and lows over the breadth window --------------
    # This needs an intraday High/Low, which not every market publishes for
    # its indices -- some exchanges release closing levels only, so nothing
    # in the breadth universe carries a High or Low column at all.  Whether
    # the true intraday calculation is available is a property of the whole
    # universe, not of any one index: a breadth statistic that read intraday
    # extremes for some members and closing extremes for others would blend
    # two different definitions into a single column, so the choice is made
    # once, across every index in `universe`, using only the fields this
    # calculation actually reads (High/Low, not the full OHLC set -- Open is
    # irrelevant here).
    has_intraday_extremes = all(
        f"{name}_High" in df.columns and f"{name}_Low" in df.columns
        for name in universe
    )

    if has_intraday_extremes:
        highs = matrix(df, universe, "{name}_High")
        lows = matrix(df, universe, "{name}_Low")
        new_high_column = f"Market_Pct_New_High_{BREADTH_WINDOW}D"
        new_low_column = f"Market_Pct_New_Low_{BREADTH_WINDOW}D"
    else:
        # Fall back to the closing price standing in for both the high and
        # the low.  This is a coarser test -- "is today's close the best/worst
        # close of the last w days" rather than "did the day's true range
        # pierce the prior w-day extreme" -- so it is emitted under a
        # different name instead of silently redefining
        # Market_Pct_New_High_{w}D underneath existing consumers.
        highs = lows = closes
        new_high_column = f"Market_Pct_New_CloseHigh_{BREADTH_WINDOW}D"
        new_low_column = f"Market_Pct_New_CloseLow_{BREADTH_WINDOW}D"

    # `rolling(w).max()` includes the current bar, so an index printing a fresh
    # high registers on the day it happens.  The window looks only backwards.
    rolling_highs = highs.rolling(BREADTH_WINDOW).max()
    rolling_lows = lows.rolling(BREADTH_WINDOW).min()

    features[new_high_column] = percent_true(
        masked_compare(highs, rolling_highs, "ge")
    )
    features[new_low_column] = percent_true(
        masked_compare(lows, rolling_lows, "le")
    )

    # --- composition: how many members are reporting -----------------------
    # The universe is today's list of indices, and members join as their
    # history starts.  Every Market_* statistic is an average over whoever is
    # reporting, so the count is published alongside them: a jump in breadth
    # caused by a new member is then visible rather than silent.
    features["Market_N_Members"] = closes.notna().sum(axis=1).astype("float64")

    # --- aggregate performance and the advance/decline balance -------------
    # Equal weighted: every index counts the same, so this diverges from the
    # cap-weighted benchmark exactly when leadership is narrow.
    features["Market_EW_Return"] = daily_returns.mean(axis=1, skipna=True)

    advances = (daily_returns > 0).astype("float64").where(daily_returns.notna())
    declines = (daily_returns < 0).astype("float64").where(daily_returns.notna())

    advance_count = advances.sum(axis=1, skipna=True)
    decline_count = declines.sum(axis=1, skipna=True)

    # A day on which nothing declines would divide by zero.  Flooring the
    # denominator at one is the conventional treatment: it keeps the series
    # finite and monotone instead of dropping the strongest sessions to NaN.
    features["Market_AD_Ratio"] = advance_count / decline_count.clip(lower=1.0)

    return attach(df, features, module="breadth")
