"""Single source of truth for every tunable in the feature library.

Retuning the whole library means editing this file and nothing else.  Modules
never declare their own windows -- they import from here, so a window set is
defined in exactly one place.

Some window sets are deliberately declared as subsets of others.  That is not
a style choice: it is how the library avoids emitting duplicate features.
``STAT_WINDOWS`` being a subset of ``VOL_WINDOWS`` lets ``rolling_statistics``
reuse the volatility columns already on the frame instead of recomputing an
identical rolling standard deviation.  The assertions at the bottom of this
file enforce those relationships at import time, so a careless edit fails
loudly instead of silently producing duplicated columns.
"""

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Window sets
# --------------------------------------------------------------------------

# Horizons for simple percentage returns.
RETURN_WINDOWS = [1, 2, 3, 5, 10, 20, 40, 60, 120, 252]

# Simple and exponential moving average lengths.
MA_WINDOWS = [5, 10, 20, 50, 100, 200]

# Rolling volatility lookbacks.
VOL_WINDOWS = [5, 10, 20, 50, 100]

# Rolling distribution statistics.  Kept deliberately small -- these are the
# most redundant features in the library, and every window here multiplies
# seven statistics across every index.
STAT_WINDOWS = [10, 20, 50]

# Momentum (price-difference) lookbacks.
MOMENTUM_WINDOWS = [5, 10, 20, 50, 100, 200]

# Relative-strength lookbacks, measured against the benchmark.
RS_WINDOWS = [20, 60, 120]

# Trend-strength (efficiency ratio) lookbacks.
TREND_WINDOWS = [20, 50, 100]

# Cross-sectional / market-wide aggregation lookbacks, measured on returns.
MARKET_WINDOWS = [20, 60]

# Market-wide aggregation lookbacks measured on volatility.  Kept separate
# from MARKET_WINDOWS because the two read different source columns and are
# therefore constrained by different subset rules.
MARKET_VOL_WINDOWS = [20, 50]

# Rate of Change is the percentage form of momentum, which makes it identical
# to a return over the same horizon.  Deriving the list rather than hardcoding
# it means ROC only ever covers horizons that ``returns`` does not already
# produce -- add 50 to RETURN_WINDOWS and ROC_50 disappears automatically.
ROC_WINDOWS = [w for w in MOMENTUM_WINDOWS if w not in RETURN_WINDOWS]

# --------------------------------------------------------------------------
# Technical indicator periods
# --------------------------------------------------------------------------

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
CCI_PERIOD = 20
STOCH_PERIOD = 14
STOCH_SMOOTH = 3
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
DONCHIAN_PERIOD = 20
KELTNER_PERIOD = 20
KELTNER_ATR_MULT = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --------------------------------------------------------------------------
# Market structure
# --------------------------------------------------------------------------

# Trading days per year, used wherever a figure is annualised.  Currently
# inert: the only reference is a docstring in volatility.py, because
# Volatility_Ann was dropped as a constant multiple of Volatility_{w}.
TRADING_DAYS = 252

# Market-specific settings live in a MarketSpec rather than in module
# constants, so one library serves several markets.  ``benchmark`` is the index
# every other index is measured against in relative_strength.  ``vol_index`` is
# the market's volatility index; it gets the full per-index treatment but is
# held out of market-wide aggregates, because "percent of indices above their
# 200-day average" stops being a breadth statistic if a volatility index is one
# of the members.
#
# Availability rules (see merge.py):
#
# ``foreign_after_close``
#     Series that trade on another exchange's calendar and settle *after* the
#     local close.  On local date t the latest usable value is the foreign close
#     dated strictly before t; same-date foreign closes are not yet known.
# ``publication_lags``
#     Series published k local trading days after their reference date (macro
#     releases, for example).  Shifted by k rows before any feature is built.


@dataclass(frozen=True)
class MarketSpec:
    """Benchmark, volatility index and data-availability rules for one market."""

    name: str
    benchmark: str
    vol_index: str | None = None
    foreign_after_close: frozenset[str] = frozenset()
    publication_lags: dict[str, int] = field(default_factory=dict)
    max_staleness_days: int = 5

    @property
    def exclude_from_breadth(self) -> set[str]:
        return {self.vol_index} if self.vol_index else set()


#: A market where every index, including the volatility index, closes together.
INDIA = MarketSpec(name="india", benchmark="NIFTY_50", vol_index="INDIA_VIX")

#: A market whose indices publish closes only, with a volatility index that is
#: computed on a foreign exchange and settles after the local close.
BRAZIL = MarketSpec(
    name="brazil",
    benchmark="IBOV",
    vol_index="VXEWZ",
    foreign_after_close=frozenset({"VXEWZ"}),
)

MARKETS = {spec.name: spec for spec in (INDIA, BRAZIL)}

# Breadth new-high / new-low and positive-return lookback.
BREADTH_WINDOW = 20

# Momentum horizons tested for positive sign in breadth.  BREADTH_WINDOW is
# deliberately absent: the sign of momentum over w days and the sign of the
# return over w days are the same test, so including it would duplicate
# Market_Pct_Positive_Return_{BREADTH_WINDOW}D exactly.
BREADTH_MOMENTUM_WINDOWS = [50, 200]

# Moving averages tested in breadth's "percent above" statistics.
BREADTH_MA_WINDOWS = [20, 50, 200]

# Volatility lookback used to classify high / low volatility regimes.
REGIME_VOL_WINDOW = 20

# Rolling window over which a volatility reading is ranked against its own
# history to decide whether the current regime is high or low volatility.
REGIME_LOOKBACK = 252
REGIME_HIGH_QUANTILE = 0.80
REGIME_LOW_QUANTILE = 0.20

# --------------------------------------------------------------------------
# Consistency checks
#
# These guard the no-duplicate-features invariant.  They run at import, so a
# bad edit fails immediately rather than silently doubling up columns.
# --------------------------------------------------------------------------

assert set(STAT_WINDOWS) <= set(VOL_WINDOWS), (
    "STAT_WINDOWS must be a subset of VOL_WINDOWS so rolling_statistics can "
    "reuse the existing Volatility_{w} columns for its z-score instead of "
    "recomputing an identical rolling standard deviation."
)

assert set(RS_WINDOWS) <= set(RETURN_WINDOWS), (
    "RS_WINDOWS must be a subset of RETURN_WINDOWS so relative_strength can "
    "reuse the existing Return_{w}D columns instead of recomputing them."
)

assert set(BREADTH_MA_WINDOWS) <= set(MA_WINDOWS), (
    "BREADTH_MA_WINDOWS must be a subset of MA_WINDOWS -- breadth reads the "
    "moving average columns produced by generate_moving_averages()."
)

assert set(BREADTH_MOMENTUM_WINDOWS) <= set(MOMENTUM_WINDOWS), (
    "BREADTH_MOMENTUM_WINDOWS must be a subset of MOMENTUM_WINDOWS -- breadth "
    "reads the momentum columns produced by generate_momentum()."
)

assert REGIME_VOL_WINDOW in VOL_WINDOWS, (
    "REGIME_VOL_WINDOW must be one of VOL_WINDOWS -- market_regime reads the "
    "volatility column produced by generate_volatility()."
)

assert BREADTH_WINDOW in RETURN_WINDOWS, (
    "BREADTH_WINDOW must be one of RETURN_WINDOWS -- breadth reads the return "
    "column produced by generate_returns()."
)

assert BREADTH_WINDOW not in BREADTH_MOMENTUM_WINDOWS, (
    "BREADTH_WINDOW must not appear in BREADTH_MOMENTUM_WINDOWS. Momentum over "
    "w days is positive exactly when the w-day return is positive, so breadth "
    "statistics for both at the same window are the identical column."
)

assert set(MARKET_WINDOWS) <= set(RETURN_WINDOWS), (
    "MARKET_WINDOWS must be a subset of RETURN_WINDOWS -- market_regime reads "
    "the return columns produced by generate_returns()."
)

assert set(MARKET_VOL_WINDOWS) <= set(VOL_WINDOWS), (
    "MARKET_VOL_WINDOWS must be a subset of VOL_WINDOWS -- market_regime reads "
    "the volatility columns produced by generate_volatility()."
)
