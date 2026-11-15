"""Shared helpers for the feature engineering library.

This module carries three responsibilities, each of which exists so that the
individual feature modules do not have to repeat themselves.

**Column discovery.**  The ``*_Close`` naming convention is interpreted here
and nowhere else.  Feature modules ask :func:`get_index_names` which indices
exist rather than scanning ``df.columns`` themselves, so changing the naming
convention is a one-line change in this file instead of a ten-file edit.

**Feature attachment.**  Every module ends by handing its finished feature
dictionary to :func:`attach`, which enforces the library-wide invariants in a
single place: never overwrite an existing column, build the block with one
concatenation, and report the count consistently.

**Causal primitives.**  Wilder smoothing, true range and rolling z-scores are
needed by more than one module, so they live here.

Leakage policy
--------------
Every helper in this package is causal: a value at row ``t`` is computed only
from rows at or before ``t``.  In practice that means the only operations
permitted anywhere in this library are ``rolling`` (never with
``center=True``), ``ewm``, ``pct_change``, ``diff`` and *non-negative*
``shift``.  A negative shift reaches into the future and must never appear.
Cross-sectional statistics read a single row at a time and are therefore
same-timestamp by construction.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("emfeatures")

CLOSE_SUFFIX = "_Close"
OHLC_FIELDS = ("Open", "High", "Low", "Close")


# --------------------------------------------------------------------------
# Column discovery
# --------------------------------------------------------------------------

def get_close_columns(df: pd.DataFrame) -> list[str]:
    """Return every closing-price column, sorted for deterministic ordering.

    This is the only place in the library that knows what a close column looks
    like.  Feature modules call it instead of scanning columns themselves.
    """
    return sorted(c for c in df.columns if c.endswith(CLOSE_SUFFIX))


def get_index_names(df: pd.DataFrame) -> list[str]:
    """Return the bare index names discovered from the close columns.

    ``IDX_A_Close`` yields ``IDX_A``.  Index names are never hardcoded
    anywhere in this library -- they are always discovered from the data.
    """
    return [c.removesuffix(CLOSE_SUFFIX) for c in get_close_columns(df)]


def get_universe(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    """Return the index names used for market-wide aggregation.

    ``exclude`` holds indices that are not tradeable price indices and would
    distort breadth or cross-sectional statistics.
    """
    excluded = exclude or set()
    return [name for name in get_index_names(df) if name not in excluded]


def has_full_ohlc(df: pd.DataFrame, name: str) -> bool:
    """True when Open, High, Low and Close all exist for ``name``."""
    return all(f"{name}_{field}" in df.columns for field in OHLC_FIELDS)


def get_ohlc(
    df: pd.DataFrame, name: str
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return the ``(open, high, low, close)`` series for one index."""
    require_columns(
        df,
        [f"{name}_{field}" for field in OHLC_FIELDS],
        produced_by="merge_frames",
    )
    return (
        df[f"{name}_Open"],
        df[f"{name}_High"],
        df[f"{name}_Low"],
        df[f"{name}_Close"],
    )


def matrix(df: pd.DataFrame, names: list[str], template: str) -> pd.DataFrame:
    """Build an aligned sub-frame for cross-sectional work.

    ``template`` is a format string such as ``"{name}_Close"``.  The returned
    frame has one column per entry in ``names``, renamed to the bare index
    name so that frames built from different templates align element-wise.

    This is what lets breadth and regime statistics stay fully vectorised:
    comparing "every close against every 20-day average" becomes a single
    DataFrame comparison rather than a loop.
    """
    columns = [template.format(name=name) for name in names]
    require_columns(df, columns, produced_by=_producer_hint(template))
    return df[columns].set_axis(names, axis=1)


def _producer_hint(template: str) -> str:
    """Best-effort guess at which module produces a templated column."""
    if "_SMA_" in template or "_EMA_" in template:
        return "moving_averages"
    if "_Volatility_" in template:
        return "volatility"
    if "_Momentum_" in template:
        return "momentum"
    if "_Return_" in template:
        return "returns"
    return "an earlier pipeline stage"


def require_columns(
    df: pd.DataFrame, columns: list[str], *, produced_by: str
) -> None:
    """Raise a diagnostic error when a dependency is missing.

    Feature modules depend on the output of earlier modules.  Without this
    check a reordered pipeline surfaces as a bare ``KeyError`` from deep
    inside a loop; with it, the error names the missing column and the module
    that produces it.
    """
    missing = [c for c in columns if c not in df.columns]
    if not missing:
        return

    preview = ", ".join(missing[:3])
    if len(missing) > 3:
        preview += f", ... (+{len(missing) - 3} more)"
    raise KeyError(
        f"{len(missing)} required column(s) missing: {preview}. "
        f"Run generate_{produced_by}() earlier in the pipeline."
    )


# --------------------------------------------------------------------------
# Feature attachment
# --------------------------------------------------------------------------

def attach(
    df: pd.DataFrame, features: dict[str, pd.Series], module: str
) -> pd.DataFrame:
    """Concatenate a feature block onto ``df`` and report the count.

    Every feature module funnels through this function, which is what makes
    the library's guarantees uniform rather than per-module conventions:

    * **Never overwrites.**  A name that already exists on the frame raises
      instead of silently replacing a column.
    * **No fragmentation.**  The whole block becomes one DataFrame and one
      ``concat``, regardless of how many features the module produced.
    * **No infinities.**  Ratios divide by rolling ranges that can be zero on
      a flat day; the resulting infinities become NaN so downstream models see
      a single missing-value sentinel.
    * **Consistent reporting.**  One count line per module.

    ``df`` is never mutated -- a new frame is returned.
    """
    if not features:
        log.info("[%-22s] generated %5d features", module, 0)
        return df

    clashes = sorted(set(features) & set(df.columns))
    if clashes:
        preview = ", ".join(clashes[:5])
        raise ValueError(
            f"{module} would overwrite {len(clashes)} existing column(s): "
            f"{preview}. Feature modules must only add new columns."
        )

    block = pd.DataFrame(features, index=df.index)
    block = block.replace([np.inf, -np.inf], np.nan)

    log.info("[%-22s] generated %5d features", module, len(block.columns))
    return pd.concat([df, block], axis=1)


# --------------------------------------------------------------------------
# Causal primitives
# --------------------------------------------------------------------------

def wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, as used by RSI, ATR and ADX.

    Wilder's original average is an exponential mean with ``alpha = 1/period``
    rather than the ``2/(period+1)`` of a conventional EMA.  Getting this wrong
    silently produces indicators that disagree with every charting package.
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Conventional exponential moving average with a warm-up guard.

    ``min_periods`` keeps the first ``span - 1`` values NaN.  Without it the
    series would open with a value that is simply the first price, which is a
    misleadingly confident reading rather than a moving average.
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range: the day's span, extended for overnight gaps."""
    previous_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def rolling_zscore(
    series: pd.Series, window: int, std: pd.Series | None = None
) -> pd.Series:
    """Standardise ``series`` against its own trailing window.

    ``std`` may be supplied when the caller already has the matching rolling
    standard deviation on the frame, which avoids recomputing it.
    """
    mean = series.rolling(window).mean()
    if std is None:
        std = series.rolling(window).std()
    return (series - mean) / std


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Kaufman's efficiency ratio -- net movement over total path travelled.

    Approaches 1 when a move is a clean straight line and 0 when the same net
    move is achieved by thrashing back and forth, which makes it a measure of
    trend quality rather than trend size.
    """
    net_move = (close - close.shift(window)).abs()
    total_move = close.diff().abs().rolling(window).sum()
    return net_move / total_move


def percent_true(mask: pd.DataFrame) -> pd.Series:
    """Row-wise fraction of True values, ignoring missing entries.

    A plain ``>`` between frames turns NaN into False, which would report a
    warm-up row as "0% of indices above their average" instead of "unknown".
    Callers pass a mask that has already been NaN-masked; this function then
    averages with ``skipna`` so warm-up rows stay NaN.
    """
    return mask.mean(axis=1, skipna=True)


def masked_compare(left: pd.DataFrame, right: pd.DataFrame, op: str) -> pd.DataFrame:
    """Compare two aligned frames, preserving NaN rather than coercing to False.

    ``op`` is a pandas comparison method name -- ``"gt"``, ``"lt"``, ``"ge"``
    or ``"le"``.  The result is a float frame of 1.0 / 0.0 with NaN wherever
    either operand was missing, ready for :func:`percent_true`.
    """
    comparison = getattr(left, op)(right).astype("float64")
    return comparison.where(left.notna() & right.notna())
