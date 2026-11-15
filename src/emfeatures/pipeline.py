"""Feature generation pipeline.

Applies every feature module in dependency order, verifies the library's
invariants, and then applies the **decision lag**.

Every module computes a value on row ``t`` from closes up to and including
``t`` -- correct for a feature "as of the close of t".  The original library
stopped there and left the alignment to the user, which made it easy to pair a
row's features with that same row's return: a same-day close leak.  Here the
generated columns are shifted by ``decision_lag`` rows (default 1), so a
feature on row ``t`` uses data up to ``t-1`` and can be joined directly to the
return over day ``t`` (``target_return``).  Raw input columns are left as they
are, and ``feature_columns`` lists only the generated, lagged ones.
"""

from __future__ import annotations

import pandas as pd

from .breadth import generate_breadth
from .config import MarketSpec
from .distance_from_ma import generate_distance_from_ma
from .market_regime import generate_market_regime
from .momentum import generate_momentum
from .moving_averages import generate_moving_averages
from .relative_strength import generate_relative_strength
from .returns import generate_returns
from .rolling_statistics import generate_rolling_statistics
from .technical_indicators import generate_technical_indicators
from .volatility import generate_volatility

# Applied in this order; each entry is (label, function, needs_spec).
STAGES = [
    ("returns", generate_returns, False),
    ("moving_averages", generate_moving_averages, False),
    ("distance_from_ma", generate_distance_from_ma, False),
    ("volatility", generate_volatility, False),
    ("rolling_statistics", generate_rolling_statistics, False),
    ("momentum", generate_momentum, False),
    ("relative_strength", generate_relative_strength, True),
    ("breadth", generate_breadth, True),
    ("market_regime", generate_market_regime, True),
    ("technical_indicators", generate_technical_indicators, False),
]


def build_features(df: pd.DataFrame, spec: MarketSpec, decision_lag: int = 1) -> pd.DataFrame:
    """Run every module on a master frame and return it with the lagged features added.

    Args:
        df: master frame from ``merge.merge_frames`` (``Date`` plus ``{NAME}_{Field}``).
        spec: the market's benchmark, volatility index and availability rules.
        decision_lag: rows by which generated features are delayed. ``1`` means a
            feature on row ``t`` uses data up to ``t-1``. ``0`` reproduces the
            original "as of the close of t" stamping and is only for comparison.
    """
    if decision_lag < 0:
        raise ValueError("decision_lag must be >= 0")
    original_columns = list(df.columns)

    for label, generate, needs_spec in STAGES:
        before = set(df.columns)
        df = generate(df, spec) if needs_spec else generate(df)
        if not before.issubset(df.columns):
            lost = sorted(before - set(df.columns))
            raise RuntimeError(f"{label} removed {len(lost)} column(s), e.g. {lost[:3]}")

    _verify(df, original_columns)
    generated = [c for c in df.columns if c not in original_columns]
    if decision_lag:
        df[generated] = df[generated].shift(decision_lag)
    df.attrs["feature_columns"] = generated
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The generated (and lagged) feature columns of a ``build_features`` output."""
    return list(df.attrs.get("feature_columns", []))


def target_return(df: pd.DataFrame, name: str) -> pd.Series:
    """Simple return of ``name`` over day ``t`` (close ``t-1`` to close ``t``), on row ``t``."""
    close = df[f"{name}_Close"]
    return (close / close.shift(1) - 1).rename(f"{name}_target")


def _verify(df: pd.DataFrame, original_columns: list[str]) -> None:
    """Assert the library's end-to-end invariants."""
    duplicates = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicates:
        raise RuntimeError(f"{len(duplicates)} duplicate column name(s): {duplicates[:5]}")
    missing = [c for c in original_columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Input columns were dropped: {missing[:5]}")
    generated = df.columns.difference(original_columns)
    empty = [c for c in generated if df[c].isna().all()]
    if empty:
        raise RuntimeError(
            f"{len(empty)} generated column(s) are entirely NaN, e.g. {empty[:5]}. "
            "Check window lengths against the sample size."
        )
