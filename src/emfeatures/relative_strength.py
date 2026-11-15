"""Every index measured against the benchmark, plus volatility-index coupling.

Absolute performance says little on its own -- a sector up 3% in a week that
carried the whole market up 5% is weak, not strong.  This module reframes each
index in benchmark-relative terms, then adds each index's evolving
relationship with the volatility index.

The benchmark and the volatility index come from the ``MarketSpec`` rather
than being hardcoded, and every peer is discovered from the data, so the module adapts
unchanged to a different market's index set.

``RS_WINDOWS`` is constrained by ``config`` to be a subset of
``RETURN_WINDOWS``, which lets the return differences below reuse existing
return columns rather than recomputing them.
"""

import pandas as pd

from .config import RS_WINDOWS, MarketSpec
from .utils import attach, get_index_names, log, require_columns


def generate_relative_strength(df: pd.DataFrame, spec: MarketSpec) -> pd.DataFrame:
    """Add benchmark-relative and VIX-relative features for every index.

    Depends on :func:`~.returns.generate_returns`.  Silently skips the
    benchmark or VIX blocks if those indices are absent from the dataset.
    """
    names = get_index_names(df)
    features: dict[str, pd.Series] = {}
    benchmark, vix = spec.benchmark, spec.vol_index

    if benchmark in names:
        _add_benchmark_features(df, names, features, benchmark)
    else:
        log.info("[relative_strength    ] benchmark %r not found, skipping", benchmark)

    if vix in names:
        _add_vix_features(df, names, features, vix)
    else:
        log.info("[relative_strength    ] %r not found, skipping VIX features", vix)

    return attach(df, features, module="relative_strength")


def _add_benchmark_features(
    df: pd.DataFrame, names: list[str], features: dict[str, pd.Series], benchmark: str
) -> None:
    """Price ratios, return differences and relative momentum vs the benchmark."""
    peers = [name for name in names if name != benchmark]

    require_columns(
        df,
        [f"{name}_Return_1D" for name in names]
        + [f"{benchmark}_Return_{w}D" for w in RS_WINDOWS]
        + [f"{name}_Return_{w}D" for name in peers for w in RS_WINDOWS],
        produced_by="returns",
    )

    benchmark_close = df[f"{benchmark}_Close"]
    benchmark_daily = df[f"{benchmark}_Return_1D"]

    for name in peers:
        # The ratio line itself: rising means the index is outperforming.
        ratio = df[f"{name}_Close"] / benchmark_close
        features[f"{name}_RS_Ratio_{benchmark}"] = ratio

        excess_daily = df[f"{name}_Return_1D"] - benchmark_daily

        for window in RS_WINDOWS:
            # Outperformance over the window, from existing return columns.
            features[f"{name}_RS_RetDiff_{window}D"] = (
                df[f"{name}_Return_{window}D"] - df[f"{benchmark}_Return_{window}D"]
            )

            # Average daily excess return -- a smoother read on persistence of
            # outperformance than a single endpoint-to-endpoint difference.
            features[f"{name}_RS_RollRelReturn_{window}"] = (
                excess_daily.rolling(window).mean()
            )

            # Momentum of the ratio line: is relative strength itself trending?
            features[f"{name}_RS_RollRelMomentum_{window}"] = ratio.pct_change(window)


def _add_vix_features(
    df: pd.DataFrame, names: list[str], features: dict[str, pd.Series], vix: str
) -> None:
    """Rolling correlation and beta of each index against the volatility index.

    Equity indices are normally negatively correlated with implied volatility.
    When that relationship weakens or flips, the market is usually behaving
    unusually -- which is exactly the sort of regime information a model
    cannot infer from price alone.
    """
    require_columns(df, [f"{vix}_Return_1D"], produced_by="returns")

    vix_daily = df[f"{vix}_Return_1D"]

    for name in [n for n in names if n != vix]:
        index_daily = df[f"{name}_Return_1D"]

        for window in RS_WINDOWS:
            rolling_index = index_daily.rolling(window)

            features[f"{name}_VIX_Corr_{window}"] = rolling_index.corr(vix_daily)

            # Beta to volatility: covariance over the variance of the VIX.
            features[f"{name}_VIX_Beta_{window}"] = (
                rolling_index.cov(vix_daily) / vix_daily.rolling(window).var()
            )
