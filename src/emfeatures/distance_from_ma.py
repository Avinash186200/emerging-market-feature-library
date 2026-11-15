"""Scale-free distance between price and its moving averages.

A raw moving average is a price level, so its usefulness to a model decays as
the index drifts over years.  ``Close / SMA - 1`` is stationary and directly
comparable across indices trading at wildly different levels, which is what
makes these the more model-ready form of the same information.
"""

import pandas as pd

from .config import MA_WINDOWS
from .utils import attach, get_index_names, require_columns


def generate_distance_from_ma(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``{INDEX}_Distance_SMA{w}`` and ``{INDEX}_Distance_EMA{w}``.

    Depends on :func:`~.moving_averages.generate_moving_averages` having run.
    """
    names = get_index_names(df)

    require_columns(
        df,
        [f"{name}_SMA_{w}" for name in names for w in MA_WINDOWS]
        + [f"{name}_EMA_{w}" for name in names for w in MA_WINDOWS],
        produced_by="moving_averages",
    )

    features: dict[str, pd.Series] = {}

    for name in names:
        close = df[f"{name}_Close"]
        for window in MA_WINDOWS:
            features[f"{name}_Distance_SMA{window}"] = (
                close / df[f"{name}_SMA_{window}"] - 1
            )
            features[f"{name}_Distance_EMA{window}"] = (
                close / df[f"{name}_EMA_{window}"] - 1
            )

    return attach(df, features, module="distance_from_ma")
