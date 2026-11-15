"""Rolling distribution statistics of the daily return series.

Where ``volatility`` measures spread, this module describes the *shape* of the
recent return distribution: its centre, its extremes, its asymmetry and its
tail weight.

Two statistics named in the original specification are intentionally absent,
because both are exact restatements of a column the library already has:

``Roll_Std_{w}``
    Identical to ``Volatility_{w}`` -- both are the rolling standard deviation
    of the same return series over the same window.

``Roll_Var_{w}``
    The square of that same quantity, and therefore a deterministic transform
    carrying no information a model cannot recover.

``STAT_WINDOWS`` is constrained by ``config`` to be a subset of
``VOL_WINDOWS``, which lets the z-score below reuse the volatility column that
already exists instead of recomputing an identical rolling standard deviation.
"""

import pandas as pd

from .config import STAT_WINDOWS
from .utils import attach, get_index_names, require_columns, rolling_zscore


def generate_rolling_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean, median, min, max, skew, kurtosis and z-score.

    Depends on :func:`~.returns.generate_returns` and
    :func:`~.volatility.generate_volatility`.
    """
    names = get_index_names(df)

    require_columns(
        df, [f"{name}_Return_1D" for name in names], produced_by="returns"
    )
    require_columns(
        df,
        [f"{name}_Volatility_{w}" for name in names for w in STAT_WINDOWS],
        produced_by="volatility",
    )

    features: dict[str, pd.Series] = {}

    for name in names:
        returns = df[f"{name}_Return_1D"]

        for window in STAT_WINDOWS:
            rolling = returns.rolling(window)

            features[f"{name}_Roll_Mean_{window}"] = rolling.mean()
            features[f"{name}_Roll_Median_{window}"] = rolling.median()
            features[f"{name}_Roll_Min_{window}"] = rolling.min()
            features[f"{name}_Roll_Max_{window}"] = rolling.max()
            features[f"{name}_Roll_Skew_{window}"] = rolling.skew()
            features[f"{name}_Roll_Kurt_{window}"] = rolling.kurt()

            # Reuse the standard deviation computed by generate_volatility
            # rather than recomputing an identical rolling std here.
            features[f"{name}_Roll_ZScore_{window}"] = rolling_zscore(
                returns, window, std=df[f"{name}_Volatility_{window}"]
            )

    return attach(df, features, module="rolling_statistics")
