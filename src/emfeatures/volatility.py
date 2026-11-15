"""Rolling volatility of daily returns, total and directionally split.

Three families are produced per index and window:

``Volatility_{w}``
    Rolling standard deviation of daily returns -- total variability.

``DownsideVol_{w}`` / ``UpsideVol_{w}``
    Semi-deviations that isolate the two tails.  A market that drifts up
    steadily and crashes occasionally has these two badly unequal, and the
    difference between them carries information that total volatility hides.

Annualised volatility is deliberately *not* emitted.  Multiplying a rolling
standard deviation by the square root of the trading year is a constant
scaling: it produces a column that is perfectly collinear with
``Volatility_{w}`` and carries no additional information for any model.  Where
an annualised figure is needed for reporting, scale by
``sqrt(config.TRADING_DAYS)`` at the point of use.
"""

import pandas as pd

from .config import VOL_WINDOWS
from .utils import attach, get_index_names, require_columns


def generate_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling total, downside and upside volatility for every index.

    Depends on :func:`~.returns.generate_returns` for the one-day return.
    """
    names = get_index_names(df)
    require_columns(
        df, [f"{name}_Return_1D" for name in names], produced_by="returns"
    )

    features: dict[str, pd.Series] = {}

    for name in names:
        returns = df[f"{name}_Return_1D"]

        # Semi-deviations use a clip rather than a filter.  Selecting only the
        # negative days with `.where(returns < 0)` would leave NaN on positive
        # days, and a rolling window containing any NaN returns NaN -- so the
        # output would be missing precisely during calm periods.  Clipping to
        # zero keeps every window full and yields the standard downside
        # deviation measured against a zero threshold.
        downside = returns.clip(upper=0.0)
        upside = returns.clip(lower=0.0)

        for window in VOL_WINDOWS:
            features[f"{name}_Volatility_{window}"] = returns.rolling(window).std()
            features[f"{name}_DownsideVol_{window}"] = (
                downside.pow(2).rolling(window).mean().pow(0.5)
            )
            features[f"{name}_UpsideVol_{window}"] = (
                upside.pow(2).rolling(window).mean().pow(0.5)
            )

    return attach(df, features, module="volatility")
