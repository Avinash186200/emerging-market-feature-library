"""Price momentum, rate of change and momentum acceleration.

The three families here are kept deliberately distinct from ``returns``:

``Momentum_{w}``
    The textbook definition, ``Close - Close.shift(w)``: net movement in price
    points.  Note that this is a *price-scale* quantity and therefore not
    stationary -- it is included because it is the standard definition, but
    ``Return_{w}D`` or ``Distance_SMA{w}`` are usually the better model input.

``ROC_{w}``
    Rate of change is the percentage form of momentum, which makes it
    arithmetically identical to a simple return over the same horizon.  It is
    therefore emitted *only* for horizons that ``returns`` does not already
    cover; ``config.ROC_WINDOWS`` derives that set automatically, so the two
    modules can never produce duplicate columns.

``Acceleration_{w}``
    The change in momentum itself -- the second difference of price.  Positive
    values mean a trend is still gathering pace rather than merely continuing.

Rolling cumulative return is likewise absent: compounding ``w`` consecutive
daily returns reproduces ``Return_{w}D`` exactly.
"""

import pandas as pd

from .config import MOMENTUM_WINDOWS, ROC_WINDOWS
from .utils import attach, get_index_names


def generate_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum, acceleration and non-duplicate rate-of-change columns."""
    features: dict[str, pd.Series] = {}

    for name in get_index_names(df):
        close = df[f"{name}_Close"]

        for window in MOMENTUM_WINDOWS:
            momentum = close - close.shift(window)
            features[f"{name}_Momentum_{window}"] = momentum

            # Second difference: is the trend speeding up or fading?
            features[f"{name}_Acceleration_{window}"] = (
                momentum - momentum.shift(window)
            )

        for window in ROC_WINDOWS:
            features[f"{name}_ROC_{window}"] = close.pct_change(window) * 100

    return attach(df, features, module="momentum")
