"""Classical technical indicators, computed from OHLC only.

Every indicator here is emitted in a **scale-free** form.  Raw band levels
(Bollinger upper, Donchian lower, Keltner mid) are price-valued: they drift
with the index over years, they are incomparable between an index at 24,000
and a volatility index at 12, and they are recoverable from a moving average the library
already carries plus the width column below.  Emitting the normalised width
and position instead keeps the information while dropping the redundancy.

Deliberate omissions, all of them exact restatements of another column:

``Williams %R``
    Equals ``Stoch_K - 100`` exactly.  Both are the position of the close
    within the same high/low range over the same period; they differ only by
    an offset and a sign, so a model cannot distinguish them.

``MACD_Signal``
    ``MACD``, ``MACD_Signal`` and ``MACD_Hist`` satisfy
    ``Hist = MACD - Signal``, so any two of the three determine the third.
    MACD and the histogram are kept as the more directly informative pair.

``Donchian_Mid`` / ``Keltner_Mid``
    The midpoint of a Donchian channel is the mean of its bounds; the Keltner
    midline is the 20-period EMA, which ``moving_averages`` already produced.

Indicators that require Wilder's smoothing (RSI, ATR, ADX) use
``utils.wilder_ema``, which applies ``alpha = 1/period`` rather than the
``2/(period+1)`` of a conventional EMA.  Using the wrong one produces values
that disagree with every charting package while still looking plausible.
"""

import numpy as np
import pandas as pd

from .config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BOLLINGER_PERIOD,
    BOLLINGER_STD,
    CCI_PERIOD,
    DONCHIAN_PERIOD,
    KELTNER_ATR_MULT,
    KELTNER_PERIOD,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    STOCH_PERIOD,
    STOCH_SMOOTH,
)
from .utils import (
    attach,
    ema,
    get_index_names,
    get_ohlc,
    has_full_ohlc,
    require_columns,
    log,
    true_range,
    wilder_ema,
)


def generate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the classical indicator set for every index with complete OHLC.

    Depends on :func:`~.moving_averages.generate_moving_averages` for the
    Bollinger midline and the Keltner midline, both of which are reused rather
    than recomputed.
    """
    names = get_index_names(df)
    usable = [name for name in names if has_full_ohlc(df, name)]

    skipped = sorted(set(names) - set(usable))
    if skipped:
        log.info(
            "[technical_indicators ] skipping %d index/indices without full OHLC: %s",
            len(skipped), ", ".join(skipped[:3]),
        )

    require_columns(
        df,
        [f"{name}_SMA_{BOLLINGER_PERIOD}" for name in usable]
        + [f"{name}_EMA_{KELTNER_PERIOD}" for name in usable],
        produced_by="moving_averages",
    )

    features: dict[str, pd.Series] = {}

    for name in usable:
        _open, high, low, close = get_ohlc(df, name)

        # True range and ATR feed both the ATR columns and the Keltner
        # channel, so they are computed once per index and shared.
        atr = wilder_ema(true_range(high, low, close), ATR_PERIOD)

        _add_rsi(features, name, close)
        _add_macd(features, name, close)
        _add_atr(features, name, close, atr)
        _add_adx(features, name, high, low, close)
        _add_cci(features, name, high, low, close)
        _add_bollinger(features, name, df, close)
        _add_donchian(features, name, high, low, close)
        _add_keltner(features, name, df, close, atr)
        _add_stochastic(features, name, high, low, close)

    return attach(df, features, module="technical_indicators")


# --------------------------------------------------------------------------
# Individual indicators
# --------------------------------------------------------------------------

def _add_rsi(features: dict, name: str, close: pd.Series) -> None:
    """Relative Strength Index -- Wilder's original smoothing."""
    change = close.diff()
    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)

    average_gain = wilder_ema(gain, RSI_PERIOD)
    average_loss = wilder_ema(loss, RSI_PERIOD)

    relative_strength = average_gain / average_loss
    features[f"{name}_RSI_{RSI_PERIOD}"] = 100 - (100 / (1 + relative_strength))


def _add_macd(features: dict, name: str, close: pd.Series) -> None:
    """MACD line and histogram (the signal line is their difference)."""
    macd = ema(close, MACD_FAST) - ema(close, MACD_SLOW)
    signal = ema(macd, MACD_SIGNAL)

    features[f"{name}_MACD"] = macd
    features[f"{name}_MACD_Hist"] = macd - signal


def _add_atr(features: dict, name: str, close: pd.Series, atr: pd.Series) -> None:
    """Average True Range, raw and as a percentage of price.

    The raw ATR is price-scale; the percentage form is what makes volatility
    comparable across indices, so both are kept.
    """
    features[f"{name}_ATR_{ATR_PERIOD}"] = atr
    features[f"{name}_ATRP_{ATR_PERIOD}"] = atr / close


def _add_adx(
    features: dict, name: str, high: pd.Series, low: pd.Series, close: pd.Series
) -> None:
    """Average Directional Index and its two directional components.

    ADX is not a function of the two DI lines at a single point in time -- it
    is a smoothed average of their normalised difference -- so all three carry
    distinct information.
    """
    up_move = high.diff()
    down_move = -low.diff()

    # A directional move counts only when it exceeds the opposing move.
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = wilder_ema(true_range(high, low, close), ADX_PERIOD)

    plus_di = 100 * wilder_ema(plus_dm, ADX_PERIOD) / atr
    minus_di = 100 * wilder_ema(minus_dm, ADX_PERIOD) / atr

    directional_index = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    features[f"{name}_DI_Plus_{ADX_PERIOD}"] = plus_di
    features[f"{name}_DI_Minus_{ADX_PERIOD}"] = minus_di
    features[f"{name}_ADX_{ADX_PERIOD}"] = wilder_ema(directional_index, ADX_PERIOD)


def _add_cci(
    features: dict, name: str, high: pd.Series, low: pd.Series, close: pd.Series
) -> None:
    """Commodity Channel Index.

    CCI normalises by mean absolute deviation, not standard deviation.  There
    is no vectorised pandas reduction for a rolling MAD, so this uses
    ``rolling().apply()`` with ``raw=True`` -- still a pandas rolling
    operation, evaluated on NumPy arrays rather than Series, and the only
    place in the library that needs it.
    """
    typical_price = (high + low + close) / 3
    rolling = typical_price.rolling(CCI_PERIOD)

    mean_absolute_deviation = rolling.apply(
        lambda window: np.abs(window - window.mean()).mean(), raw=True
    )

    features[f"{name}_CCI_{CCI_PERIOD}"] = (
        typical_price - rolling.mean()
    ) / (0.015 * mean_absolute_deviation)


def _add_bollinger(
    features: dict, name: str, df: pd.DataFrame, close: pd.Series
) -> None:
    """Bollinger bands as normalised width and position.

    The midline is the existing ``SMA_20`` column rather than a recomputation.
    ``Width`` captures band separation relative to price; ``PctB`` places the
    close within the bands, where 0 is the lower band and 1 the upper.
    """
    middle = df[f"{name}_SMA_{BOLLINGER_PERIOD}"]
    deviation = close.rolling(BOLLINGER_PERIOD).std()

    upper = middle + BOLLINGER_STD * deviation
    lower = middle - BOLLINGER_STD * deviation

    features[f"{name}_BB_Width_{BOLLINGER_PERIOD}"] = (upper - lower) / middle
    features[f"{name}_BB_PctB_{BOLLINGER_PERIOD}"] = (close - lower) / (upper - lower)


def _add_donchian(
    features: dict, name: str, high: pd.Series, low: pd.Series, close: pd.Series
) -> None:
    """Donchian channel as normalised width and position within the range.

    The window includes the current bar, which is the usual convention and is
    backward-looking regardless.
    """
    upper = high.rolling(DONCHIAN_PERIOD).max()
    lower = low.rolling(DONCHIAN_PERIOD).min()

    features[f"{name}_Donchian_Width_{DONCHIAN_PERIOD}"] = (upper - lower) / close
    features[f"{name}_Donchian_PctRange_{DONCHIAN_PERIOD}"] = (close - lower) / (
        upper - lower
    )


def _add_keltner(
    features: dict, name: str, df: pd.DataFrame, close: pd.Series, atr: pd.Series
) -> None:
    """Keltner channel as normalised width and position.

    The midline is the existing ``EMA_20`` column.  Keltner differs from
    Bollinger in using ATR rather than closing-price deviation, so it reacts to
    intraday range and gaps that a close-only measure misses.
    """
    middle = df[f"{name}_EMA_{KELTNER_PERIOD}"]
    band = KELTNER_ATR_MULT * atr

    features[f"{name}_Keltner_Width_{KELTNER_PERIOD}"] = (2 * band) / middle
    features[f"{name}_Keltner_PctPos_{KELTNER_PERIOD}"] = (
        close - (middle - band)
    ) / (2 * band)


def _add_stochastic(
    features: dict, name: str, high: pd.Series, low: pd.Series, close: pd.Series
) -> None:
    """Stochastic oscillator, fast %K and smoothed %D."""
    highest = high.rolling(STOCH_PERIOD).max()
    lowest = low.rolling(STOCH_PERIOD).min()

    percent_k = 100 * (close - lowest) / (highest - lowest)

    features[f"{name}_Stoch_K_{STOCH_PERIOD}"] = percent_k
    features[f"{name}_Stoch_D_{STOCH_PERIOD}"] = percent_k.rolling(STOCH_SMOOTH).mean()
