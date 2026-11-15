"""Synthetic per-index raw data with the same columns as the two original datasets.

No real data ships with this repo. The generator reproduces the *shape* of the
inputs (index names, which fields each index publishes, calendars, a volatility
index that starts part-way through the sample) so the full pipeline runs from a
clone.

Market factor model: each index's daily log return is ``beta * market + sector +
idiosyncratic``, with clustered volatility. The volatility index moves against
the market.

**Planted timing trap (``brazil`` preset).** The volatility index there settles on
a foreign exchange after the local close. Part of the local market's next-day
move, the news that arrives between the local close and the foreign close,
is already in that foreign close. A same-date join therefore lets a feature
"see" part of tomorrow's local return. The ``india`` preset has no such trap:
its volatility index closes with the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INDIA_NAMES = [
    "INDIA_VIX", "NIFTY_100", "NIFTY_200", "NIFTY_50", "NIFTY_500", "NIFTY_ALPHA_50",
    "NIFTY_ALPHA_LOW_VOLATILITY_30", "NIFTY_AUTO", "NIFTY_BANK", "NIFTY_CONSUMER_DURABLES",
    "NIFTY_ENERGY", "NIFTY_FINANCIAL_SERVICES", "NIFTY_FMCG", "NIFTY_HEALTHCARE",
    "NIFTY_HIGH_BETA_50", "NIFTY_INFRASTRUCTURE", "NIFTY_IT", "NIFTY_LARGEMIDCAP_250",
    "NIFTY_MEDIA", "NIFTY_METAL", "NIFTY_MIDCAP_100", "NIFTY_MIDCAP_50", "NIFTY_NEXT_50",
    "NIFTY_OIL_AND_GAS", "NIFTY_PHARMA", "NIFTY_PRIVATE_BANK", "NIFTY_PSU_BANK", "NIFTY_REALTY",
    "NIFTY_SMALLCAP_100", "NIFTY_SMALLCAP_250", "NIFTY_SMALLCAP_50",
]
BRAZIL_NAMES = [
    "BDRX", "IBOV", "IBRA", "IBXL", "IBXX", "ICON", "IDIV", "IFIX", "IFNC", "IGCT", "IGCX",
    "IMAT", "IMOB", "INDX", "ITAG", "IVBX", "MLCX", "SMLL", "UTIL", "VXEWZ",
]
PRESETS = {
    "india": {"names": INDIA_NAMES, "vol": "INDIA_VIX", "bench": "NIFTY_50",
              "close_only": False, "foreign_vol": False, "vol_start": 0.0},
    "brazil": {"names": BRAZIL_NAMES, "vol": "VXEWZ", "bench": "IBOV",
               "close_only": True, "foreign_vol": True, "vol_start": 0.53},
}


def _calendar(start: str, n: int, rng: np.random.Generator, holiday_rate: float) -> pd.DatetimeIndex:
    days = pd.bdate_range(start=start, periods=int(n * (1 + 2 * holiday_rate)) + 10)
    keep = rng.random(len(days)) > holiday_rate
    return days[keep][:n]


def _ohlc(close: np.ndarray, sig: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.25, len(close)) * sig)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.5, len(close))) * sig)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.5, len(close))) * sig)
    return {"Open": open_, "High": high, "Low": low, "Close": close}


def make_synthetic_frames(
    market: str = "brazil", n_days: int = 1600, seed: int = 7, start: str = "2020-01-02"
) -> dict[str, pd.DataFrame]:
    """Raw frames keyed by index name, each with ``Index, Date`` and its published fields.

    Args:
        market: ``"india"`` (all indices OHLC, volatility index on the same calendar) or
            ``"brazil"`` (closes only, plus a foreign-calendar OHLC volatility index that
            starts part-way through the sample and settles after the local close).
        n_days: local trading days.
        seed: RNG seed.
        start: first calendar date.
    """
    if market not in PRESETS:
        raise ValueError(f"market must be one of {sorted(PRESETS)}")
    p = PRESETS[market]
    rng = np.random.default_rng(seed)
    local_days = _calendar(start, n_days, rng, holiday_rate=0.04)
    n = len(local_days)

    lsig = np.zeros(n)
    for t in range(1, n):
        lsig[t] = 0.97 * lsig[t - 1] + 0.1 * rng.standard_normal()
    sig = 0.011 * np.exp(lsig)

    # news between the local close on t and the foreign close on t: part of tomorrow's market move
    post_close = 0.45 * sig * rng.standard_normal(n)
    market_ret = 0.9 * sig * rng.standard_normal(n)
    market_ret[1:] += post_close[:-1]

    frames: dict[str, pd.DataFrame] = {}
    equity = [nm for nm in p["names"] if nm != p["vol"]]
    n_sectors = 6
    sector_ret = 0.5 * sig[:, None] * rng.standard_normal((n, n_sectors))
    for i, name in enumerate(equity):
        beta = 1.0 if name == p["bench"] else rng.uniform(0.6, 1.4)
        idio = (0.15 if name == p["bench"] else 0.6) * sig * rng.standard_normal(n)
        sector = 0.0 if name == p["bench"] else sector_ret[:, i % n_sectors]
        r = beta * market_ret + sector + idio
        close = 1000 * rng.uniform(1, 100) * np.exp(np.cumsum(r))
        fields = {"Close": close} if p["close_only"] else _ohlc(close, sig, rng)
        frames[name] = pd.DataFrame({"Index": name, "Date": local_days, **fields})

    # volatility index: moves against the market, mean-reverting in logs
    if p["foreign_vol"]:
        vol_days = _calendar(start, n_days, np.random.default_rng(seed + 1), holiday_rate=0.04)
        shock = pd.Series(market_ret + post_close, index=local_days)  # includes post-close news
        shock = shock.reindex(vol_days.union(local_days)).fillna(0.0).reindex(vol_days).to_numpy()
        days = vol_days
    else:
        shock = market_ret
        days = local_days
    m = len(days)
    lv = np.empty(m)
    lv[0] = np.log(20.0)
    for t in range(1, m):
        lv[t] = lv[t - 1] + 0.03 * (np.log(20.0) - lv[t - 1]) - 4.0 * shock[t] + 0.02 * rng.standard_normal()
    vclose = np.exp(lv)
    vsig = np.full(m, 0.03)
    vol = pd.DataFrame({"Index": p["vol"], "Date": days, **_ohlc(vclose, vsig, rng)})
    first = int(p["vol_start"] * m)
    frames[p["vol"]] = vol.iloc[first:].reset_index(drop=True)
    return frames
