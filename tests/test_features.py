"""Features mean what their names claim; no overwrites; no affine duplicates; data layout."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from emfeatures import MARKETS, build_features, make_synthetic_frames
from emfeatures.config import MA_WINDOWS, RETURN_WINDOWS, ROC_WINDOWS, RSI_PERIOD, STOCH_PERIOD
from emfeatures.synthetic import BRAZIL_NAMES, INDIA_NAMES
from emfeatures.utils import attach, get_index_names, has_full_ohlc


@pytest.fixture(scope="module")
def brazil(brazil_master):
    return build_features(brazil_master.copy(), MARKETS["brazil"], decision_lag=0)


@pytest.fixture(scope="module")
def india(india_master):
    return build_features(india_master.copy(), MARKETS["india"], decision_lag=0)


def test_synthetic_layout_matches_the_original_files():
    b = make_synthetic_frames("brazil", n_days=400, seed=1)
    i = make_synthetic_frames("india", n_days=400, seed=1)
    assert sorted(b) == sorted(BRAZIL_NAMES) and len(b) == 20
    assert sorted(i) == sorted(INDIA_NAMES) and len(i) == 31
    assert all(list(b[n].columns) == ["Index", "Date", "Close"] for n in b if n != "VXEWZ")
    assert list(b["VXEWZ"].columns) == ["Index", "Date", "Open", "High", "Low", "Close"]
    assert all(list(f.columns) == ["Index", "Date", "Open", "High", "Low", "Close"] for f in i.values())
    assert b["VXEWZ"]["Date"].min() > b["IBOV"]["Date"].min()  # starts part-way through


def test_ohlc_families_appear_exactly_where_ohlc_exists(brazil, brazil_master):
    names = get_index_names(brazil_master)
    with_ohlc = [n for n in names if has_full_ohlc(brazil_master, n)]
    assert with_ohlc and len(with_ohlc) < len(names)
    for n in names:
        assert (f"{n}_RSI_{RSI_PERIOD}" in brazil.columns) is (n in with_ohlc)


def test_breadth_uses_close_extremes_without_high_low(brazil, india):
    assert "Market_Pct_New_CloseHigh_20D" in brazil.columns
    assert "Market_Pct_New_High_20D" not in brazil.columns
    assert "Market_Pct_New_High_20D" in india.columns


@pytest.mark.parametrize("which", ["brazil", "india"])
def test_bounded_and_fractional_features(which, request):
    f = request.getfixturevalue(which)
    for c in [c for c in f.columns if c.endswith(f"_RSI_{RSI_PERIOD}") or f"_Stoch_K_{STOCH_PERIOD}" in c]:
        assert f[c].dropna().between(0, 100).all(), c
    for c in [c for c in f.columns if c.startswith("Market_Pct_")]:
        assert f[c].dropna().between(0, 1).all(), c
    for c in [c for c in f.columns if "_Regime_" in c]:
        assert set(f[c].dropna().unique()) <= {0.0, 1.0}, c
    for c in [c for c in f.columns if "_TrendStrength_" in c]:
        assert f[c].dropna().between(0, 1).all(), c


def test_member_count_tracks_reporting_indices(brazil, brazil_master):
    universe = [n for n in get_index_names(brazil_master) if n != "VXEWZ"]
    expected = brazil_master[[f"{n}_Close" for n in universe]].notna().sum(axis=1)
    np.testing.assert_allclose(brazil["Market_N_Members"], expected)


def test_sma_and_distance_match_direct_computation(india):
    close = india["NIFTY_50_Close"]
    pd.testing.assert_series_equal(india["NIFTY_50_SMA_20"], close.rolling(20).mean(), check_names=False)
    pd.testing.assert_series_equal(
        india["NIFTY_50_Distance_SMA50"], close / close.rolling(50).mean() - 1, check_names=False
    )
    for w in MA_WINDOWS:
        assert india[f"NIFTY_50_SMA_{w}"].iloc[: w - 1].isna().all()


def test_decision_lag_shifts_every_generated_column(india_master):
    a = build_features(india_master.copy(), MARKETS["india"], decision_lag=0)
    b = build_features(india_master.copy(), MARKETS["india"], decision_lag=1)
    gen = a.attrs["feature_columns"]
    pd.testing.assert_frame_equal(b[gen], a[gen].shift(1))
    pd.testing.assert_series_equal(a["NIFTY_50_Close"], b["NIFTY_50_Close"])  # inputs untouched


def test_attach_never_overwrites():
    df = pd.DataFrame({"A_Close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        attach(df, {"A_Close": pd.Series([0.0, 0.0])}, module="t")


def test_no_infinities(brazil, india):
    for f in (brazil, india):
        num = f.select_dtypes(include=[np.number])
        assert not np.isinf(num.to_numpy()).any()


def _signature(s: pd.Series) -> str | None:
    v = s.to_numpy(dtype="float64", na_value=np.nan)
    ok = ~np.isnan(v)
    if ok.sum() < 30:
        return None
    x = v[ok]
    sd = x.std()
    if not np.isfinite(sd) or sd == 0:
        return None
    z = np.round((x - x.mean()) / sd, 8)
    return hashlib.blake2b(ok.tobytes() + min(z.tobytes(), (-z).tobytes()), digest_size=16).hexdigest()


def test_no_two_features_are_affine_restatements(india):
    groups: dict[str, list[str]] = {}
    for c in india.attrs["feature_columns"]:
        sig = _signature(india[c])
        if sig:
            groups.setdefault(sig, []).append(c)
    dups = [g for g in groups.values() if len(g) > 1]
    assert not dups, dups[:5]


def test_roc_never_overlaps_return_windows():
    assert not set(ROC_WINDOWS) & set(RETURN_WINDOWS)
