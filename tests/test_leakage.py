"""Causality tests: truncation, perturbation, source scan, and the two timing fixes."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emfeatures import MARKETS, MarketSpec, build_features, feature_columns, merge_frames, target_return
from emfeatures.audit import decision_lag_effect, foreign_join_effect

TRUNCATION = 60


@pytest.mark.parametrize("market", ["brazil", "india"])
@pytest.mark.parametrize("lag", [0, 1])
def test_features_identical_when_the_future_is_removed(market, lag, request):
    master = request.getfixturevalue(f"{market}_master")
    spec = MARKETS[market]
    full = build_features(master.copy(), spec, decision_lag=lag)
    short = build_features(master.iloc[:-TRUNCATION].copy(), spec, decision_lag=lag)
    assert list(full.columns) == list(short.columns)
    n = len(short)
    leaking = [
        c for c in short.columns
        if not full[c].iloc[:n].reset_index(drop=True).equals(short[c].reset_index(drop=True))
    ]
    assert not leaking, leaking[:10]


def test_lagged_features_at_t_ignore_row_t(brazil_master):
    """With decision_lag=1, scrambling every input from row t onward leaves row t untouched."""
    spec = MARKETS["brazil"]
    t = 700
    base = build_features(brazil_master.copy(), spec)
    scrambled = brazil_master.copy()
    cols = [c for c in scrambled.columns if c != "Date"]
    rng = np.random.default_rng(0)
    scrambled.loc[t:, cols] = scrambled.loc[t:, cols] * rng.uniform(0.5, 1.5, (len(scrambled) - t, len(cols)))
    new = build_features(scrambled, spec)
    fc = feature_columns(base)
    pd.testing.assert_series_equal(base.loc[t, fc], new.loc[t, fc], check_names=False)
    assert not np.allclose(base.loc[t + 1, fc].astype(float), new.loc[t + 1, fc].astype(float), equal_nan=True)


def test_foreign_close_dated_t_is_not_visible_on_local_row_t(brazil_frames):
    spec = MARKETS["brazil"]
    master = merge_frames(brazil_frames, spec)
    vol = brazil_frames[spec.vol_index].copy()
    day = vol["Date"].iloc[200]
    vol.loc[vol["Date"] == day, "Close"] *= 10.0
    altered = merge_frames({**brazil_frames, spec.vol_index: vol}, spec)
    row = master.index[master["Date"] == day]
    col = f"{spec.vol_index}_Close"
    if len(row):  # the foreign day is also a local day
        assert master.loc[row, col].equals(altered.loc[row, col])
    later = master.index[master["Date"] > day][0]
    assert altered.loc[later, col] != master.loc[later, col]


def test_foreign_series_goes_missing_after_a_long_gap(brazil_frames):
    spec = MARKETS["brazil"]
    vol = brazil_frames[spec.vol_index]
    cut = vol["Date"].iloc[100]
    gappy = vol[(vol["Date"] < cut) | (vol["Date"] > cut + pd.Timedelta(days=20))]
    master = merge_frames({**brazil_frames, spec.vol_index: gappy}, spec)
    inside = master[(master["Date"] > cut + pd.Timedelta(days=spec.max_staleness_days + 1))
                    & (master["Date"] <= cut + pd.Timedelta(days=20))]
    assert len(inside) and inside[f"{spec.vol_index}_Close"].isna().all()


def test_publication_lag_shifts_a_series(india_frames):
    name = sorted(n for n in india_frames if n.startswith("NIFTY_A"))[0]
    base = merge_frames(india_frames, MARKETS["india"])
    spec = MarketSpec("t", "NIFTY_50", "INDIA_VIX", publication_lags={name: 2})
    lagged = merge_frames(india_frames, spec)
    pd.testing.assert_series_equal(lagged[f"{name}_Close"], base[f"{name}_Close"].shift(2))


def test_same_day_close_leak_is_measured_and_removed(brazil_master):
    eff = decision_lag_effect(brazil_master, MARKETS["brazil"])
    assert eff.loc[0, "benchmark_return_1d_corr"] == pytest.approx(1.0)
    assert abs(eff.loc[1, "benchmark_return_1d_corr"]) < 0.15


def test_same_date_foreign_join_leaks_and_strict_join_does_not(brazil_frames):
    eff = foreign_join_effect(brazil_frames, MARKETS["brazil"])
    old, new = eff.iloc[0], eff.iloc[1]
    assert abs(old["vol_return_1d_corr"]) > 0.15
    assert abs(new["vol_return_1d_corr"]) < 0.12


def test_target_return_is_close_to_close(brazil_master):
    y = target_return(brazil_master, "IBOV")
    c = brazil_master["IBOV_Close"]
    assert y.iloc[10] == pytest.approx(c.iloc[10] / c.iloc[9] - 1)


def test_no_future_reading_constructs_in_source():
    """Parse every module for a negative shift, a centred window or a backward fill."""
    src = Path(__file__).resolve().parents[1] / "src" / "emfeatures"
    offenders = []
    for path in sorted(src.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            called = getattr(node.func, "attr", None)
            if called == "shift" and node.args and isinstance(node.args[0], ast.UnaryOp) \
                    and isinstance(node.args[0].op, ast.USub):
                offenders.append(f"{path.name}:{node.lineno} negative shift")
            if called in {"bfill", "backfill"}:
                offenders.append(f"{path.name}:{node.lineno} backward fill")
            for kw in node.keywords:
                if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    offenders.append(f"{path.name}:{node.lineno} centred window")
                if kw.arg == "direction" and isinstance(kw.value, ast.Constant) \
                        and kw.value.value in {"forward", "nearest"}:
                    offenders.append(f"{path.name}:{node.lineno} forward as-of join")
    assert not offenders, offenders


def test_truncation_test_cannot_see_the_foreign_timing_leak(brazil_frames):
    """Why the original suite passed: the same-date join is row-causal, so truncation is blind to it."""
    from emfeatures.audit import same_date_join

    spec = MARKETS["brazil"]
    master = same_date_join(brazil_frames, spec)
    full = build_features(master.copy(), spec)
    short = build_features(master.iloc[:-TRUNCATION].copy(), spec)
    n = len(short)
    cols = [c for c in short.columns if c.startswith("VXEWZ_")]
    assert all(full[c].iloc[:n].reset_index(drop=True).equals(short[c].reset_index(drop=True)) for c in cols)
