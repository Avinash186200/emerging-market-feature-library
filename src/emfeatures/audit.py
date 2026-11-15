"""Measure the two timing leaks found in the original pipeline, on the same data.

1. **Same-date foreign join.** The original merger left-joined the foreign
   volatility index on the same calendar date and forward filled it.
   ``same_date_join`` reproduces that so the effect can be measured against the
   strict as-of join in ``merge.merge_frames``.
2. **Same-day close.** Features stamped "as of the close of t" paired with the
   return over day t. Measured with ``decision_lag=0`` against ``1``.

Both are reported as the correlation of a feature with the benchmark's return
over day ``t`` on the same row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MarketSpec
from .merge import prefix_frame
from .pipeline import build_features, feature_columns, target_return


def same_date_join(frames: dict[str, pd.DataFrame], spec: MarketSpec) -> pd.DataFrame:
    """The original merge: local outer merge, then foreign series joined on the same date and forward filled."""
    local = [n for n in frames if n not in spec.foreign_after_close]
    master = None
    for name in sorted(local):
        df = prefix_frame(name, frames[name])
        master = df if master is None else master.merge(df, on="Date", how="outer")
    master = master.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    for name in sorted(spec.foreign_after_close & set(frames)):
        df = prefix_frame(name, frames[name])
        cols = [c for c in df.columns if c != "Date"]
        master = master.merge(df, on="Date", how="left")
        master[cols] = master[cols].ffill()
    return master


def _corr(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 30 or a[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def foreign_join_effect(frames: dict[str, pd.DataFrame], spec: MarketSpec) -> pd.DataFrame:
    """Correlation of volatility-index features with the benchmark's same-row return, old join vs fixed."""
    from .merge import merge_frames

    vix = spec.vol_index
    rows = []
    for label, master in (("same-date join (original)", same_date_join(frames, spec)),
                          ("strict as-of join (fixed)", merge_frames(frames, spec))):
        feats = build_features(master.copy(), spec, decision_lag=1)
        y = target_return(feats, spec.benchmark)
        cols = [c for c in feature_columns(feats) if c.startswith(f"{vix}_")]
        cors = pd.Series({c: _corr(feats[c], y) for c in cols}).dropna()
        rows.append({
            "join": label,
            "vol_return_1d_corr": _corr(feats[f"{vix}_Return_1D"], y),
            "max_abs_corr": float(cors.abs().max()),
            "n_features_abs_corr_gt_0.1": int((cors.abs() > 0.1).sum()),
            "n_vol_features": len(cors),
        })
    return pd.DataFrame(rows).set_index("join")


def decision_lag_effect(master: pd.DataFrame, spec: MarketSpec) -> pd.DataFrame:
    """Share of features strongly correlated with the same-row benchmark return, lag 0 vs lag 1."""
    rows = []
    for lag in (0, 1):
        feats = build_features(master.copy(), spec, decision_lag=lag)
        y = target_return(feats, spec.benchmark)
        cors = pd.Series({c: _corr(feats[c], y) for c in feature_columns(feats)}).dropna()
        rows.append({
            "decision_lag": lag,
            "benchmark_return_1d_corr": _corr(feats[f"{spec.benchmark}_Return_1D"], y),
            "share_abs_corr_gt_0.1": float((cors.abs() > 0.1).mean()),
            "max_abs_corr": float(cors.abs().max()),
        })
    return pd.DataFrame(rows).set_index("decision_lag")
