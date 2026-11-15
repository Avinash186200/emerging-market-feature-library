"""Per-index raw frames -> one master frame on the local trading calendar.

Each raw frame has ``Date`` plus whichever of ``Open, High, Low, Close`` the
source publishes (some exchanges publish closes only).  Columns are prefixed
with the index name, so ``Close`` of ``IDX_A`` becomes ``IDX_A_Close``.

Local indices outer-merge into the date spine.  Series listed in
``MarketSpec.foreign_after_close`` trade on another calendar and settle after
the local close, so they are attached with a *strict* as-of join: on local date
``t`` they carry the latest foreign value dated strictly before ``t``.  A
same-date join, as in the original pipeline, hands every local row a foreign
close that printed after the local market had shut.  The join is limited to
``max_staleness_days`` so a long foreign outage shows up as missing rather than
as a flat line.  Series in ``publication_lags`` are then shifted by their lag.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import MarketSpec

OHLC_FIELDS = ("Open", "High", "Low", "Close")


def prefix_frame(name: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Keep ``Date`` and the OHLC fields present, prefixed with ``name``."""
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    keep = ["Date"] + [f for f in OHLC_FIELDS if f in df.columns]
    return df[keep].rename(columns={f: f"{name}_{f}" for f in OHLC_FIELDS})


def merge_frames(frames: dict[str, pd.DataFrame], spec: MarketSpec) -> pd.DataFrame:
    """Merge raw per-index frames into one master frame following ``spec``'s rules."""
    local = {n: f for n, f in frames.items() if n not in spec.foreign_after_close}
    foreign = {n: f for n, f in frames.items() if n in spec.foreign_after_close}
    if not local:
        raise ValueError("no local series to define the date spine")

    master: pd.DataFrame | None = None
    for name in sorted(local):
        df = prefix_frame(name, local[name])
        master = df if master is None else master.merge(df, on="Date", how="outer")
    assert master is not None
    master = master.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    for name in sorted(foreign):
        df = prefix_frame(name, foreign[name]).sort_values("Date")
        master = pd.merge_asof(
            master,
            df,
            on="Date",
            direction="backward",
            allow_exact_matches=False,  # a foreign close dated t is not known at local close t
            tolerance=pd.Timedelta(days=spec.max_staleness_days),
        )

    for name, lag in spec.publication_lags.items():
        cols = [c for c in master.columns if c.startswith(f"{name}_")]
        master[cols] = master[cols].shift(lag)

    return master


def load_csv_dir(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read ``<NAME>.csv`` files (``Date`` plus OHLC or Close) from a directory."""
    files = sorted(Path(path).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files in {path}")
    return {f.stem: pd.read_csv(f, parse_dates=["Date"]) for f in files}
