"""Build the feature matrix for a market and run the timing audit.

Examples:
    python scripts/build_features.py --market brazil          # synthetic, same columns as the original
    python scripts/build_features.py --market india --no-audit
    python scripts/build_features.py --market brazil --csv-dir path/to/your/csvs --out results
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emfeatures import MARKETS, build_features, feature_columns, make_synthetic_frames, merge_frames  # noqa: E402
from emfeatures.audit import decision_lag_effect, foreign_join_effect  # noqa: E402
from emfeatures.merge import load_csv_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", choices=sorted(MARKETS), default="brazil")
    ap.add_argument("--csv-dir", help="directory of <NAME>.csv files (Date + OHLC or Close); default synthetic")
    ap.add_argument("--n-days", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--decision-lag", type=int, default=1)
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--out", help="write features.csv.gz here (not committed)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    spec = MARKETS[args.market]
    frames = load_csv_dir(args.csv_dir) if args.csv_dir else make_synthetic_frames(args.market, args.n_days, args.seed)
    master = merge_frames(frames, spec)
    feats = build_features(master.copy(), spec, decision_lag=args.decision_lag)
    cols = feature_columns(feats)
    ohlc = sum(1 for n in frames if {"High", "Low"} <= set(frames[n].columns))
    print(f"Market {args.market}: {len(frames)} indices ({ohlc} with OHLC), {len(master)} local days, "
          f"{len(cols)} features, decision lag {args.decision_lag}")
    print(f"Source: {'CSV directory' if args.csv_dir else f'synthetic, seed {args.seed}'}")

    if not args.no_audit:
        pd.set_option("display.width", 140)
        print("\nSame-day close (correlation with the benchmark's return on the same row):")
        print(decision_lag_effect(master, spec).round(3).to_string())
        if spec.foreign_after_close:
            print("\nForeign volatility index joined on the same date vs strictly before (decision lag 1):")
            print(foreign_join_effect(frames, spec).round(3).to_string())

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        feats.to_csv(out / "features.csv.gz", index=False)
        print(f"\nWrote {out / 'features.csv.gz'}")


if __name__ == "__main__":
    main()
