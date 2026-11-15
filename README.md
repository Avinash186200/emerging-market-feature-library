# Emerging Market Feature Library

[![CI](https://github.com/Avinash186200/emerging-market-feature-library/actions/workflows/ci.yml/badge.svg)](https://github.com/Avinash186200/emerging-market-feature-library/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11-blue) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A causal feature library for multi-index equity panels: returns, trend, volatility, distribution shape, relative strength, breadth, regime and classical technical indicators, generated for every index it finds, with no duplicates and no look-ahead.

It started as feature work on two emerging-market index panels, which I audited and rebuilt into one market-agnostic package. One panel had every index with full OHLC and a volatility index on the same exchange. The other had closes only, plus a volatility index computed on a foreign exchange. The original index data is third-party and is **not** in this repo, not even as a sample. Everything here runs on a **synthetic generator with the same index names and columns**, and you can point it at your own CSVs.

## At a glance

| | |
|---|---|
| **Input** | One CSV per index: `Date` plus `Open, High, Low, Close` or `Close` only. Synthetic by default (31 indices with OHLC, or 19 close-only indices plus a foreign-calendar volatility index) |
| **Method** | Ten feature modules run in dependency order, index names discovered from the data, market specifics in a `MarketSpec`, and duplicate and affine-duplicate checks |
| **Output** | About 3,850 features for the 31-index panel and 2,160 for the 20-index panel, each lagged so that row t uses data up to t-1 |
| **Leakage controls** | A decision lag (default 1), strict as-of joins for foreign series that settle after the local close, publication lags, truncation and perturbation tests, and an AST scan for negative shifts, centred windows and backward fills |
| **Status** | Library complete and tested on synthetic data. It is a feature library, so it makes no performance claims |

## How it works

```
per-index CSVs ──► merge (local spine, strict as-of foreign join, publication lags) ──► 10 modules ──► decision lag ──► features
                   merge.py                                                              pipeline.py
```

Modules, in order: `returns`, `moving_averages`, `distance_from_ma`, `volatility` (total, downside, upside), `rolling_statistics`, `momentum`, `relative_strength` (against the benchmark and the volatility index), `breadth`, `market_regime`, `technical_indicators` (RSI, MACD, ATR, ADX, CCI, Bollinger, Donchian, Keltner and stochastic, only where full OHLC exists). Every module is additive and never overwrites a column. Window sets live in `config.py`, with import-time assertions that stop one window set from silently duplicating another module's columns.

## The timing audit

The original pipeline was already careful about row-level causality: trailing windows only, no negative shifts, and a truncation test. I found two timing problems that a truncation test cannot see, because neither involves reading a later row.

`python scripts/build_features.py --market brazil` measures both on synthetic data. Each figure is the correlation with the benchmark's return over day t, on the same row:

| Problem | Original | Fixed |
|---|---|---|
| Features stamped "as of the close of t" and paired with day t's return | 1-day return feature: **1.00**; 36% of features above 0.1 | 0.01; 0% above 0.1 |
| Foreign volatility index joined on the **same date** (it settles at or after the local close) | Its 1-day change: **-0.22**; 7 of 118 volatility-index features above 0.1 | 0.07; 0 above 0.1 |

The second effect is **planted** in the synthetic generator: part of the next local day's move is news that arrives before the foreign close. It shows the mechanism, not its size on real data. That depends on how far the two closes are apart, which varies with the season and the closing auction. The fix does not depend on the size: on local date t, use only foreign closes dated strictly before t. A test confirms that the original truncation test passes on the leaky join, which is why the original suite missed it.

## Leakage and bias audit

| Issue | Effect | Fix |
|---|---|---|
| Features were stamped at the close of t and the alignment to a target was left to the user | A same-row join to day t's return is a direct same-day close leak (correlation 1.00 for the return feature) | `build_features(..., decision_lag=1)` shifts every generated column, and `target_return` gives the matching same-row target; tested |
| The foreign-calendar volatility index was left-joined on the same date and forward filled | Each local row got a foreign close that settles at or after the local close | A strict backward as-of join (`allow_exact_matches=False`); tested at the row level |
| The forward fill of the foreign series had no limit | A long foreign outage became a flat line, with zero returns feeding every rolling volatility-index feature | Staleness limit of 5 calendar days, after which the value is missing; tested |
| No mechanism for publication lags | Any macro or other lagged series added later would be used on its reference date | `MarketSpec.publication_lags` shifts a series by k local rows before any feature is built; tested |
| Breadth averages over whichever indices are reporting, from today's list of indices | Silent jumps in breadth when a member's history starts, and survivorship (discontinued indices are absent) | `Market_N_Members` is published next to every breadth statistic. Survivorship needs historical membership data and is documented, not fixed |
| Same-day close leaks are invisible to a truncation test | False confidence | A perturbation test (scramble rows from t onward, row t must not move) and the audit above |

The original source contained no macro series, so there was no macro publication-lag bug to fix in the data itself. The hook exists so that one cannot be introduced later. The source's design decisions were sound and are kept: Wilder smoothing, scale-free band features, no affine duplicates, a close-only breadth fallback under its own column name, and rolling, not full-sample, regime thresholds.

## Repo layout

```
src/emfeatures/   config.py · merge.py · pipeline.py · audit.py · synthetic.py · utils.py
                  returns · moving_averages · distance_from_ma · volatility · rolling_statistics
                  momentum · relative_strength · breadth · market_regime · technical_indicators
scripts/          build_features.py
tests/            truncation, perturbation, foreign join, staleness, publication lag, AST scan,
                  bounded ranges, OHLC-only families, affine duplicates, data layout
docs/             methodology.md
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q
python scripts/build_features.py --market brazil
python scripts/build_features.py --market india --no-audit
python scripts/build_features.py --market india --csv-dir path/to/your/csvs --out results
```


## License

MIT

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [CHANGELOG](CHANGELOG.md).
