# Methodology and timing audit

## Scope

This is a feature library, not a strategy. It turns a panel of daily index prices into a wide, non-redundant, causal feature matrix that a model can use. Nothing here is a performance claim.

## Data contract

One frame per index, with `Date` plus whichever of `Open, High, Low, Close` the source publishes. After merging, columns are `{NAME}_{Field}` on one row per local trading day. Index names are never hardcoded in the feature modules: they are discovered from `*_Close` columns. Market specifics are in a `MarketSpec`:

| Field | Meaning |
|---|---|
| `benchmark` | Index the relative-strength features are measured against |
| `vol_index` | Volatility index. It gets per-index features and drives the VIX-coupling features, but is held out of breadth and cross-sectional aggregates |
| `foreign_after_close` | Series on another exchange's calendar that settle at or after the local close |
| `publication_lags` | Series published k local rows after their reference date |
| `max_staleness_days` | Maximum age of a foreign value before it is treated as missing |

Two presets ship: a 31-index panel (all OHLC, with the volatility index on the same exchange) and a 20-index panel (19 close-only indices plus a foreign-calendar OHLC volatility index that starts about half-way through the sample). The synthetic generator reproduces both layouts exactly: index names, fields per index, calendars and the late start. The original data files were not copied.

## Timing convention

1. **Merge.** Local series define the spine (outer merge). Foreign series are attached with `merge_asof(direction="backward", allow_exact_matches=False, tolerance=max_staleness_days)`. On local date t they carry the latest foreign value dated strictly before t, and nothing older than the staleness limit. Publication-lagged series are then shifted.
2. **Modules.** Every module computes row t from rows up to t: trailing `rolling` (never `center=True`), `ewm`, `pct_change`, `diff`, non-negative `shift`, and cross-sectional statistics over a single row.
3. **Decision lag.** All generated columns are shifted by `decision_lag` (default 1). A feature on row t then uses data up to t-1 and can be joined to `target_return` on the same row (the close-to-close return over day t).

## Tests

| Test | What it proves |
|---|---|
| Truncation (both presets, lag 0 and 1) | Deleting the last 60 rows changes no earlier value: no negative shift, centred window, full-sample statistic or backward fill |
| Perturbation (lag 1) | Scrambling every input from row t onward leaves every feature on row t unchanged, so the decision lag really removes day t |
| Foreign join | Changing a foreign close dated d leaves local row d unchanged and changes the next local row |
| Staleness | A 20-day foreign gap becomes missing after 5 days rather than flat |
| Publication lag | A lagged series equals the unlagged one shifted by k |
| AST scan | No negative `shift`, `center=True`, `bfill`/`backfill` or forward/nearest as-of join anywhere in the package |
| Audit | The same-day leak gives correlation 1.0 at lag 0 and about 0 at lag 1. The same-date foreign join leaks on the planted generator, and the strict join does not |
| Truncation blind spot | The truncation test **passes** on the leaky same-date join, which documents why it is not sufficient on its own |
| Correctness | RSI and stochastic stay in 0-100, breadth shares in 0-1, regime flags are 0/1, efficiency ratios are in 0-1, SMA and distance match direct computation, and there are no infinities |
| Structure | OHLC-only families appear exactly where OHLC exists. The close-only breadth fallback uses its own column names. No overwrite. No two features are affine restatements (z-score hash). ROC never duplicates a return window |

## Audit findings

### 1. Same-day close (fixed)

The original modules stamp each value at the close of t, which is right for "as of t". The pipeline wrote that matrix out and left the alignment to the modelling stage. A same-row join with day t's return then pairs `Return_1D` with itself (correlation 1.00). On synthetic data, 36% of all features correlate above 0.1 with the same-row return at lag 0, against 0% at lag 1. The default is now lag 1.

### 2. Foreign volatility index on the same date (fixed)

In the close-only panel, the volatility index is computed from options on a foreign-listed ETF and settles at the foreign close. Depending on the season and the local closing auction, that is at or after the local close, so the same-date value cannot be assumed known when the local close prints. The original merger left-joined it on the same date. The fix uses the latest value dated strictly before the local date. On the synthetic generator, where I planted the mechanism (part of the next local move is news that arrives before the foreign close), the same-date join gives the volatility index's 1-day change a -0.22 correlation with the next local return, and the strict join about 0. The real-data size depends on the overlap window. I did not measure it on real data and do not claim one.

### 3. Unlimited forward fill (fixed)

Forward filling across foreign holidays is causal. But it creates zero-return days (the original notes counted 25 of them, about 3% of the covered rows), and with no limit, a long outage would look like a perfectly calm market. As-of joins now stop after 5 calendar days.

### 4. Publication lags (hook added)

The original panels contained no macro series, so there was nothing to fix in the data. The audit asked for it explicitly, so `publication_lags` exists and is tested. Any series added later is shifted before any feature sees it.

### 5. Breadth composition and survivorship (documented, partly mitigated)

Breadth and cross-sectional statistics average over whichever indices report on a given day, from today's index list. New members cause jumps, and discontinued indices are missing (one index in the original list was already unavailable from the source endpoint). `Market_N_Members` makes the composition visible. A real fix needs historical membership, which is out of scope.

### Kept from the original design

- Wilder smoothing (`alpha = 1/period`) for RSI, ATR and ADX.
- Scale-free forms for band indicators (width and position, not price-level bands).
- Removal of affine duplicates: no annualised vol, Williams %R, MACD signal, rolling std/var or cumulative return.
- A close-only breadth fallback under distinct column names, so a close-based "new high" is never mixed with an intraday one.
- Volatility regimes ranked against a rolling 252-day history, not the full sample.

### Known limitations

- `Momentum_{w}`, `Acceleration_{w}`, raw ATR, MACD, SMA/EMA levels and `RS_Ratio` are price-scale and non-stationary. They are kept because they are the standard definitions. The scale-free versions (`Return_{w}D`, `Distance_SMA{w}`, `ATRP`) are usually better model inputs.
- Warm-up rows are kept as NaN. Trimming them is a modelling decision.
- The volatility-index features are NaN before that index starts.

## What was not ported

- The data files (index CSVs and parquet outputs), which are third-party data and were not copied in any form.
- The download scripts for the original data sources, and the project's design notes.
- The original per-market duplicate code: two near-identical packages became one, parameterised by `MarketSpec`.
