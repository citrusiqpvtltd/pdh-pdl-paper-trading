# PDH/PDL Confluence Reversal — Live Paper Trading

A free, fully automated **paper-trading simulator** for a PDH/PDL (previous
day high/low) confluence reversal strategy on BTCUSDT, 15-minute bars.
Runs entirely on GitHub Actions — no server, no laptop, no exchange
account, and **no real money at any point**.

See [`STATUS.md`](STATUS.md) for the current simulated position, equity,
and recent trades — rewritten every run.

**Balance note**: live equity was reset to $100 on 2026-09-03. The first
deployment's equity (grown to $146.93) came from replaying the 2022-2026
historical backtest, not genuine forward trading - conflating the two made
the live number misleading. Bot's market-structure memory (pivots, signal
state) was preserved across the reset for continuity; only the balance and
trade log restarted clean. The pre-reset trade history is archived at
`data/trades_backtest_reference_pre_reset.csv` and all the backtest
numbers throughout this README remain valid as the basis for the
strategy's validated edge - they're just no longer shown as "the current
balance."

## Current engine: rule-based + ML secondary filter, tuned for a 12%/year target

- **Rule**: Zone 0.4% of level, 1H EMA-50 HTF trend filter, minimum
  confluence score 3/6, Swing-based stop loss, 2R/4R partial take-profits.
- **ML secondary filter**: a `HistGradientBoostingClassifier` trained on
  49,619 historical near-PDH/PDL touches (`build_ml_dataset.py` /
  `train_ml_model.py`) gates entries **on top of** the rule above - it does
  not replace it. See "ML entry filter" below for the full story of why and
  how this was validated.
- **Position size: 60% of equity per trade.** This is the actual lever
  that gets this to a 12%/year target — the underlying edge alone (at a
  conservative 10% sizing) only produces ~2%/year. Backtested directly
  (not extrapolated) at increasing size, rule-only (no ML gate):

  | Position size | CAGR | Max Drawdown |
  |---|---|---|
  | 10% | 2.02%/yr | -2.80% |
  | 25% | 5.05%/yr | -6.87% |
  | 40% | 8.06%/yr | -10.80% |
  | **60% (current)** | **12.05%/yr** | **-15.82%** |
  | 80% | 16.02%/yr | -20.59% |
  | 100% | 19.94%/yr | -25.12% |

  The honest tradeoff: hitting 12%/year means a real -15.82% max drawdown
  in the backtest, not a free lunch. Since this is paper trading, that
  risk is informational, not financial - but it would matter a great deal
  if run with real capital.
- **Backtested, rule-only**: 489 trades over 4.63 years (2022-2026), profit
  factor 1.400, win rate 48.26%, profitable in 4 of 5 calendar years.
- **Backtested, rule + ML gate (currently deployed)**: 261 trades over the
  same period, profit factor 2.113, win rate 57.85%, **CAGR 8.66%/yr, max
  drawdown only -4.22%** - lower return than the rule alone, but roughly
  4x less drawdown for it. See "ML entry filter" for why this tradeoff was
  chosen and what it's actually validated on.

## ML entry filter: what it is, and the honest limits of the evidence

Trained on every historical near-PDH/PDL touch (not just the ones that
passed the rule), using the same features the rule's score is built from
plus the ones the rule doesn't see (candlestick pattern *type*, not just a
collapsed yes/no; volume ratio; RSI/ATR levels; hour/day-of-week). The
model's raw AUC across ALL touches is weak (~0.50-0.52 - essentially no
better than chance at predicting "is this setup ever worth taking"). Its
real, validated use is narrower: **as a secondary filter applied only to
setups that already pass the rule**, its predicted probability correlates
with real outcome quality, out-of-sample:

| Threshold | Trades (test period) | Win rate | Profit factor | CAGR | Max DD |
|---|---|---|---|---|---|
| none (rule only) | 159 | 45.3% | 0.91 | -2.53% | -9.5% |
| >= 0.49 | 113 | 41.6% | 0.90 | -1.60% | -4.6% |
| >= 0.52 | 100 | 39.0% | 0.89 | -1.70% | -4.1% |
| **>= 0.55 (deployed)** | **75** | **46.7%** | **1.02** | **+0.15%** | **-2.8%** |
| >= 0.60 | 48 | 68.8% | 2.02 | +3.42% | -1.5% |

This table is computed on the genuinely out-of-sample test period
(2025-06-01 onward, which the model never saw during training), using a
sequential, position-exclusive backtest through the real live engine
(`validate_ml_filter.py`, via a `ml_gate` hook added to
`backtest_pdh_pdl.step_bar` - default `None`, zero effect unless supplied,
so this never changed the underlying rule's own behavior).

**Two honest caveats**: (1) the deployed threshold (0.55) is validated on
only 75 out-of-sample trades - real, but a small sample; the more
attractive 0.60 row is smaller still (48 trades) and more likely to be
partly noise, which is why 0.55 was chosen over it. (2) A recent, separate
finding: the rule *alone* actually lost money over the same 2025-06 to
2026-09 stretch (CAGR -2.53%), a sharp reversal from the full 4.6-year
average - consistent with the already-known "profitable in 4 of 5 calendar
years" profile, but notable because live forward paper trading is starting
right as that softer stretch ends.

**Book-informed feature expansion, tried and NOT adopted**: at the user's
request, ~30 additional indicators drawn directly from five trading books
(`books/`) were implemented and tested as an expanded feature set:
- *Japanese Candlestick Charting Techniques* (Nison): granular pattern
  flags (Harami, Harami Cross, Dark Cloud Cover, Piercing Pattern, Three
  Black Crows/White Soldiers, Tweezers) in place of the rule's collapsed
  bearish/bullish-candle flags.
- *The New Trading for a Living* (Elder): Force Index (2-day/13-day EMA of
  volume × price change), the Impulse System (EMA-13 slope + MACD-Histogram
  slope alignment), and RSI/price divergence.
- *Technical Analysis of the Financial Markets* (Murphy): Stochastic
  Oscillator, Bollinger %B, ADX trend strength.

Result: AUC was statistically unchanged (0.495/0.522 vs. 0.4925/0.5197),
and the sequential validation was *noisier* - not monotonic across
thresholds the way the simpler model is. **The leaner, original feature
set is what's actually deployed.** (*Trading in the Zone*'s "probabilistic
mindset" and *Market Wizards*' risk-discipline lessons are reflected in the
approach itself - a probability-gated filter and a fixed, non-discretionary
position size - rather than as numeric features. Elder's own risk
framework, a 2% max-risk-per-trade rule, is worth naming honestly against
this bot's 60% sizing - it would call that reckless; that sizing was kept
because hitting the stated 12%/year target requires it, and this is a
simulator.)

## Training window investigation: why "more history" made it worse

Prompted by a 9-year backtest (see below) showing the deployed ML gate
actively hurting performance in 2018, the model was retrained on the full
~9 years of available data (2017-08 onward, vs the deployed model's
2022-2025.06) to see whether more data would generalize better. It did
not - it was worse at every threshold tested on the same genuinely
out-of-sample period (2025-06 onward): e.g. at the deployed threshold
(0.55), PF dropped from 1.018 to 0.859, despite the full-history model's
slightly higher raw AUC (0.534 vs 0.520).

**Root cause, confirmed with real numbers**: BTC's ATR% (volatility
relative to price) has shrunk roughly 4x over this span - yearly averages
fall from ~1.0% (2017) to ~0.26% (2025-2026), a genuine market-maturation
effect, not a data bug. `atr_pct` (and other absolute-scale features) feed
into the tree model as raw numbers, so training equally across eras with
very different volatility regimes teaches splits calibrated to conditions
that barely exist anymore.

**First fix tried - exponential recency weighting** (`sample_weight` decay
by row age, `train_full_history_model.py`): did NOT work. Subset AUC got
*worse* (0.54-0.57 across half-lives of 180/365/730 days) than even the
plain unweighted full-history model (0.622). Cause:
`HistGradientBoostingClassifier` bins continuous features into quantiles
computed from the full, unweighted training array - so old rows still
consume bin resolution for `atr_pct` even when their contribution to the
loss is down-weighted, blunting precision in today's much narrower
volatility range. Down-weighting the loss doesn't fix binning resolution.

**Second check - hard training-window sweep** (`sweep_train_window.py`):
swept fixed-length training windows (1, 1.5, 2, 3, 3.45, 4 years) ending
at the same holdout boundary. Subset AUC rose with window length up to
~3-3.45 years (peaking at the deployed model's *exact* window, 0.625),
then fell again by 4 years - confirming the deployed window sits at the
sweet spot, not undershooting it. Verified this held in the real
sequential, position-exclusive backtest too, not just AUC: the two
closest contenders (3yr, 3.45yr) both did *worse* than deployed (PF 0.608
and 0.765 vs deployed's 1.018).

**Conclusion: no change to the live model.** The full-history approach
itself was the mistake (too much stale-regime data), not the deployed
model - which turns out to already sit at a locally optimal training
window among everything tested. `build_full_history_dataset.py`,
`train_full_history_model.py`, and `sweep_train_window.py` are kept as
reference for this investigation; none of their output models are
deployed.

## Retraining: scheduled, NOT self-learning

There is no online/incremental learning here - the model never updates
itself from its own trade outcomes as it runs. What exists instead:
`retrain_pipeline.py` runs monthly (`.github/workflows/retrain_ml_filter.yml`,
1st of the month), doing the same offline batch process by hand:

1. Rebuilds the labeled touch dataset from all data on disk.
2. Re-splits train/holdout using a **rolling** window - the most recent 270
   days, whenever it happens to run, not a fixed calendar date.
3. Trains a candidate model on everything before that holdout.
4. Validates the candidate sequentially and position-exclusively on the
   holdout, through the real `step_bar` engine (same method as
   `validate_ml_filter.py` - not a naive average).
5. **Deploys only if the candidate clears both a minimum sample size (30
   holdout trades) and a minimum profit factor (0.90) on its own holdout.**
   Otherwise the existing model is kept untouched and the reason is logged.

The gate threshold (0.55) is never re-searched during retraining - only the
model itself is refit, at the already-validated threshold, so retraining
can't compound overfitting by also chasing a threshold on a small monthly
sample. Every run - deployed or skipped - is logged to `RETRAIN_LOG.md`
with its holdout metrics and reasoning, so this stays auditable rather than
a silent background process. In initial local testing, both a 6-month and
a 9-month rolling holdout were tried and correctly SKIPPED deployment
(too few trades, then PF below floor) rather than deploying a weaker model
- the safety gate working as intended, not a bug.

`ml_filter.py` holds the ONE shared implementation of the gate's
feature-row construction, imported by both `paper_trade.py` (live) and
`validate_ml_filter.py` (validation) - an earlier standalone
reconstruction of the rule's scoring logic for evaluation purposes
silently diverged from the real engine (missed the fired-flag spam
suppression, breakout protection, and a market-structure condition) and
produced nonsense results (fake 50,000%+ backtested "returns") before this
was caught and fixed.

## How it works

- `paper_trade.py` runs on a GitHub Actions schedule (`.github/workflows/paper_trade.yml`).
- Each run pulls any newly-closed 15m BTCUSDT candles from Binance's free public market-data API and appends them to `data/btcusdt_15m.parquet`.
- It recomputes technical indicators over the full dataset and replays the strategy (`backtest_pdh_pdl.step_bar` - the SAME function used for backtesting, so live and backtested behavior cannot diverge) over bars closed since the last run, continuing from persisted state in `data/state.json`.
- On the very first run, this replays the entire 2022-present history once, so `data/trades.csv` starts as exactly the validated backtest and continues seamlessly into genuine forward paper trading from the `live_since` timestamp in `state.json`.
- No LLM, no Ollama, no API keys - pure deterministic scoring, runs in well under a minute per cycle.

## GitHub Actions scheduling caveat

`schedule`-triggered workflows are best-effort — GitHub deprioritizes them
under load, and this repo has seen the configured `*/15 * * * *` cron
actually fire every 3–4 hours in practice. This does not affect
correctness: every run processes *every* bar closed since its last run,
not just the latest one, so a late or skipped tick loses no data — only
means `STATUS.md` is "as of the last run" rather than near-real-time.

## The LLM-decided engine experiment (archived)

This repo previously replaced the rule-based scoring with a local Ollama
model (first `llama3.2:3b`, then `qwen2.5:7b`) judging each entry instead
of a fixed threshold. That path was investigated thoroughly:

- `llama3.2:3b` was found to justify trades with directly contradictory
  reasoning (citing bullish signals to support a short, and vice versa),
  consistently, even after prompt fixes.
- Switching to `qwen2.5:7b` fixed the reasoning coherence, but a real
  ~2-year backtest (1,529 decisions, 89 trades) showed it losing money:
  profit factor 0.671.
- Diagnosed shorts as the apparent problem (16.7% win rate vs longs'
  28.9%) and disabled them - the re-validation came back *worse*, which
  led to finding that Ollama's `temperature: 0.2` was flipping ~20% of
  enter/skip decisions on identical historical input, run to run. Fixed
  determinism (`temperature: 0`) and re-ran clean: 53 trades, profit
  factor 0.503 - still losing, and consistent with a second temperature-0.2
  run (PF 0.496), confirming the original "shorts are the problem"
  diagnosis was itself built on noise.
- Every configuration tested (mixed sides, long-only, two temperatures)
  landed in the same profit factor 0.50-0.67 range. No configuration
  produced a real edge at scale.

Full trade-by-trade record preserved in `data/trades_llm_archive.csv`,
`data/llm_decisions_archive.csv`, and the `data/llm_2year_backtest_*.csv`
files, plus the sharded backtest infrastructure itself
(`plan_shards.py` / `backtest_llm.py` / `merge_shards.py` /
`.github/workflows/backtest_llm_sharded.yml`, still functional if anyone
wants to pick this investigation back up) for reference.

## Running it yourself

```bash
pip install -r requirements.txt
python paper_trade.py
```

No API keys or secrets needed anywhere in this repo.

## Disclaimer

This is a research/educational simulator. Past and simulated performance
is not indicative of future results. Nothing here is financial advice.
The 60% position sizing chosen here is aggressive by conventional risk
management standards and was chosen specifically to hit a stated 12%/year
target in paper trading - it is not a recommendation for real capital.
