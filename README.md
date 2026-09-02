# PDH/PDL Confluence Reversal — Live Paper Trading

A free, fully automated **paper-trading simulator** for a PDH/PDL (previous
day high/low) confluence reversal strategy on BTCUSDT, 15-minute bars.
Runs entirely on GitHub Actions — no server, no laptop, no exchange
account, and **no real money at any point**.

See [`STATUS.md`](STATUS.md) for the current simulated position, equity,
and recent trades — rewritten every run.

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
