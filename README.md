# PDH/PDL Confluence Reversal — Live Paper Trading

A free, fully automated **paper-trading simulator** for a PDH/PDL (previous
day high/low) confluence reversal strategy on BTCUSDT, 15-minute bars.
Runs entirely on GitHub Actions — no server, no laptop, no exchange
account, and **no real money at any point**.

See [`STATUS.md`](STATUS.md) for the current simulated position, equity,
and recent trades — rewritten every run.

## Current engine: rule-based, tuned for a 12%/year target

- **Parameters**: Zone 0.4% of level, 1H EMA-50 HTF trend filter, minimum
  confluence score 3/6, Swing-based stop loss, 2R/4R partial take-profits.
- **Position size: 60% of equity per trade.** This is the actual lever
  that gets this to a 12%/year target — the underlying edge alone (at a
  conservative 10% sizing) only produces ~2%/year. Backtested directly
  (not extrapolated) at increasing size:

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
- **Backtested**: 489 trades over 4.63 years (2022-2026), profit factor
  1.400 (was 1.501 at 10% sizing - position sizing doesn't change PF in a
  perfectly linear way once compounding and commission drag are in play),
  win rate 48.26%, profitable in 4 of 5 calendar years.

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
