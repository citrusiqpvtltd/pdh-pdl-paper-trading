# PDH/PDL Reversal — Live Paper Trading (LLM-decided entries)

A free, fully automated **paper-trading simulator** for a PDH/PDL (previous
day high/low) reversal strategy on BTCUSDT, 15-minute bars. Runs entirely
on GitHub Actions — no server, no laptop, no exchange account, and **no
real money at any point**.

See [`STATUS.md`](STATUS.md) for the current simulated position, equity,
and the LLM's recent trade-entry reasoning — rewritten every run.

## ⚠️ This is an experiment, not a validated strategy

This repo previously ran a fixed, rule-based confluence-scoring strategy
that was backtested and validated over 4.67 years of history (355 trades,
profit factor 1.483, +4.08%, profitable in all 5 calendar years — preserved
in `data/trades_rulebased_archive.csv` for reference). It has been
**replaced** with an LLM making the entry judgment call instead of the
fixed score threshold.

That replacement has **not** been backtested — doing so would mean
replaying thousands of historical setups through a real LLM call each,
which is a separate, larger effort. Everything in `data/trades.csv` from
here forward is a live forward-test with no historical validation behind
it. Watch `STATUS.md` over time to see how it actually performs.

## How it works

- `paper_trade.py` runs on a GitHub Actions schedule (`.github/workflows/paper_trade.yml`).
- Each run pulls any newly-closed 15m BTCUSDT candles from Binance's free public market-data API and appends them to `data/btcusdt_15m.parquet`.
- It recomputes technical indicators over the full dataset (ATR, RSI, MACD, pivots, candlestick/chart pattern proxies, market structure, 4H EMA trend filter — via `backtest_pdh_pdl.compute_indicators`, the same code the archived backtest used).
- Whenever price is testing yesterday's high or low **and the bot is flat**, it hands that technical context — patterns, structure, volume, momentum, HTF trend, and the last 10 candles — to a **local Ollama model** (`llama3.2:3b`, installed fresh each run, no API key, no cost) and asks it to judge: is this specific setup worth taking, or skip? The model does not choose direction (fixed by which level is being tested) and does not set prices.
- If it says "enter", the same deterministic, previously-validated math computes the actual stop-loss and take-profit levels (Swing-based SL, 1.5R/3R partial take-profits) and manages the exit every subsequent bar — only the entry judgment moved to the model.
- Trades and every LLM decision (including skips, with reasoning) are logged to `data/trades.csv` and `data/llm_decisions.csv`; `STATUS.md` is rewritten with a human-readable summary.
- The workflow commits and pushes the updated data back to this repo.

## Why Ollama instead of the Claude/OpenAI APIs

This needed to run with no ongoing machine kept on and no per-call cost.
Since there's no persistent host to run Ollama's daemon on, each CI run
installs Ollama fresh, pulls the model, runs one inference, and tears the
whole thing down — which is slower and heavier per run than an API call
would be, and a 3B local model reasons more shallowly than a hosted
frontier model. That tradeoff (free + local-in-spirit vs. faster + sharper
judgment for a few dollars a month) was a deliberate choice, not a
technical requirement — the architecture would look the same with the
Ollama-specific block in `paper_trade.py`/`llm_decide.py` swapped for a
hosted API call.

## GitHub Actions scheduling caveat

`schedule`-triggered workflows are best-effort — GitHub deprioritizes them
under load, and this repo has seen the configured `*/15 * * * *` cron
actually fire every 3–4 hours in practice, especially for a newly-added
workflow. This does not affect correctness: every run processes *every*
bar closed since its last run, not just the latest one, so a late or
skipped tick loses no data — only means `STATUS.md` is "as of the last
run" rather than near-real-time.

## Running it yourself

```bash
pip install -r requirements.txt
# separately: install Ollama (https://ollama.com) and `ollama pull llama3.2:3b`
python paper_trade.py
```

No API keys or secrets needed anywhere in this repo.

## Disclaimer

This is a research/educational simulator. Past and simulated performance
is not indicative of future results. Nothing here is financial advice.
