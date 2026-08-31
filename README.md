# PDH/PDL Confluence Reversal — Live Paper Trading

A free, fully automated **paper-trading simulator** for a PDH/PDL (previous
day high/low) confluence reversal strategy on BTCUSDT, 15-minute bars. Runs
entirely on GitHub Actions — no server, no laptop, no exchange account, and
**no real money at any point**.

See [`STATUS.md`](STATUS.md) for the current simulated position, equity, and
recent trade log — it's rewritten by the bot every 15 minutes.

## How it works

- `paper_trade.py` runs on a GitHub Actions schedule (`.github/workflows/paper_trade.yml`, every 15 minutes).
- Each run pulls any newly-closed 15m BTCUSDT candles from Binance's free public API (no key required) and appends them to `data/btcusdt_15m.parquet`.
- It recomputes indicators over the full dataset and replays the strategy (`backtest_pdh_pdl.step_bar`) over only the bars closed since the last run, continuing from persisted state in `data/state.json` (equity, any open simulated position, pivot history).
- Trades that close are appended to `data/trades.csv`; `STATUS.md` is rewritten with a human-readable summary.
- The workflow commits and pushes the updated data back to this repo.

The **same `step_bar` function** is used both here and for backtesting
(`backtest_pdh_pdl.py`), so live behavior can't silently drift from what was
validated.

## Strategy origin & validation

Ported from a TradingView Pine Script v6 strategy (PDH/PDL zones,
candlestick + pivot-based chart-pattern confirmation, 6-point confluence
scoring, HTF trend filter, breakout protection, RR-based SL/TP). Tuned and
validated against 4.67 years (2022–2026) of free Binance 15m history — see
`data/trades.csv` for the full trade-by-trade record, which starts as that
validated backtest and continues seamlessly into genuine forward paper
trading from the `live_since` timestamp recorded in `data/state.json`.

Current parameters (see `STRATEGY_PARAMS` in `paper_trade.py`): Zone 0.3% of
level, minimum confluence score 5/6, 4H EMA-50 trend filter, Swing-based
stop loss, RR-based take profit (1.5R / 3R partial exits), 10% of equity per
trade, 0.05% commission, 1-tick slippage.

## Running it yourself

```bash
pip install -r requirements.txt
python paper_trade.py
```

No API keys or secrets needed anywhere in this repo.

## Disclaimer

This is a research/educational simulator. Past and simulated performance is
not indicative of future results. Nothing here is financial advice.
