# PDH/PDL Reversal — Live Paper Trading (LLM-decided entries)

_Simulator only. No real money, no exchange account, no API keys. Entries are judged live by a local Ollama model (llama3.2:3b), not the fixed rule-based score. **This decision logic is unvalidated / experimental** - see README. Last updated: 2026-09-01T12:24:12.889627+00:00_

## Current State

- Equity: **100.07 USDT** (started at 100.00)
- Last processed bar: 2026-09-01 12:00:00+00:00
- Position: **flat**

## Live Stats (since switching to LLM-decided entries)

- Total trades: 2
- Win rate: 100.00%
- Total PnL: 0.07 USDT (0.07%)
- Profit factor: inf

## Most Recent LLM Decisions

| Time | Side | Level | Action | Reasoning |
|---|---|---|---|---|
| 2026-09-01T05:30:00.000 | sell | 79250.00 | enter | Volume is above 1.2x its 40-bar average, and the 4H trend is above its EMA-50, indicating a bullish structure that supports the short reversal. The bearish candlestick pattern and chart pattern flags are false, and the RSI is oversold, further supporting the short entry. |

## Reference: prior validated rule-based backtest

See `data/trades_rulebased_archive.csv` - the fixed-rule engine this replaced, validated over 4.67 years of history (355 trades, profit factor 1.483, +4.08%, profitable in all 5 calendar years). That validation does **not** carry over to this LLM-decided engine.
