# PDH/PDL Reversal — Live Paper Trading (LLM-decided entries)

_Simulator only. No real money, no exchange account, no API keys. Entries are judged live by a local Ollama model (llama3.2:3b), not the fixed rule-based score. **This decision logic is unvalidated / experimental** - see README. Last updated: 2026-09-01T07:00:37.730167+00:00_

## Current State

- Equity: **100.00 USDT** (started at 100.00)
- Last processed bar: 2026-09-01 06:30:00+00:00
- Position: **SHORT** 0.000126 BTC @ 79167.50 (SL 79435.46, TP1 78765.59, TP2 78363.67)
  - LLM's reasoning at entry: _Volume is above 1.2x its 40-bar average, and the 4H trend is above its EMA-50, indicating a bullish structure that supports the short reversal. The bearish candlestick pattern and chart pattern flags are false, and the RSI is oversold, further supporting the short entry._

No trades closed yet under the LLM-decided engine.

## Most Recent LLM Decisions

| Time | Side | Level | Action | Reasoning |
|---|---|---|---|---|
| 2026-09-01T05:30:00.000 | sell | 79250.00 | enter | Volume is above 1.2x its 40-bar average, and the 4H trend is above its EMA-50, indicating a bullish structure that supports the short reversal. The bearish candlestick pattern and chart pattern flags are false, and the RSI is oversold, further supporting the short entry. |

## Reference: prior validated rule-based backtest

See `data/trades_rulebased_archive.csv` - the fixed-rule engine this replaced, validated over 4.67 years of history (355 trades, profit factor 1.483, +4.08%, profitable in all 5 calendar years). That validation does **not** carry over to this LLM-decided engine.
