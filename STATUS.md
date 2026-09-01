# PDH/PDL Reversal — Live Paper Trading (LLM-decided entries)

_Simulator only. No real money, no exchange account, no API keys. Entries are judged live by a local Ollama model (llama3.2:3b), not the fixed rule-based score. **This decision logic is unvalidated / experimental** - see README. Last updated: 2026-09-01T01:56:01.168689+00:00_

## Current State

- Equity: **100.00 USDT** (started at 100.00)
- Last processed bar: 2026-09-01 01:30:00+00:00
- Position: **flat**

No trades closed yet under the LLM-decided engine.


## Reference: prior validated rule-based backtest

See `data/trades_rulebased_archive.csv` - the fixed-rule engine this replaced, validated over 4.67 years of history (355 trades, profit factor 1.483, +4.08%, profitable in all 5 calendar years). That validation does **not** carry over to this LLM-decided engine.
