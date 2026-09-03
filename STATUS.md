# PDH/PDL Confluence Reversal — Live Paper Trading (rule-based + ML secondary filter)

_Simulator only. No real money, no exchange account, no API keys. Deterministic score>=3+HTF rule, entries additionally gated by a trained ML filter (ml_entry_filter.joblib, threshold 0.55) - see README for how these parameters, the 60% position size, and the ML gate were chosen (target: 12%/year). Last updated: 2026-09-03T01:29:30.381256+00:00_

## Current State

- Equity: **100.00 USDT** (started at 100.00)
- Last processed bar: 2026-09-03 01:00:00+00:00
- Position: **flat**

## Genuine Live Stats (since balance reset)

No trades yet since the reset - equity is exactly the $100 starting balance.


## Balance reset

Equity was reset to $100 (from $146.93) because that growth came from replaying the 2022-2026 historical backtest on first deployment, not from genuine forward trading. Pivot/signal state (market-structure memory) was preserved across the reset for signal continuity - only the balance and trade log restarted clean. The full backtest-validated trade history (532 trades over ~9 years at various configurations) remains in `data/trades_backtest_reference_pre_reset.csv` and is documented in full in README.md - it's still the basis for believing this strategy has an edge, just no longer conflated with the live dashboard number.


## Reference: prior LLM-decided engine (archived)

See `data/trades_llm_archive.csv` / `data/llm_decisions_archive.csv` - the LLM-judgment engine this replaced. Tested at real scale (~2 years, multiple configurations) and consistently showed profit factor 0.50-0.67 (net losing) with no configuration found that produced a real edge. See README for the full investigation.
