# PDH/PDL Confluence Reversal — Live Paper Trading (rule-based + ML secondary filter)

_Simulator only. No real money, no exchange account, no API keys. Deterministic score>=3+HTF rule, entries additionally gated by a trained ML filter (ml_entry_filter.joblib, threshold 0.55) - see README for how these parameters, the 60% position size, and the ML gate were chosen (target: 12%/year). Last updated: 2026-09-03T00:12:05.914957+00:00_

## Current State

- Equity: **146.93 USDT** (started at 100.00)
- Last processed bar: 2026-09-02 23:45:00+00:00
- Position: **flat**

## All-Time Stats (backtest + live combined)

- Total trades: 261
- Win rate: 57.85%
- Total PnL: 46.93 USDT (46.93%)
- Profit factor: 2.113
- CAGR: 8.66%/year (target: 12%/year)
- Max drawdown: -4.22%

## Most Recent Trades

| Entry time | Side | Exit reason | Entry | Exit | PnL |
|---|---|---|---|---|---|
| 2026-08-30 02:15:00 | short | SL | 78042.49 | 78330.06 | -0.41 |
| 2026-08-10 01:45:00 | long | SL | 65190.00 | 64792.04 | -0.63 |
| 2026-08-10 00:00:00 | short | SL | 65144.66 | 65300.06 | -0.15 |
| 2026-08-10 00:00:00 | short | TP1 | 65144.66 | 64833.91 | 0.17 |
| 2026-08-09 01:15:00 | short | SL | 64863.02 | 65150.06 | -0.48 |
| 2026-08-02 00:00:00 | short | SL | 62929.74 | 63127.13 | -0.37 |
| 2026-07-29 22:30:00 | short | SL | 64015.36 | 64648.85 | -0.98 |
| 2026-07-29 01:30:00 | short | SL | 63811.35 | 64100.06 | -0.50 |
| 2026-07-26 01:45:00 | short | SL | 64461.99 | 64475.34 | -0.11 |
| 2026-07-20 01:00:00 | long | SL | 64808.01 | 64347.83 | -0.73 |
| 2026-07-13 00:00:00 | short | TP2 | 64294.99 | 63457.86 | 0.54 |
| 2026-07-13 00:00:00 | short | TP1 | 64294.99 | 63876.43 | 0.25 |
| 2026-06-29 12:30:00 | short | SL | 59951.67 | 60346.33 | -0.34 |
| 2026-06-29 12:30:00 | short | TP1 | 59951.67 | 59162.40 | 0.55 |
| 2026-06-17 01:30:00 | long | SL | 65759.77 | 65559.94 | -0.18 |

## Reference: prior LLM-decided engine (archived)

See `data/trades_llm_archive.csv` / `data/llm_decisions_archive.csv` - the LLM-judgment engine this replaced. Tested at real scale (~2 years, multiple configurations) and consistently showed profit factor 0.50-0.67 (net losing) with no configuration found that produced a real edge. See README for the full investigation.
