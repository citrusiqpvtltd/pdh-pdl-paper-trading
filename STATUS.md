# PDH/PDL Confluence Reversal — Live Paper Trading (rule-based)

_Simulator only. No real money, no exchange account, no API keys. Deterministic scoring engine (no LLM) - see README for how these parameters and the 60% position size were chosen (target: 12%/year). Last updated: 2026-09-02T22:09:09.141292+00:00_

## Current State

- Equity: **169.43 USDT** (started at 100.00)
- Last processed bar: 2026-09-02 21:45:00+00:00
- Position: **flat**

## All-Time Stats (backtest + live combined)

- Total trades: 489
- Win rate: 48.26%
- Total PnL: 69.43 USDT (69.43%)
- Profit factor: 1.400
- CAGR: 12.05%/year (target: 12%/year)
- Max drawdown: -15.82%

## Most Recent Trades

| Entry time | Side | Exit reason | Entry | Exit | PnL |
|---|---|---|---|---|---|
| 2026-08-30 02:15:00 | short | SL | 78042.49 | 78330.06 | -0.48 |
| 2026-08-28 14:15:00 | long | SL | 78946.09 | 78484.91 | -0.70 |
| 2026-08-23 04:00:00 | long | SL | 77004.31 | 76771.94 | -0.41 |
| 2026-08-19 06:15:00 | long | TP2 | 64277.53 | 64714.95 | 0.30 |
| 2026-08-19 06:15:00 | long | TP1 | 64277.53 | 64496.23 | 0.12 |
| 2026-08-16 15:30:00 | long | SL | 63114.03 | 62968.39 | -0.34 |
| 2026-08-16 00:30:00 | short | SL | 63069.99 | 63125.05 | -0.19 |
| 2026-08-10 01:45:00 | long | SL | 65190.00 | 64792.04 | -0.73 |
| 2026-08-10 00:00:00 | short | SL | 65144.66 | 65300.06 | -0.17 |
| 2026-08-10 00:00:00 | short | TP1 | 65144.66 | 64833.91 | 0.19 |
| 2026-08-09 23:30:00 | short | SL | 64903.03 | 65300.06 | -0.74 |
| 2026-08-09 01:15:00 | short | SL | 64863.02 | 65150.06 | -0.56 |
| 2026-08-03 13:15:00 | long | TP2 | 62769.01 | 64645.20 | 1.48 |
| 2026-08-03 13:15:00 | long | TP1 | 62769.01 | 63707.10 | 0.71 |
| 2026-08-02 00:00:00 | short | SL | 62929.74 | 63127.13 | -0.43 |

## Reference: prior LLM-decided engine (archived)

See `data/trades_llm_archive.csv` / `data/llm_decisions_archive.csv` - the LLM-judgment engine this replaced. Tested at real scale (~2 years, multiple configurations) and consistently showed profit factor 0.50-0.67 (net losing) with no configuration found that produced a real edge. See README for the full investigation.
