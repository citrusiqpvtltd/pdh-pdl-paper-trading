# PDH/PDL Reversal — Live Paper Trading (LLM-decided entries)

_Simulator only. No real money, no exchange account, no API keys. Entries are judged live by a local Ollama model (qwen2.5:7b), not the fixed rule-based score. **This decision logic is unvalidated / experimental** - see README. Last updated: 2026-09-02T13:08:44.221431+00:00_

## Current State

- Equity: **100.04 USDT** (started at 100.00)
- Last processed bar: 2026-09-02 12:45:00+00:00
- Position: **flat**

## Live Stats (since switching to LLM-decided entries)

- Total trades: 5
- Win rate: 60.00%
- Total PnL: 0.04 USDT (0.04%)
- Profit factor: 1.834

## Most Recent LLM Decisions

| Time | Side | Level | Action | Reasoning |
|---|---|---|---|---|
| 2026-09-02T12:45:00.000 | buy | 76420.00 | skip | No bullish candlestick pattern or chart pattern supports the potential buy, and the 4H downtrend fights the trade direction. |
| 2026-09-02T09:15:00.000 | buy | 76420.00 | skip | The market structure is bullish, but the 4H trend is down, fighting the trade. There are no confirming bullish candlestick or chart patterns, and momentum is not turning up. |
| 2026-09-01T16:15:00.000 | buy | 77392.00 | enter | RSI turning up from oversold (39.2) supports the long reversal, and volume above 1.2x its 40-bar average confirms the setup, while the 4H trend filter (below its EMA-50) and lack of clear structure favor the long direction. |
| 2026-09-01T13:00:00.000 | buy | 77392.00 | enter | Multiple confirmations support the long reversal: bearish candlestick pattern, volume above 1.2x its 40-bar average, and 4H trend filter above its EMA-50, with no contradictory signals present. |
| 2026-09-01T05:30:00.000 | sell | 79250.00 | enter | Volume is above 1.2x its 40-bar average, and the 4H trend is above its EMA-50, indicating a bullish structure that supports the short reversal. The bearish candlestick pattern and chart pattern flags are false, and the RSI is oversold, further supporting the short entry. |

## Reference: prior validated rule-based backtest

See `data/trades_rulebased_archive.csv` - the fixed-rule engine this replaced, validated over 4.67 years of history (355 trades, profit factor 1.483, +4.08%, profitable in all 5 calendar years). That validation does **not** carry over to this LLM-decided engine.
