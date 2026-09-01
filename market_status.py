"""
Broader crypto market-status context, fed into the LLM's decision alongside
the BTCUSDT-specific technicals - free, no API key required for either
source (CoinMarketCap's API requires a paid-signup key we don't have; these
are genuinely free, no-signup equivalents):

  - Fear & Greed Index (alternative.me) - the standard crypto market-wide
    sentiment gauge, 0 (Extreme Fear) to 100 (Extreme Greed).
  - CoinGecko global market data - total crypto market cap and BTC's share
    of it (dominance).

Fetched once per run (not once per decision - this is slow-moving,
market-wide context, not something that changes bar to bar). Fails soft:
any network error just means this section is omitted from the LLM's
context, never crashes the run or blocks a trading decision.
"""
import sys
from datetime import datetime, timezone

import requests

FNG_URL = "https://api.alternative.me/fng/?limit=1"
FNG_HISTORY_URL = "https://api.alternative.me/fng/?limit=0"  # full history, free, no key
COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"


def fetch_market_status() -> dict:
    status = {}

    try:
        r = requests.get(FNG_URL, timeout=10)
        r.raise_for_status()
        d = r.json()["data"][0]
        status["fear_greed_value"] = int(d["value"])
        status["fear_greed_label"] = d["value_classification"]
    except Exception as e:
        print(f"Fear & Greed Index fetch failed ({e!r}) - omitting from context.", file=sys.stderr)

    try:
        r = requests.get(COINGECKO_GLOBAL_URL, timeout=10)
        r.raise_for_status()
        d = r.json()["data"]
        status["total_market_cap_usd"] = d["total_market_cap"]["usd"]
        status["btc_dominance_pct"] = d["market_cap_percentage"]["btc"]
        status["market_cap_change_24h_pct"] = d["market_cap_change_percentage_24h_usd"]
    except Exception as e:
        print(f"CoinGecko global data fetch failed ({e!r}) - omitting from context.", file=sys.stderr)

    return status


def fetch_fear_greed_history() -> dict:
    """Full daily Fear & Greed history (free, no key, back to Feb 2018) as
    {date_str "YYYY-MM-DD": (value:int, label:str)}. For point-in-time-
    correct historical backtesting - NOT "today's" value applied to every
    replayed bar, which would be lookahead bias at any real scale."""
    try:
        r = requests.get(FNG_HISTORY_URL, timeout=30)
        r.raise_for_status()
        out = {}
        for row in r.json()["data"]:
            d = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d")
            out[d] = (int(row["value"]), row["value_classification"])
        return out
    except Exception as e:
        print(f"Fear & Greed history fetch failed ({e!r}) - historical backtest will omit market status.", file=sys.stderr)
        return {}


def format_for_context_historical(date_str: str, fg_history: dict) -> str:
    """Point-in-time market status for a historical backtest bar. Only Fear
    & Greed is available historically for free; CoinGecko's global market
    cap/BTC dominance history requires a paid plan, so those are omitted
    here (they're still used, current-only, in the live bot)."""
    if date_str not in fg_history:
        return ""
    value, label = fg_history[date_str]
    return f"Broader crypto market status (as of {date_str}):\n  - Fear & Greed Index: {value}/100 ({label})"


def format_for_context(status: dict) -> str:
    if not status:
        return "Broader market status: unavailable this run."
    lines = ["Broader crypto market status:"]
    if "fear_greed_value" in status:
        lines.append(f"  - Fear & Greed Index: {status['fear_greed_value']}/100 ({status['fear_greed_label']})")
    if "total_market_cap_usd" in status:
        lines.append(f"  - Total crypto market cap: ${status['total_market_cap_usd']/1e9:,.1f}B "
                      f"({status['market_cap_change_24h_pct']:+.2f}% in 24h)")
    if "btc_dominance_pct" in status:
        lines.append(f"  - BTC dominance: {status['btc_dominance_pct']:.1f}% of total market cap")
    return "\n".join(lines)
