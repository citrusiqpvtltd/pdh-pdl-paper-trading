"""
LLM-based trade-entry judgment via a local Ollama model.

Replaces the fixed 6-point confluence score threshold with an actual model
judgment call: given the same technical context (patterns, structure,
volume, momentum, HTF trend, recent candles), decide whether THIS specific
PDH/PDL reversal setup is worth taking, or should be skipped.

Scope, deliberately kept narrow:
  - The LLM decides ENTER vs SKIP only. It does not pick the direction
    (that's fixed by which level - PDH or PDL - is being tested) and it
    does not compute prices. Stop-loss / take-profit levels are still
    computed by the same deterministic, backtested math as the rule-based
    engine (see backtest_pdh_pdl.py) - only the entry judgment moved to
    the model.
  - No API key, no cost: talks to a local Ollama server (installed fresh
    each CI run - see .github/workflows/paper_trade.yml).

IMPORTANT: unlike the rule-based engine this replaced, this decision logic
has NOT been backtested - replaying it over historical data would mean
thousands of real Ollama calls, which is a separate, larger effort. Treat
this as a live experiment, not a validated strategy.
"""
import json
import subprocess
import sys
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a trading judgment assistant for a PDH/PDL (previous day high/low) reversal strategy on BTCUSDT.

The strategy's premise: price is testing yesterday's high (a potential SELL/short reversal) or yesterday's low (a potential BUY/long reversal). You are given the technical context for one such touch and must judge whether this specific setup has enough real confluence to be worth taking, or should be skipped.

You are NOT choosing the direction (it's fixed by which level is being tested) and you are NOT setting prices - only deciding enter vs skip, with your reasoning.

Weigh: does price action actually show rejection at the level (candlestick pattern), does a chart pattern support reversal, is volume confirming, does market structure already favor this direction, does momentum show a turn, and does the higher-timeframe trend not fight the trade. A setup with little confluence (most signals false/neutral) should usually be skipped. A setup stacking multiple independent confirmations is a stronger candidate.

Respond with ONLY a JSON object, no other text: {"action": "enter" or "skip", "reasoning": "one or two sentences explaining your judgment"}"""


def _ensure_server_running():
    """Best-effort: start `ollama serve` if it isn't already up. Safe to call every run."""
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return
    except requests.exceptions.RequestException:
        pass
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        try:
            requests.get("http://localhost:11434/api/tags", timeout=2)
            return
        except requests.exceptions.RequestException:
            time.sleep(1)


def build_context(side: str, level_price: float, close: float, atr: float, rsi: float,
                   vol_ratio: float, patterns: dict, structure: str, htf_trend_up: bool,
                   recent_candles: list) -> str:
    level_name = "PDH (previous day high)" if side == "sell" else "PDL (previous day low)"
    lines = [
        f"Setup: potential {'SHORT' if side == 'sell' else 'LONG'} reversal at {level_name} = {level_price:.2f}",
        f"Current close: {close:.2f}   ATR(14): {atr:.2f}   RSI(21): {rsi:.1f}",
        f"Volume vs its 40-bar average: {vol_ratio:.2f}x",
        f"Market structure: {structure}",
        f"4H trend filter: price is {'ABOVE' if htf_trend_up else 'BELOW'} its 4H EMA-50 ({'up' if htf_trend_up else 'down'}trend)",
        "Pattern flags (true = confirmed present):",
    ]
    for name, val in patterns.items():
        lines.append(f"  - {name}: {val}")
    lines.append("Last 10 15-minute candles (O/H/L/C):")
    for c in recent_candles[-10:]:
        lines.append(f"  {c['time']}  O:{c['o']:.2f} H:{c['h']:.2f} L:{c['l']:.2f} C:{c['c']:.2f}")
    return "\n".join(lines)


def decide_trade(context_text: str) -> dict:
    """Returns {"action": "enter"|"skip", "reasoning": str}. Never raises - any
    failure (server down, timeout, bad JSON) fails safe to "skip"."""
    _ensure_server_running()
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context_text},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        parsed = json.loads(content)
        action = str(parsed.get("action", "skip")).strip().lower()
        if action not in ("enter", "skip"):
            action = "skip"
        reasoning = str(parsed.get("reasoning", "")).strip()[:500]
        return {"action": action, "reasoning": reasoning}
    except Exception as e:
        print(f"LLM decision failed ({e!r}); defaulting to skip.", file=sys.stderr)
        return {"action": "skip", "reasoning": f"[fallback: LLM call failed - {e}]"}
