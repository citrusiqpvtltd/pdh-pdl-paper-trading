"""
Split a backtest window into N shards of roughly equal near-PDH/PDL touch
count, each sized to comfortably finish within GitHub Actions' hard 6-hour
per-job cap at qwen2.5:7b's observed ~100-150s/call. Outputs a JSON matrix
for .github/workflows/backtest_llm_sharded.yml to fan out over.

Shards are contiguous, non-overlapping, and chronological - touch N's shard
boundary is set right after touch N's own bar, so every touch is evaluated
by exactly one shard.

Usage:
    python plan_shards.py --days 365 --target-per-shard 120
Prints one line: matrix=<json>   (for $GITHUB_OUTPUT)
"""
import argparse
import json
import sys

import pandas as pd

import backtest_pdh_pdl as bt
import paper_trade as pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, required=True)
    ap.add_argument("--target-per-shard", type=int, default=120,
                     help="Touches per shard - keep well under ~200 to stay clear of the 6h job cap")
    ap.add_argument("--max-shards", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_parquet("data/btcusdt_15m.parquet")
    df = bt.compute_indicators(df, zone_pct=pt.ZONE_PCT, htf_rule=pt.HTF_RULE)
    near = (df["near_pdh"] | df["near_pdl"])

    cutoff = df["open_time"].max() - pd.Timedelta(days=args.days)
    window = df[(df["open_time"] >= cutoff) & near]
    touch_times = window["open_time"].tolist()
    n_touches = len(touch_times)

    n_shards = min(args.max_shards, max(1, -(-n_touches // args.target_per_shard)))  # ceil div
    print(f"{n_touches} touches over {args.days} days -> {n_shards} shards "
          f"(~{n_touches / n_shards:.0f} touches/shard)", file=sys.stderr)

    shards = []
    chunk_size = -(-n_touches // n_shards)  # ceil div, so last chunk may be smaller
    for idx, start in enumerate(range(0, n_touches, chunk_size)):
        chunk = touch_times[start:start + chunk_size]
        if not chunk:
            continue
        shard_start = chunk[0]
        shard_end = chunk[-1] + pd.Timedelta(minutes=15)  # exclusive end, includes the last touch's own bar
        shards.append({
            "index": idx,
            "start": shard_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end": shard_end.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_touches": len(chunk),
        })

    print(f"Planned {len(shards)} shards, total touches covered: {sum(s['n_touches'] for s in shards)}", file=sys.stderr)
    print(f"matrix={json.dumps(shards)}")


if __name__ == "__main__":
    main()
