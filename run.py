#!/usr/bin/env python3
"""一键回测入口。

用法：
    python run.py                       # 跑动量轮动（默认）
    python run.py --strategy ma         # 跑双均线择时
    python run.py --strategy all --start 2015-01-01
"""

from __future__ import annotations

import argparse

from qs.data import ETF_POOL
from qs.engine import format_stats, perf_stats, run_backtest


def run_ma(args) -> dict:
    from strategies.ma_timing import build_weights

    rets, w = build_weights(symbol="510300", fast=args.fast, slow=args.slow,
                            start=args.start, end=args.end)
    nav, _ = run_backtest(rets, w, cost_rate=args.cost)
    stats = perf_stats(nav, w)
    print("\n=== 双均线择时（510300 沪深300ETF，MA%d/MA%d）===" % (args.fast, args.slow))
    print(format_stats(stats))
    return stats


def run_momentum(args) -> dict:
    from strategies.etf_momentum import build_weights

    rets, w = build_weights(list(ETF_POOL), lookback=args.lookback, top_n=args.top_n,
                            start=args.start, end=args.end)
    nav, pos = run_backtest(rets, w, cost_rate=args.cost)
    stats = perf_stats(nav, w)
    print("\n=== ETF 动量轮动（%d 只池，%d 日动量，取前 %d，月度调仓）===" %
          (len(ETF_POOL), args.lookback, args.top_n))
    print(format_stats(stats))
    print("最新持仓：", {ETF_POOL.get(k, k): f"{v:.0%}" for k, v in pos.iloc[-1].items() if v > 0})
    return stats


def run_benchmark(args) -> dict:
    """基准：始终满仓沪深300ETF（买入持有）。"""
    from qs.data import load_etf

    close = load_etf("510300", start=args.start, end=args.end)["close"]
    rets = close.pct_change(fill_method=None).dropna().to_frame("510300")
    w = rets.copy()
    w[:] = 1.0
    nav, _ = run_backtest(rets, w, cost_rate=0.0)
    stats = perf_stats(nav)
    print("\n=== 基准：买入持有 510300 ===")
    print(format_stats(stats))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["ma", "momentum", "all", "bench"], default="momentum")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost", type=float, default=0.0005, help="单边手续费率，默认万5")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=60)
    ap.add_argument("--lookback", type=int, default=250, help="动量窗口，默认 250 日≈12 个月")
    ap.add_argument("--top_n", type=int, default=3)
    args = ap.parse_args()

    if args.strategy in ("bench", "all"):
        run_benchmark(args)
    if args.strategy in ("ma", "all"):
        run_ma(args)
    if args.strategy in ("momentum", "all"):
        run_momentum(args)


if __name__ == "__main__":
    main()
