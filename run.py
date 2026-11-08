#!/usr/bin/env python3
"""一键回测入口。

用法：
    python run.py                       # 跑动量轮动（默认）
    python run.py --strategy ma         # 跑双均线择时
    python run.py --strategy all --start 2015-01-01
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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


def run_research(args) -> None:
    from research.study import CENTERS, END, PRIMARY_COST, START, backtest_inputs, metrics
    from strategies import candidate_a, candidate_b, high_return

    module, name = {
        "high": (high_return, high_return.SELECTED_CANDIDATE),
        "candid_a": (candidate_a, "A"),
        "candid_b": (candidate_b, "B"),
    }[args.strategy]
    center = CENTERS[name]
    start, end = args.start or START, args.end or END
    lookback = args.lookback if args.lookback is not None else center.lookback
    top_n = args.top_n if args.top_n is not None else center.top_n
    frequency = args.frequency or center.frequency
    if args.strategy == "high" and (start, end, lookback, top_n, frequency, args.cost) == (
        START, END, center.lookback, center.top_n, center.frequency, PRIMARY_COST,
    ):
        from research.report import run_study

        stats = run_study()
        print("已生成 reports/high_return.md 与 results/high_return_*.csv")
    else:
        returns, weights = module.build_weights(
            lookback=lookback, top_n=top_n, start=start, end=end, frequency=frequency,
        )
        returns, weights = backtest_inputs(returns, weights, start=start, end=end)
        nav, _ = run_backtest(returns, weights, cost_rate=args.cost)
        stats = metrics(nav, weights)
    print(f"\n=== 候选 {name}（{lookback} 日 / 前 {top_n} / {frequency}）===")
    print(format_stats(stats))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategy",
        choices=["ma", "momentum", "all", "bench", "high", "candid_a", "candid_b", "lowvol_momentum"],
        default="momentum",
    )
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost", type=float, default=0.0005, help="单边手续费率，默认万5")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=60)
    ap.add_argument("--lookback", type=int, help="动量窗口，现有动量默认 250 日")
    ap.add_argument("--top_n", type=int)
    ap.add_argument("--frequency", choices=["monthly", "biweekly"], help="候选策略调仓频率")
    ap.add_argument("--cache", type=Path, help="低波动动量数据缓存")
    args = ap.parse_args()

    if args.strategy == "lowvol_momentum":
        from qs.lowvol_momentum_data import CoverageError
        from research.lowvol_momentum_report import run_study
        from research.lowvol_momentum_scan import DEFAULT_CACHE

        if any(value is not None for value in (args.start, args.end, args.lookback, args.top_n,
                                                args.frequency)) or args.cost != 0.0005:
            ap.error("lowvol_momentum uses a fixed research protocol")
        os.environ.update(no_proxy="*", NO_PROXY="*")
        try:
            print(format_stats(run_study(args.cache or DEFAULT_CACHE)))
        except (CoverageError, FileNotFoundError) as exc:
            ap.exit(1, f"低波动动量数据未通过勘测：{exc}\n"
                    "请先运行 python -m research.lowvol_momentum_scan 和 "
                    "python -m research.lowvol_momentum_fetch --resume。\n")
        return

    if args.strategy in ("high", "candid_a", "candid_b"):
        run_research(args)
        return

    if args.start is None:
        args.start = "2015-01-01"
    if args.lookback is None:
        args.lookback = 250
    if args.top_n is None:
        args.top_n = 3
    if args.strategy in ("bench", "all"):
        run_benchmark(args)
    if args.strategy in ("ma", "all"):
        run_ma(args)
    if args.strategy in ("momentum", "all"):
        run_momentum(args)


if __name__ == "__main__":
    main()
