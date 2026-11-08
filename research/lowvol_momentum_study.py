"""Run the fixed IS/OOS protocol and compute V1–V8 for the main-board strategy."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from qs.data import ETF_POOL, load_etf, load_index_close, price_adjustment
from qs.engine import perf_stats, run_backtest
from qs.lowvol_momentum_data import CoverageError, load_market
from qs.lowvol_momentum_execution import MAX_WEIGHT, simulate
from research.lowvol_momentum_scan import DEFAULT_CACHE, END, IS_END, OOS_START, START, WARMUP_START
from strategies import etf_momentum, lowvol_momentum as strategy

WINDOWS = (84, 105, 126, 147, 168)
HOLDINGS = (15, 20, 25, 30, 35)
PRIMARY_FEE = 0.0005
SLIPPAGE = 0.001
PLACEBOS = 200


def measure(result):
    stats = perf_stats(result.nav)
    years = (result.nav.index[-1] - result.nav.index[0]).days / 365.25
    stats["年化换手"] = result.turnover.sum() / years
    return stats


def sample_anchor(nav, start):
    boundary = pd.Timestamp(start)
    anchor = nav.loc[nav.index < boundary].iloc[[-1]]
    return pd.concat([anchor, nav.loc[boundary:]])


def coverage_gate(market):
    missing = market.audit.loc[
        market.audit["missing_status_file"].eq(1) | market.audit["missing_raw_file"].eq(1)
    ]
    if len(missing):
        codes = ", ".join(missing["code"].head(12))
        raise CoverageError(f"{len(missing)} 个代码缺失历史日线或状态文件，含 {codes}。")


def is_grid(market, volume):
    results = []
    lagged = volume.shift().to_numpy()
    for lookback in WINDOWS:
        ranks = strategy.monthly_ranks(market, lookback, volume, START)
        for top_n in HOLDINGS:
            orders = strategy.target_orders(ranks, top_n)
            run = simulate(market, orders, lagged, start=START)
            stats = measure(run)
            results.append((lookback, top_n, stats["年化收益"], stats["夏普"], stats["最大回撤"],
                            run.max_weight))
    return pd.DataFrame(results, columns=["lookback", "top_n", "cagr", "sharpe", "mdd", "max_weight"])


def plateau(grid):
    default = (strategy.DEFAULT_LOOKBACK, strategy.DEFAULT_TOP_N)
    returns = grid.pivot(index="lookback", columns="top_n", values="cagr")
    limits = grid.pivot(index="lookback", columns="top_n", values="max_weight")
    passes = float((returns > 0.12).to_numpy().mean())
    if passes < 0.70:
        return default, passes, False
    candidates = []
    for i, lookback in enumerate(WINDOWS[1:-1], 1):
        for j, top_n in enumerate(HOLDINGS[1:-1], 1):
            neighborhood = returns.iloc[i - 1:i + 2, j - 1:j + 2].to_numpy()
            center = returns.iloc[i, j]
            if (center > 0.12 and neighborhood.min() >= center / 2
                    and limits.iloc[i - 1:i + 2, j - 1:j + 2].to_numpy().max() <= MAX_WEIGHT + 1e-9):
                distance = abs(i - 2) + abs(j - 2)
                candidates.append((np.median(neighborhood), -distance, -lookback, -top_n))
    if not candidates:
        return default, passes, False
    _, _, lookback, top_n = max(candidates)
    return (-lookback, -top_n), passes, True


def placebo_orders(ranks, real_orders, rng):
    """Random draws retain an incumbent slot when the real order retains it."""
    mapping = {}
    randomized = {}
    for day, real in real_orders.items():
        allowed = set(ranks[day])
        mapping = {key: code for key, code in mapping.items() if key in real and code in allowed}
        available = np.array(sorted(allowed.difference(mapping.values())), dtype=int)
        missing = [key for key in real if key not in mapping]
        for key, code in zip(missing, rng.choice(available, len(missing), replace=False)):
            mapping[key] = code
        randomized[day] = {mapping[key]: weight for key, weight in real.items()}
    return randomized


def same_turnover(real, trial):
    return float(real.sub(trial).abs().max()) <= 0.0001


def random_controls(market, ranks, orders, lagged, reference):
    rng = np.random.default_rng(42)
    matched, all_draws = [], []
    for number in range(1, PLACEBOS + 1):
        randomized = placebo_orders(ranks, orders, rng)
        trial = simulate(market, randomized, lagged, start=START)
        cagr = measure(trial)["年化收益"]
        all_draws.append(cagr)
        if same_turnover(reference.turnover, trial.turnover):
            matched.append(cagr)
        if number % 25 == 0:
            print(f"V6 {number}/{PLACEBOS} draws; turnover matched: {len(matched)}", flush=True)
    return np.array(matched), np.array(all_draws)


def style_factors(market, volume, lookback):
    """Lag monthly top-minus-bottom portfolios; the factors are diagnostic only."""
    result = np.zeros((len(market.dates), 3))
    months = market.dates.get_indexer(market.month_ends)
    next_months = np.r_[months[1:] + 1, len(market.dates)]
    for day, end in zip(months, next_months):
        finish = day - strategy.SKIP_DAYS + 1
        if finish < lookback or day + 1 < strategy.VOL_DAYS:
            continue
        momentum = np.prod(1 + market.returns[finish - lookback:finish], axis=0) - 1
        volatility = np.std(market.returns[day + 1 - strategy.VOL_DAYS:day + 1], axis=0, ddof=1)
        predictors = (-volatility, -volume.iloc[day].to_numpy(), momentum)
        for factor, scores in enumerate(predictors):
            allowed = np.flatnonzero(market.eligible[day] & np.isfinite(scores))
            count = len(allowed) // 5
            if not count:
                continue
            order = allowed[np.argsort(scores[allowed], kind="stable")]
            first = np.nan_to_num(market.returns[day + 1:end, order[-count:]], nan=0)
            second = np.nan_to_num(market.returns[day + 1:end, order[:count]], nan=0)
            result[day + 1:end, factor] = first.mean(axis=1) - second.mean(axis=1)
    return pd.DataFrame(result, index=market.dates, columns=["低波", "流动性", "动量"])


def attribution(market, volume, run, lookback, index_close):
    market_return = index_close.pct_change(fill_method=None)
    factors = style_factors(market, volume, lookback).assign(市场=market_return).reindex(run.nav.index)
    if factors.isna().any().any():
        raise CoverageError("沪深 300 指数日线缺失，无法完成 V8 归因")
    net = run.nav.pct_change().fillna(0)
    gross = net + run.friction
    is_factors = factors.loc[:IS_END]
    design = np.column_stack([np.ones(len(is_factors)), is_factors.to_numpy()])
    coefficients = np.linalg.lstsq(design, gross.loc[:IS_END].to_numpy(), rcond=None)[0][1:]
    contributions = factors * coefficients
    scale = np.ones(len(net))
    np.divide(np.log1p(net), net, out=scale, where=net.ne(0))
    market_piece = contributions["市场"]
    style_piece = contributions.drop(columns="市场").sum(axis=1)
    parts = pd.DataFrame({
        "市场 beta": market_piece,
        "风格 beta": style_piece,
        "选股残差": gross - market_piece - style_piece,
        "交易摩擦": -run.friction,
    }).mul(scale, axis=0).sum()
    # The share is undefined when the strategy's total positive gain is absent.
    return parts, float(parts["风格 beta"] / parts.sum()) if parts.sum() > 0 else np.nan


def baselines():
    with price_adjustment("hfq"):
        prices = load_etf("510300", start=WARMUP_START, end=END)["close"]
        momentum_returns, momentum_weights = etf_momentum.build_weights(
            list(ETF_POOL), 250, 3, WARMUP_START, END
        )
    returns = prices.pct_change(fill_method=None).loc[START:].to_frame("510300")
    target = pd.DataFrame({"510300": [1.0]}, index=returns.index[:1])
    anchor = pd.Timestamp(START)
    returns.loc[anchor] = 0
    momentum_returns = momentum_returns.loc[START:].copy()
    momentum_returns.loc[anchor] = 0
    benchmark = run_backtest(returns, target, cost_rate=PRIMARY_FEE + SLIPPAGE)[0]
    momentum = run_backtest(
        momentum_returns, momentum_weights.loc[START:], cost_rate=PRIMARY_FEE + SLIPPAGE
    )[0]
    return {"沪深 300 ETF 买入持有": perf_stats(benchmark), "既有 ETF 动量": perf_stats(momentum)}


def drawdown_episodes(nav, market_index):
    """Describe historically observed 10% drawdowns and their market context."""
    episodes = []
    peak_date = trough_date = nav.index[0]
    peak = trough = nav.iloc[0]
    for date, value in nav.iloc[1:].items():
        if value >= peak:
            if trough < peak * 0.90:
                episodes.append((peak_date, trough_date, date, trough / peak - 1))
            peak_date = trough_date = date
            peak = trough = value
        elif value < trough:
            trough_date, trough = date, value
    if trough < peak * 0.90:
        episodes.append((peak_date, trough_date, pd.NaT, trough / peak - 1))
    return [
        (start, bottom, recovery, decline, market_index.asof(bottom) / market_index.asof(start) - 1)
        for start, bottom, recovery, decline in episodes
    ]


def analyze(root=DEFAULT_CACHE):
    root = Path(root)
    print("读取 IS 行情及状态数据", flush=True)
    is_market = load_market(root, end=IS_END)
    coverage_gate(is_market)
    is_volume = strategy.average_volume(is_market)
    grid = is_grid(is_market, is_volume)
    (lookback, top_n), neighborhood_rate, stable = plateau(grid)
    is_main = strategy.backtest(is_market, lookback, top_n)
    listing = pd.read_parquet(root / "listings.parquet")
    exits = set(listing.loc[listing["exit_registry"], "code"])
    survivors = replace(is_market, eligible=is_market.eligible & ~is_market.codes.isin(exits))
    is_survivors = strategy.backtest(survivors, lookback, top_n)
    del survivors, is_market, is_volume
    lock = {"lookback": lookback, "top_n": top_n, "is_end": IS_END, "stable": stable}
    (root / "is_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(f"IS 已锁定 {lock}，读取全样本", flush=True)

    market = load_market(root)
    market.audit.to_csv(root / "coverage_by_code.csv", index=False)
    coverage_gate(market)
    index_close = load_index_close("sh000300", WARMUP_START, END)
    if index_close.reindex(market.dates[market.dates >= START]).isna().any():
        raise CoverageError("沪深 300 指数日线缺失，无法完成 V8 归因")
    benchmark = baselines()
    volume = strategy.average_volume(market)
    lagged = volume.shift().to_numpy()
    ranks = strategy.monthly_ranks(market, lookback, volume, START)
    orders = strategy.target_orders(ranks, top_n)
    main = simulate(market, orders, lagged, start=START, record=True)
    engine_nav, _ = run_backtest(main.returns, main.weights, cost_rate=PRIMARY_FEE + SLIPPAGE)
    oos = perf_stats(sample_anchor(main.nav, OOS_START))
    engine_oos = perf_stats(sample_anchor(engine_nav, OOS_START))
    costs = {PRIMARY_FEE: measure(main)}
    for commission in (0.001, 0.002):
        costs[commission] = measure(simulate(market, orders, lagged, start=START, commission=commission))
    stress_slip = measure(simulate(market, orders, lagged, start=START, slip=0.002))
    stress_recovery = measure(simulate(
        market, orders, lagged, start=START, zero_recovery=True, write_off_unknown=True
    ))
    returns = main.nav.pct_change().dropna()
    periods = returns.index.to_period("Y")
    yearly = (1 + returns).groupby(periods).prod() - 1
    year_end = main.nav.groupby(main.nav.index.to_period("Y")).last()
    yearly_pnl = year_end.diff()
    yearly_pnl.iloc[0] = year_end.iloc[0] - main.nav.iloc[0]
    total = main.nav.iloc[-1] - main.nav.iloc[0]
    largest_year = yearly_pnl.idxmax()
    contribution = yearly_pnl.max() / total if total > 0 else np.nan
    best_days = returns.nlargest(5)
    rest = returns.drop(best_days.index)
    years = (main.nav.index[-1] - main.nav.index[0]).days / 365.25 * len(rest) / len(returns)
    trimmed = (1 + rest).prod() ** (1 / years) - 1
    peak = main.nav.eq(main.nav.cummax())
    last_peak = pd.Series(main.nav.index.where(peak), index=main.nav.index).dropna()
    gaps = last_peak.diff().dt.days.dropna()
    last_gap = (main.nav.index[-1] - last_peak.iloc[-1]).days
    underwater = max(gaps.max() if len(gaps) else 0, last_gap)
    matched, controls = random_controls(market, ranks, orders, lagged, main)
    attribution_parts, style_share = attribution(market, volume, main, lookback, index_close)
    capacity_by_day = main.trades.groupby("date")["capacity"].min()
    return {
        "grid": grid, "lock": lock, "stable": stable, "neighborhood_rate": neighborhood_rate,
        "is": measure(is_main), "is_survivors": measure(is_survivors), "main": main, "oos": oos,
        "engine_oos": engine_oos, "engine_stats": perf_stats(engine_nav), "costs": costs,
        "stress_slip": stress_slip, "stress_recovery": stress_recovery, "yearly": yearly,
        "largest_year": largest_year, "contribution": contribution, "best_days": best_days,
        "trimmed": trimmed, "underwater": underwater, "matched": matched, "controls": controls,
        "attribution": attribution_parts, "style_share": style_share, "capacity": capacity_by_day,
        "audit": market.audit, "baselines": benchmark,
        "drawdown_episodes": drawdown_episodes(main.nav, index_close),
        "scan": json.loads((root / "scan.json").read_text()),
    }
