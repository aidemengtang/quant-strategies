"""Frozen ETF research protocol. Selection only receives the IS prefix."""

from dataclasses import dataclass, replace
from itertools import product

import pandas as pd

from qs.data import ETF_POOL, load_index_close, load_panel, price_adjustment
from qs.engine import perf_stats, run_backtest
from research.placebo import cagr_distribution
from strategies import candidate_a, candidate_b, etf_momentum, high_return, ma_timing
from strategies._rotation import daily_returns

HISTORY_START = "2015-01-01"
START = "2016-01-01"
IS_END = "2021-12-31"
OOS_START = "2022-01-01"
END = "2026-11-07"
PRIMARY_COST = 0.0005


@dataclass(frozen=True)
class Parameters:
    lookback: int
    top_n: int
    frequency: str = "monthly"


CANDIDATES = {"A": candidate_a, "B": candidate_b}
CENTERS = {
    name: Parameters(module.DEFAULT_LOOKBACK, module.DEFAULT_TOP_N)
    for name, module in CANDIDATES.items()
}


def prices(end=END, start=HISTORY_START):
    symbols = list(candidate_b.ETF_UNIVERSE)
    with price_adjustment("hfq"):
        panel = load_panel(symbols, start=start, end=end)
    missing = set(symbols).difference(panel.dropna(axis=1, how="all").columns)
    if missing:
        raise RuntimeError(f"Missing ETF data: {', '.join(sorted(missing))}")
    return panel.loc[:end, symbols]


def target_weights(name, params, panel, start=START):
    return CANDIDATES[name].target_weights(
        panel, params.lookback, params.top_n, start, params.frequency
    )


def backtest_inputs(returns, weights, start=START, end=END):
    returns = returns.loc[start:end]
    # A cash anchor makes first-session returns and costs visible to perf_stats.
    anchor = min(pd.Timestamp(start), returns.index[0] - pd.Timedelta(days=1))
    initial = pd.DataFrame(0.0, index=[anchor], columns=returns.columns)
    return pd.concat([initial, returns]), weights.loc[start:end]


def turnover(weights):
    return weights.diff().fillna(weights.iloc[0]).abs().sum(axis=1)


def metrics(nav, weights):
    stats = perf_stats(nav)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    stats["年化换手"] = turnover(weights).sum() / years
    return stats


def select(panel):
    """Compare predeclared centers by their neighborhood median IS Sharpe."""
    panel = panel.loc[:IS_END]
    returns = daily_returns(panel)
    rows = []
    for name, center in CENTERS.items():
        grid = product(
            (round(center.lookback * factor) for factor in (0.6, 1.0, 1.4)),
            range(center.top_n - 1, center.top_n + 2),
            ("monthly", "biweekly"),
        )
        for lookback, top_n, frequency in grid:
            params = replace(center, lookback=lookback, top_n=top_n, frequency=frequency)
            weights = target_weights(name, params, panel)
            test_returns, test_weights = backtest_inputs(returns, weights, end=IS_END)
            nav, _ = run_backtest(test_returns, test_weights, cost_rate=PRIMARY_COST)
            stats = metrics(nav, test_weights)
            rows.append({
                "candidate": name,
                "lookback": lookback,
                "top_n": top_n,
                "frequency": frequency,
                "center": params == center,
                "cagr": stats["年化收益"],
                "sharpe": stats["夏普"],
                "mdd": stats["最大回撤"],
            })
    grid = pd.DataFrame(rows)
    center_cagr = grid[grid["center"]].set_index("candidate")["cagr"]
    median_sharpe = grid.groupby("candidate")["sharpe"].median()
    winner = max(CANDIDATES, key=lambda name: (median_sharpe[name], center_cagr[name]))
    return winner, grid


def baseline_weights(returns):
    first_day = returns["510300"].loc[START:END].first_valid_index()
    benchmark = pd.DataFrame({"510300": [1.0]}, index=[first_day])
    with price_adjustment("hfq"):
        _, moving_average = ma_timing.build_weights(start=HISTORY_START, end=END)
        _, momentum = etf_momentum.build_weights(
            list(ETF_POOL), lookback=250, top_n=3, start=HISTORY_START, end=END
        )
    return {"买入持有 510300": benchmark, "双均线 MA20/60": moving_average,
            "现有 ETF 动量（250/前3）": momentum}


def period_returns(returns, frequency):
    return returns.groupby(returns.index.to_period(frequency)).agg(
        lambda period: (1 + period).prod() - 1
    )


def out_of_sample(nav):
    path = nav.loc[OOS_START:]
    anchor = pd.Series(nav.loc[:IS_END].iloc[-1], index=[pd.Timestamp(OOS_START)])
    return perf_stats(pd.concat([anchor, path]))


def market_states(nav, benchmark_nav):
    close = load_index_close("sh000300", HISTORY_START, END)
    market = period_returns(close.pct_change(fill_method=None).loc[START:END], "M")
    monthly = period_returns(nav.pct_change().dropna(), "M")
    if not monthly.index.isin(market.index).all():
        raise RuntimeError("CSI 300 index data do not cover all study months")
    return pd.DataFrame({
        "strategy": monthly,
        "benchmark": period_returns(benchmark_nav.pct_change().dropna(), "M"),
        "market": market,
    })


def evaluate():
    """Compute the full-period paths once at the primary cost after IS selection."""
    panel = prices()
    winner, grid = select(panel.loc[:IS_END])
    if winner != high_return.SELECTED_CANDIDATE:
        raise ValueError("Input data no longer reproduce the frozen IS selection")
    returns = daily_returns(panel)
    schedules = baseline_weights(returns)
    schedules.update({
        name: target_weights(name, params, panel) for name, params in CENTERS.items()
    })
    runs = {}
    for name, weights in schedules.items():
        test_returns, test_weights = backtest_inputs(returns[weights.columns], weights)
        if not (test_weights.ge(0).all().all() and test_weights.sum(axis=1).le(1 + 1e-9).all()):
            raise ValueError(f"Invalid weights for {name}")
        nav, positions = run_backtest(test_returns, test_weights, cost_rate=PRIMARY_COST)
        runs[name] = (nav, positions, metrics(nav, test_weights))
    oos = {name: out_of_sample(runs[name][0]) for name in CANDIDATES}
    return panel, returns, grid, schedules[winner], runs, oos


def analyze():
    panel, returns, grid, weights, runs, oos = evaluate()
    winner = high_return.SELECTED_CANDIDATE
    nav, positions, stats = runs[winner]
    test_returns, weights = backtest_inputs(returns, weights)
    costs = {PRIMARY_COST: stats}
    for cost in (0.001, 0.002):
        cost_nav, _ = run_backtest(test_returns, weights, cost_rate=cost)
        costs[cost] = metrics(cost_nav, weights)

    daily = nav.pct_change().dropna()
    yearly = period_returns(daily, "Y")
    monthly = period_returns(daily, "M")
    market_months = market_states(nav, runs["买入持有 510300"][0])
    annual_pnl = nav.groupby(nav.index.to_period("Y")).last().diff()
    annual_pnl.iloc[0] = nav.loc[str(yearly.index[0])].iloc[-1] - nav.iloc[0]
    total_pnl = nav.iloc[-1] - nav.iloc[0]
    largest_contribution = annual_pnl.max() / total_pnl if total_pnl > 0 else float("inf")
    best_months = monthly.nlargest(3)
    remaining = monthly.drop(best_months.index)
    trimmed_cagr = (1 + remaining).prod() ** (12 / len(remaining)) - 1

    lookback = CENTERS[winner].lookback
    available = panel.notna() & panel.shift(lookback).notna()
    if winner == "A":
        available &= panel.rolling(candidate_a.TREND_WINDOW).count().eq(candidate_a.TREND_WINDOW)
    random_cagr = cagr_distribution(test_returns, weights, available, PRIMARY_COST)
    return {
        "panel": panel, "grid": grid, "weights": weights, "runs": runs, "oos": oos,
        "costs": costs, "yearly": yearly, "market_months": market_months,
        "largest_contribution": largest_contribution, "largest_year": annual_pnl.idxmax(),
        "best_months": best_months, "trimmed_cagr": trimmed_cagr, "random_cagr": random_cagr,
    }
