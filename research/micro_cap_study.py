"""Two-stage IS research and once-frozen microcap validation."""

from dataclasses import asdict, replace
from itertools import product

import numpy as np
import pandas as pd

from qs.micro_cap_data import END, IS_END, START
from qs.micro_cap_execution import CAPITAL, fee_rates, fees, simulate
from strategies.micro_cap import Parameters, Selection, eligible, plan, rank, schedule, trailing, universe

WINDOWS = (5, 10, 15, 20, 25)
NS = (15, 20, 25, 30, 35)
FLOORS = tuple(range(30_000_000, 60_000_001, 5_000_000))
WIDTHS = (20, 50, 100)
OOS = "2021-01-01"
FEE_GRID = tuple(product((.0005, .001, .002), (.001, .002)))
EPISODES = (
    ("股灾与熔断", "2015-06-01", "2016-01-31"),
    ("小盘估值下修", "2017-01-01", "2017-12-31"),
    ("抱团切换", "2021-02-01", "2021-03-31"),
    ("微盘踩踏", "2024-01-01", "2024-02-29"),
    ("新国九条后", "2024-04-01", END),
)
PERIODS = (("2015–2020", START, IS_END), ("2021–2023", OOS, "2023-12-31"), ("2024–2026", "2024-01-01", END))


def growth_stats(factors):
    nav = np.cumprod(factors)
    if not len(nav):
        return np.nan, np.nan
    return nav[-1] ** (252 / len(nav)) - 1, np.min(nav / np.maximum.accumulate(np.r_[1., nav])[1:] - 1)


def factors(nav):
    return nav / nav.shift(1, fill_value=1.)


def coarse(market, days, selected, n, stamp, transfer):
    """Vectorize every holding interval and all rebalance deltas on the adjusted panel."""
    days = np.asarray(days)
    fills = days + 1
    valid = fills < len(market.dates)
    fills = fills[valid]
    # Use a cash column for unused slots and mask it out of both transaction legs.
    sentinel = market.adjusted.shape[1]
    chosen = np.full((len(fills), n), sentinel, dtype=np.int32)
    for i, row in enumerate(selected[:len(fills)]):
        chosen[i, :min(n, len(row))] = row[:n]
    next_fill = np.r_[fills[1:], len(market.dates) - 1]
    data = market.adjusted
    start = market.dates.searchsorted(START)
    if not len(fills):
        return 0., 0., 0.
    active = chosen != sentinel
    safe = np.minimum(chosen, sentinel - 1)
    baseline = data[fills[:, None], safe]
    terminal = np.where(active, data[next_fill[:, None], safe] / baseline, 0.)
    segment_value = 1 + (terminal - active).sum(axis=1) / n
    prior_weights = terminal[:-1] / (n * segment_value[:-1, None])
    overlap = (chosen[:-1, :, None] == chosen[1:, None, :]) & active[:-1, :, None] & active[1:, None, :]
    sold = np.zeros_like(terminal)
    bought = active.astype(float) / n
    sold[1:] = np.maximum(prior_weights - overlap.any(axis=2) / n, 0)
    bought[1:] = np.maximum(bought[1:] - (prior_weights[:, :, None] * overlap).sum(axis=1), 0)
    cost = (fees(sold * CAPITAL, .0005, (.001 + transfer[fills] + stamp[fills])[:, None])
            + fees(bought * CAPITAL, .0005, (.001 + transfer[fills])[:, None])).sum(axis=1) / CAPITAL
    sessions = np.arange(start, len(market.dates))
    interval = fills.searchsorted(sessions, side="right") - 1
    live = interval >= 0
    ids = interval[live]
    held = active[ids]
    ratios = data[sessions[live, None], safe[ids]] / baseline[ids]
    relative = 1 + (np.where(held, ratios - 1, 0.)).sum(axis=1) / n
    end_values = segment_value * (1 - cost)
    prefixes = np.r_[1., np.cumprod(end_values[:-1])]
    nav = np.ones(len(sessions))
    nav[live] = relative * (1 - cost[ids]) * prefixes[ids]
    ret = nav[-1] ** (252 / len(nav)) - 1
    dd = np.min(nav / np.maximum.accumulate(nav) - 1)
    turnover = (sold.sum() + bought.sum()) * 252 / len(nav)
    return ret, dd, turnover


def screen(market, order):
    stamp, transfer = fee_rates(market.dates)
    sessions = {cadence: list(schedule(market, cadence)) for cadence in (10, 15, 20)}
    all_days = sorted(set().union(*sessions.values()))
    pools = {(floor, day): eligible(market, order[day], day, floor)
             for floor in FLOORS for day in all_days}
    rows = []
    for window, skip in product(WINDOWS, range(3)):
        history = {day: trailing(market, day, window, skip) for day in all_days}
        for floor, width in product(FLOORS, WIDTHS):
            ranks = {day: rank(pools[floor, day], history[day], width).names for day in all_days}
            for cadence, n in product(sessions, NS):
                days = sessions[cadence]
                ret, dd, turnover = coarse(market, days, [ranks[day] for day in days], n, stamp, transfer)
                rows.append((window, skip, n, cadence, floor, width, ret, dd, turnover))
        print(f"IS 粗筛进度 {len(rows)}/4725", flush=True)
    return pd.DataFrame(rows, columns=[*asdict(Parameters(0, 0, 0, 0, 0, 0)),
                                       "cagr", "drawdown", "turnover"])


def execute(market, params, selections, **kwargs):
    return simulate(market, selections, params.n, adv_floor=params.floor, **kwargs)


def metrics(ledger):
    daily = factors(ledger.nav)
    annual, dd = growth_stats(daily.to_numpy())
    r = daily - 1
    invested = (1 - ledger.cash) / ledger.count.replace(0, np.nan)
    return {
        "cagr": annual, "dd": dd, "sharpe": r.mean() / r.std() * np.sqrt(252),
        "calmar": annual / -dd if dd < 0 else np.nan,
        "turnover": ledger.turnover.mean() * 252, "average_weight": invested.mean(),
        "order_p10": ledger.order_capacity.quantile(.1), "hold_p10": ledger.holding_capacity.quantile(.1),
        "order_median": ledger.order_capacity.median(), "hold_median": ledger.holding_capacity.median(),
    }


def neighborhood(grid, params):
    fixed = grid.loc[(grid.skip == params.skip) & (grid.cadence == params.cadence)
                     & (grid.floor == params.floor) & (grid.width == params.width)]
    local = fixed.loc[fixed.window.sub(params.window).abs().le(5) & fixed.n.sub(params.n).abs().le(5)
                      & ~((fixed.window == params.window) & (fixed.n == params.n)), "cagr"]
    return fixed.cagr.gt(.12).mean(), local.min(), local.median()


def calibrate(market, order, grid):
    keys = ["window", "skip", "n", "cadence", "floor"]
    top = grid.sort_values(["cagr", *keys, "width"], ascending=[False, *([True] * 6)])
    top = top.drop_duplicates(keys).head(20)
    candidates = []
    for i, row in enumerate(top.itertuples(index=False), 1):
        params = Parameters(*(getattr(row, key) for key in [*keys, "width"]))
        selections = plan(market, params, order)
        primary = execute(market, params, selections)
        stats = metrics(primary)
        stressed = metrics(execute(market, params, selections, rate=.002, slip=.002))["cagr"]
        share, local_min, local_mid = neighborhood(grid, params)
        qualified = (stats["cagr"] > .12 and share >= .70 and local_min >= stats["cagr"] / 2
                     and stats["dd"] >= -.40 and stats["turnover"] <= 35
                     and stats["average_weight"] <= .045 and stats["order_p10"] >= 30_000_000
                     and stressed >= .12)
        candidates.append(dict(**asdict(params), coarse=row.cagr, **stats, stressed=stressed,
                               grid_pass_share=share, local_min=local_min, local_median=local_mid,
                               qualified=qualified))
        print(f"IS 逐笔精算 {i}/20", flush=True)
    candidates = pd.DataFrame(candidates)
    qualified = candidates.loc[candidates.qualified].sort_values(
        ["local_median", "turnover", "floor", "window", "skip", "n", "cadence", "width"],
        ascending=[False, True, True, True, True, True, True, True])
    return candidates, params_from(qualified.iloc[0]) if not qualified.empty else None


def params_from(row):
    return Parameters(*(int(row[key]) for key in Parameters.__dataclass_fields__))


def matched(selections, n):
    output = {}
    for day, chosen in selections.items():
        groups, count = np.unique(chosen.bucket_ids, return_counts=True)
        names = np.concatenate([chosen.buckets[g] for g in groups]) if len(groups) else np.array([], dtype=int)
        weights = np.concatenate([np.full(len(chosen.buckets[g]), size / n / len(chosen.buckets[g]))
                                  for g, size in zip(groups, count)]) if len(groups) else np.array([])
        output[day] = Selection(names, chosen.buckets, chosen.bucket_ids, weights)
    return output


def placebo(market, params, selections, runs=200):
    rng = np.random.default_rng(42)
    values = []
    for i in range(runs):
        control = {}
        for day, chosen in selections.items():
            groups, counts = np.unique(chosen.bucket_ids, return_counts=True)
            draws = [rng.choice(chosen.buckets[g], size, replace=False) for g, size in zip(groups, counts)]
            control[day] = replace(chosen, names=np.concatenate(draws) if draws else chosen.names)
        values.append(metrics(execute(market, params, control))["cagr"])
        if (i + 1) % 25 == 0:
            print(f"随机对照 {i+1}/{runs}", flush=True)
    return np.array(values)


def _cross_mean(values, mask):
    counts = mask.sum(axis=1)
    return np.divide(np.where(mask, values, 0.).sum(axis=1), counts,
                     out=np.zeros(len(values)), where=counts > 0)


def attribution(market, ledger):
    """Fit exposures in IS, apply frozen exposures to subsequent daily log returns."""
    start = market.dates.searchsorted(START)
    today = market.returns[start:]
    # Lagged size and 20-session reversal separate known style exposures from today's return.
    cap = market.cap[start - 1:len(market.dates) - 1]
    adj = market.adjusted
    trend = adj[start - 1:len(market.dates) - 1] / adj[start - 21:len(market.dates) - 21] - 1
    valid = np.isfinite(today) & np.isfinite(cap)
    with np.errstate(all="ignore"):
        small, large = np.nanquantile(cap, [.2, .8], axis=1)
        weak, strong = np.nanquantile(np.where(valid, trend, np.nan), [.2, .8], axis=1)
    market_ret = _cross_mean(today, valid)
    size = _cross_mean(today, valid & (cap <= small[:, None])) - _cross_mean(today, valid & (cap >= large[:, None]))
    style = _cross_mean(today, valid & (trend >= strong[:, None])) - _cross_mean(today, valid & (trend <= weak[:, None]))
    x = np.column_stack([market_ret, size, style])
    gross = np.log(ledger.gross.to_numpy())
    net = np.log(factors(ledger.nav).to_numpy())
    is_mask = ledger.nav.index <= IS_END
    betas = np.linalg.lstsq(x[is_mask], gross[is_mask], rcond=None)[0]
    components = (x * betas).sum(axis=0)
    return np.r_[components, gross.sum() - components.sum(), (net - gross).sum()]


def perturb(market, start):
    row = market.dates.searchsorted(start)
    returns = market.returns.copy()
    returns[row:] = np.where(np.isfinite(returns[row:]), returns[row:] * .5 + .01, np.nan)
    from qs.micro_cap_data import adjusted_prices
    close = market.close.copy()
    close[row:] *= 1.13
    return replace(market, returns=returns, adjusted=adjusted_prices(returns), close=close)


def validate(market, params, order, survivor_mask):
    selections = plan(market, params, order)
    print("主账本、成本与同市值基准", flush=True)
    primary = execute(market, params, selections, record=True)
    costs = {(rate, slip): metrics(primary if (rate, slip) == (.0005, .001) else
                                   execute(market, params, selections, rate=rate, slip=slip))["cagr"]
             for rate, slip in FEE_GRID}
    bench = simulate(market, matched(selections, params.n), 400, adv_floor=params.floor)
    surviving = execute(market, params, plan(market, params, universe(market, survivor_mask)))
    zero = execute(market, params, selections, recovery=0., record=True)
    random_cagrs = placebo(market, params, selections)
    limit = market.dates[-1] - pd.DateOffset(years=1)
    cut = market.through(limit)
    prefix = execute(cut, params, plan(cut, params))
    changed = perturb(market, limit + pd.Timedelta(days=1))
    changed_run = execute(changed, params, plan(changed, params))
    clean_prefix = primary.nav.loc[:limit]
    return {
        "primary": primary, "costs": costs, "benchmark": bench, "surviving": surviving,
        "zero": zero, "placebo": random_cagrs, "selections": selections, "attribution": attribution(market, primary),
        "truncate_ok": prefix.nav.equals(clean_prefix), "perturb_ok": changed_run.nav.loc[:limit].equals(clean_prefix),
        "limit": limit,
    }
