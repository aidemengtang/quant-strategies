"""Monthly main-board stock selection by low volatility and positive momentum."""

from dataclasses import replace
from numbers import Integral

import numpy as np
import pandas as pd

from qs.lowvol_momentum_data import DEFAULT_CACHE, END, START, load_market
from qs.lowvol_momentum_execution import simulate

DEFAULT_LOOKBACK = 126
DEFAULT_TOP_N = 25
SKIP_DAYS = 20
VOL_DAYS = 60
MIN_ADV = 20_000_000
TOP_N_RANGE = (15, 35)
LOOKBACK_RANGE = (84, 168)


def validate_parameters(lookback, top_n):
    if not isinstance(lookback, Integral) or not LOOKBACK_RANGE[0] <= lookback <= LOOKBACK_RANGE[1]:
        raise ValueError(f"lookback must be in {LOOKBACK_RANGE}")
    if not isinstance(top_n, Integral) or not TOP_N_RANGE[0] <= top_n <= TOP_N_RANGE[1]:
        raise ValueError(f"top_n must be in {TOP_N_RANGE}")


def average_volume(market):
    return pd.DataFrame(market.amount).rolling(20).mean()


def monthly_ranks(market, lookback, volume, start):
    """Return ranked columns and the eligible positive-momentum pool per signal day."""
    days = market.dates.get_indexer(market.month_ends[market.month_ends >= pd.Timestamp(start)])
    ranks = {}
    for day in days:
        finish = day - SKIP_DAYS + 1
        if finish < lookback or day + 1 < VOL_DAYS:
            continue
        # Complete windows are required; prices are never filled across suspensions.
        momentum = np.prod(1 + market.returns[finish - lookback:finish], axis=0) - 1
        vol = np.std(market.returns[day + 1 - VOL_DAYS:day + 1], axis=0, ddof=1)
        liquidity = volume.iloc[day].to_numpy()
        valid = market.eligible[day] & np.isfinite(liquidity)
        if not valid.any():
            ranks[day] = np.array([], dtype=int)
            continue
        lower, upper = np.quantile(liquidity[valid], [0.30, 0.80])
        valid &= (
            (liquidity >= max(lower, MIN_ADV)) & (liquidity <= upper)
            & np.isfinite(momentum) & np.isfinite(vol)
        )
        columns = np.flatnonzero(valid)
        score = pd.Series(momentum[columns], index=columns).rank(pct=True)
        score += pd.Series(-vol[columns], index=columns).rank(pct=True)
        score = score.loc[momentum[columns] > 0]
        ranks[day] = score.sort_values(ascending=False, kind="stable").index.to_numpy(dtype=int)
    return ranks


def target_orders(ranks, top_n):
    return {day: dict.fromkeys(columns[:top_n], 1 / top_n) for day, columns in ranks.items()}


def backtest(market, lookback=DEFAULT_LOOKBACK, top_n=DEFAULT_TOP_N, start=START, **execution):
    validate_parameters(lookback, top_n)
    volume = average_volume(market)
    orders = target_orders(monthly_ranks(market, lookback, volume, start), top_n)
    return simulate(market, orders, volume.shift().to_numpy(), start=start, **execution)


def build_weights(symbols=None, lookback=DEFAULT_LOOKBACK, top_n=DEFAULT_TOP_N,
                  start=START, end=END, *, cache=DEFAULT_CACHE):
    """Return share-ledger marked returns and daily post-close actual weights."""
    validate_parameters(lookback, top_n)
    market = load_market(cache, end=end or END)
    if symbols is not None:
        # Keep index alignment for corporate-action and delisting records.
        allowed = market.codes.isin(symbols)
        market = replace(market, eligible=market.eligible & allowed)
    result = backtest(market, lookback, top_n, start, record=True)
    return result.returns, result.weights
