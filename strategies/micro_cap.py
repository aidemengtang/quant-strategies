"""Microcap reversal plans using only the signal close and earlier observations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qs.micro_cap_data import START


@dataclass(frozen=True, order=True)
class Parameters:
    window: int
    skip: int
    n: int
    cadence: int
    floor: int
    width: int


@dataclass(frozen=True)
class Selection:
    names: np.ndarray
    buckets: tuple
    bucket_ids: np.ndarray
    weights: np.ndarray | None = None


def universe(market, allowed=None):
    cap = market.cap if allowed is None else np.where(allowed, market.cap, np.nan)
    return np.argsort(cap, axis=1, kind="stable")[:, :400]


def schedule(market, cadence):
    return range(market.dates.searchsorted(START), len(market.dates), cadence)


def eligible(market, pool, day, floor):
    return pool[np.isfinite(market.cap[day, pool]) & (market.adv[day, pool] >= floor)
                & market.known[day, pool] & ~market.risk[day, pool]]


def trailing(market, day, window, skip):
    end = day - skip
    begin = end - window
    if begin < 0:
        return np.full(len(market.codes), np.nan)
    observed = np.isfinite(market.returns[begin + 1:end + 1]).all(axis=0)
    growth = market.adjusted[end] / market.adjusted[begin] - 1
    return np.where(observed, growth, np.nan)


def rank(pool, growth, width):
    pool = pool[np.isfinite(growth[pool])]
    buckets = tuple(pool[i:i + width] for i in range(0, len(pool), width))
    scores = np.full(len(pool), np.inf)
    for i, bucket in enumerate(buckets):
        values = growth[bucket]
        deviation = values.std()
        if deviation > 0:
            scores[i * width:i * width + len(bucket)] = (values - values.mean()) / deviation
    index = np.flatnonzero(scores < 0)
    index = index[np.argsort(scores[index], kind="stable")]
    return Selection(pool[index], buckets, index // width)


def plan(market, params, order=None, *, rng=None):
    order = universe(market) if order is None else order
    selections = {}
    for day in schedule(market, params.cadence):
        pool = eligible(market, order[day], day, params.floor)
        selection = rank(pool, trailing(market, day, params.window, params.skip), params.width)
        names = selection.names[:params.n]
        if rng is not None and len(names):
            groups, counts = np.unique(selection.bucket_ids[:params.n], return_counts=True)
            names = np.concatenate([rng.choice(selection.buckets[g], count, replace=False)
                                    for g, count in zip(groups, counts)])
        selections[day] = Selection(names, selection.buckets, selection.bucket_ids[:params.n])
    return selections


def build_weights(market, params):
    """Return adjusted daily returns and daily target weights, with cash kept implicit."""
    target = np.zeros(market.cap.shape, dtype=float)
    selections = list(plan(market, params).items())
    for i, (day, selected) in enumerate(selections):
        end = selections[i + 1][0] if i + 1 < len(selections) else len(target)
        target[day:end, selected.names] = 1 / params.n
    return (pd.DataFrame(market.returns, index=market.dates, columns=market.codes),
            pd.DataFrame(target, index=market.dates, columns=market.codes))
