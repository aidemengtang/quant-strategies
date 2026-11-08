"""Randomize asset identities while preserving every trade's turnover."""

from functools import lru_cache

import numpy as np
import pandas as pd

from qs.engine import run_backtest


def randomized_weights(weights, available, *, trials=200, seed=42):
    values = weights.to_numpy()
    held = values > 0
    eligible = available.reindex_like(weights).fillna(False).to_numpy(dtype=bool)
    allowed = ~(held.T.astype(int) @ (~eligible).astype(int)).astype(bool)
    order = np.argsort(allowed.sum(axis=1), kind="stable")
    choices = [tuple(np.flatnonzero(allowed[source])) for source in order]
    size = len(order)

    @lru_cache(maxsize=None)
    def completions(used):
        row = used.bit_count()
        if row == size:
            return 1
        return sum(
            completions(used | (1 << int(target)))
            for target in choices[row] if not used & (1 << int(target))
        )

    if not completions(0):
        raise ValueError("No eligible permutation of ETF weight trajectories")

    rng = np.random.default_rng(seed)
    for _ in range(trials):
        used = 0
        mapping = np.empty(size, dtype=int)
        for source, candidates in zip(order, choices):
            options = [int(target) for target in candidates if not used & (1 << int(target))]
            counts = [completions(used | (1 << target)) for target in options]
            draw = rng.integers(sum(counts))
            target = options[np.searchsorted(np.cumsum(counts), draw, side="right")]
            mapping[source] = target
            used |= 1 << target
        shuffled = np.empty_like(values)
        shuffled[:, mapping] = values
        yield pd.DataFrame(shuffled, index=weights.index, columns=weights.columns)


def cagr_distribution(returns, weights, available, cost_rate):
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    distribution = []
    for shuffled in randomized_weights(weights, available):
        nav, _ = run_backtest(returns, shuffled, cost_rate=cost_rate)
        distribution.append((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)
    return np.array(distribution)
