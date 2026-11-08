"""Positive momentum and trend, with inverse-volatility allocations."""

import pandas as pd

from strategies._rotation import daily_returns, load_prices, rebalance_dates
from strategies.candidate_b import ETF_UNIVERSE

DEFAULT_LOOKBACK = 120
DEFAULT_TOP_N = 3
VOL_WINDOW = 60
TREND_WINDOW = 200


def target_weights(panel, lookback, top_n, start, frequency="monthly"):
    momentum = panel.div(panel.shift(lookback)).sub(1)
    trend = panel.gt(panel.rolling(TREND_WINDOW).mean())
    volatility = daily_returns(panel).rolling(VOL_WINDOW).std()
    scores = momentum.where(momentum.gt(0) & trend & volatility.gt(0))
    dates = rebalance_dates(panel.index, start, frequency)
    weights = pd.DataFrame(0.0, index=dates, columns=panel.columns)
    for date in dates:
        picks = scores.loc[date].dropna().nlargest(top_n).index
        if len(picks):
            inverse_vol = 1 / volatility.loc[date, picks]
            weights.loc[date, picks] = inverse_vol / inverse_vol.sum()
    return weights


def build_weights(symbols=None, lookback=DEFAULT_LOOKBACK, top_n=DEFAULT_TOP_N,
                  start="2016-01-01", end=None, *, frequency="monthly"):
    panel = load_prices(list(symbols or ETF_UNIVERSE), lookback, start, end)
    weights = target_weights(panel, lookback, top_n, start, frequency)
    return daily_returns(panel).loc[start:end], weights
