"""ETF momentum rotation with breadth regimes and a rank buffer."""

import pandas as pd

from strategies._rotation import daily_returns, load_prices, rebalance_dates

# Selected by asset coverage before computing returns.
ETF_UNIVERSE = {
    "510300": ("沪深300ETF", "A股宽基"),
    "510500": ("中证500ETF", "A股宽基"),
    "512100": ("中证1000ETF", "A股宽基"),
    "159915": ("创业板ETF", "A股宽基"),
    "588000": ("科创50ETF", "A股宽基"),
    "513100": ("纳指ETF", "跨境"),
    "513500": ("标普500ETF", "跨境"),
    "513520": ("日经ETF", "跨境"),
    "513030": ("德国ETF", "跨境"),
    "159920": ("恒生ETF", "跨境"),
    "513180": ("恒生科技ETF", "跨境"),
    "518880": ("黄金ETF", "商品"),
    "159985": ("豆粕ETF", "商品"),
    "511010": ("国债ETF", "债券"),
    "511360": ("短融ETF", "债券"),
}
BONDS = tuple(symbol for symbol, (_, group) in ETF_UNIVERSE.items() if group == "债券")
DEFAULT_LOOKBACK = 120
DEFAULT_TOP_N = 2
BREADTH_THRESHOLD = 0.60
BOND_BUDGET = 0.30
RANK_BUFFER = 1


def buffered_picks(scores, incumbents, count):
    ranked = scores.sort_values(ascending=False, kind="stable").index
    keep = [symbol for symbol in ranked[:count + RANK_BUFFER] if symbol in incumbents]
    return (keep + [symbol for symbol in ranked if symbol not in keep])[:count]


def target_weights(panel, lookback, top_n, start, frequency="monthly"):
    momentum = panel.div(panel.shift(lookback)).sub(1)
    dates = rebalance_dates(panel.index, start, frequency)
    weights = pd.DataFrame(0.0, index=dates, columns=panel.columns)
    incumbents = set()
    for date in dates:
        scores = momentum.loc[date].dropna()
        positive = scores[scores.gt(0)]
        if scores.empty:
            incumbents = set()
            continue
        if len(positive) / len(scores) >= BREADTH_THRESHOLD:
            picks = buffered_picks(positive, incumbents, top_n)
            if picks:
                weights.loc[date, picks] = 1 / len(picks)
        else:
            risky = positive.drop(list(BONDS), errors="ignore")
            picks = buffered_picks(risky, incumbents, top_n + 3)
            if picks:
                weights.loc[date, picks] = (1 - BOND_BUDGET) / len(picks)
            bonds = positive.loc[positive.index.intersection(BONDS)]
            if len(bonds):
                bond = buffered_picks(bonds, incumbents, 1)[0]
                weights.loc[date, bond] = BOND_BUDGET
        incumbents = set(weights.columns[weights.loc[date].gt(0)])
    return weights


def build_weights(symbols=None, lookback=DEFAULT_LOOKBACK, top_n=DEFAULT_TOP_N,
                  start="2016-01-01", end=None, *, frequency="monthly"):
    panel = load_prices(list(symbols or ETF_UNIVERSE), lookback, start, end)
    weights = target_weights(panel, lookback, top_n, start, frequency)
    return daily_returns(panel).loc[start:end], weights
