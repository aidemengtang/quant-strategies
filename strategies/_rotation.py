"""Price and calendar operations shared by the two ETF candidates."""

import pandas as pd

from qs.data import load_panel, price_adjustment

BIWEEKLY_ANCHOR = pd.Timestamp("2015-01-05")


def load_prices(symbols, lookback, start, end):
    start_date = pd.Timestamp(start)
    history_start = min(
        start_date - pd.DateOffset(years=1),
        start_date - pd.offsets.BDay(lookback),
    )
    with price_adjustment("hfq"):
        panel = load_panel(symbols, start=history_start.strftime("%Y-%m-%d"), end=end)
    return panel.reindex(columns=symbols)


def daily_returns(panel):
    # Use the prior actual close on resumption, keeping absent sessions missing.
    return panel.apply(lambda close: close.dropna().pct_change().reindex(panel.index))


def rebalance_dates(index, start, frequency):
    if frequency == "monthly":
        periods = index.to_period("M")
    elif frequency == "biweekly":
        periods = (index - BIWEEKLY_ANCHOR).days // 14
    else:
        raise ValueError(f"unsupported rebalance frequency: {frequency}")
    return index[(~periods.duplicated()) & (index >= pd.Timestamp(start))]
