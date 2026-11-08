"""Replay the frozen ETF strategy and audit it without changing its trades."""

from pathlib import Path

import numpy as np
import pandas as pd

from qs.capacity import ledger_from_rows, snapshot
from qs.data import CACHE_DIR
from qs.engine import run_backtest
from research.capacity_report import (attribution, capacity_audit, export_capacity, load_factors,
                                      verify_nav, write_report)
from research.study import (CENTERS, END, HISTORY_START, IS_END, PRIMARY_COST, backtest_inputs,
                            prices, target_weights)
from strategies._rotation import daily_returns
from strategies.high_return import SELECTED_CANDIDATE

ROOT = Path(__file__).resolve().parents[1]


def audit(returns, weights, amount, *, cost_rate=PRIMARY_COST):
    """Use the engine's closing holdings; inspect same-close rebalance orders."""
    nav, positions = run_backtest(returns, weights, cost_rate=cost_rate)
    adv = amount.reindex(columns=returns.columns).rolling(20, min_periods=20).mean().shift(1)
    rows, diagnostics = [], []
    previous = np.zeros(len(returns.columns))
    costs = pd.Series(0., index=nav.index)
    for date, current in positions.iterrows():
        actual = current.to_numpy()
        orders = np.abs(actual - previous) if date in weights.index else np.zeros_like(actual)
        costs.loc[date] = orders.sum() * cost_rate
        if date in amount.index:
            values, diag = snapshot(date, returns.columns, adv.loc[date].to_numpy(), actual, orders)
            rows.append(values)
            diagnostics.append(diag)
        previous = actual
    gross = (nav / nav.shift(1, fill_value=1.)) / (1 - costs)
    dates = nav.index[nav.index.isin(amount.index)]
    return nav, gross, ledger_from_rows(dates, rows, diagnostics)


def load_amount(symbols, dates):
    """Read cached yuan turnover directly: load_etf intentionally exposes only OHLCV."""
    values = {}
    for code in symbols:
        path = CACHE_DIR / f"{code}_{HISTORY_START.replace('-', '')}_{END.replace('-', '')}_hfq.parquet"
        frame = pd.read_parquet(path)
        if "amount" not in frame:
            raise ValueError(f"{path}: missing verified traded amount")
        values[code] = frame["amount"]
    return pd.DataFrame(values).reindex(dates)


def run_report(*, out=ROOT / "results", destination=ROOT / "reports/high_return_capacity.md",
               factor_returns=None):
    panel = prices()
    weights = target_weights(SELECTED_CANDIDATE, CENTERS[SELECTED_CANDIDATE], panel)
    returns, weights = backtest_inputs(daily_returns(panel), weights)
    nav, gross, ledger = audit(returns, weights, load_amount(panel.columns, panel.index))
    verify_nav(nav, out / "high_return_nav.csv")
    factor_returns = load_factors() if factor_returns is None else factor_returns
    result = attribution(nav, gross, factor_returns, is_end=IS_END)
    export_capacity(ledger, out, "high_return")
    factor_returns.reindex(nav.index, fill_value=0.).to_csv(out / "high_return_factors.csv", index_label="date")
    write_report("high_return", ledger, result, destination, is_end=IS_END, prefix="high_return",
                 details="沿用已冻结的候选 A：120 日动量、前 3、月度、逆波动权重。重放净值与原归档逐日核对。容量使用 ETF 缓存的人民币成交额，不以成交量代替。原引擎按固定目标权重计算日收益，正式调仓间恢复权重的隐含成交未记费或计入订单；容量只审计现有账本的持仓与正式调仓，未增加参与率撮合、涨跌停或滑点假设。2016-01-01 非交易日现金锚点仅用于净值与归因，不计容量日数。A 股三因子对跨境、商品和债券 ETF 的解释能力有限。")
    return ledger, result


if __name__ == "__main__":
    run_report()
