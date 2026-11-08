"""策略二：宽基 ETF 动量轮动（主推）。

逻辑：
  1. 每月最后交易日收盘打分：score = 近 lookback 个交易日涨幅（默认 120 日 ≈ 半年动量）；
  2. 只保留 score > 0 的标的（绝对动量过滤，全为负则空仓持现金）；
  3. 取前 top_n 只等权，下月持有，月末再平衡。

优势：标的少（6 只）、无个股黑天鹅与退市风险、换手低、无需成分股/财务数据，
      没有幸存者偏差（这几只 ETF 在回测起点均已上市）。
"""

from __future__ import annotations

import pandas as pd

from qs.data import ETF_POOL, load_panel


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每个月最后一个实际交易日。"""
    s = pd.Series(index, index=index)
    return s.groupby([index.year, index.month]).max().values


def build_weights(symbols: list[str] | None = None, lookback: int = 120,
                  top_n: int = 2, start: str = "2015-01-01", end: str | None = None,
                  min_score: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = symbols or list(ETF_POOL)
    panel = load_panel(symbols, start=start, end=end)
    rets = panel.pct_change(fill_method=None)

    mom = panel / panel.shift(lookback) - 1.0
    rebal_dates = month_end_dates(panel.index[panel.index >= pd.Timestamp(start)])

    rows = {}
    for d in rebal_dates:
        score = mom.loc[d].dropna()
        if score.empty:
            continue
        score = score[score > min_score].sort_values(ascending=False)
        pick = score.head(top_n)
        w = pd.Series(0.0, index=panel.columns)
        if len(pick):
            w[pick.index] = 1.0 / len(pick)
        rows[d] = w

    weights = pd.DataFrame(rows).T.sort_index()
    return rets.loc[weights.index[0]:] if len(weights) else rets, weights
