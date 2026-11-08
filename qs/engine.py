"""极简回测引擎：给定「每日收益面板 + 目标权重表」，逐日撮合出净值曲线。

约定（避免未来函数）：
  * 权重表某一行 date=t 表示「在 t 日收盘完成调仓」，收益从下一交易日开始计入；
  * 未分配权重的部分视为现金，收益为 0（不计利息，保守处理）；
  * 调仓当日按换手额收取单边手续费 cost_rate。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(returns: pd.DataFrame, weights: pd.DataFrame,
                 cost_rate: float = 0.0005, init_nav: float = 1.0) -> tuple[pd.Series, pd.DataFrame]:
    """返回 (每日净值序列, 每日持仓明细)。"""
    returns = returns.sort_index()
    weights = weights.reindex(columns=returns.columns).fillna(0.0)

    nav = init_nav
    cur_w = pd.Series(0.0, index=returns.columns)   # 调仓后、次日收益前的权重
    nav_list, w_list, idx = [], [], []

    rebalance_map = {d: weights.loc[d] for d in weights.index if d in returns.index}

    for i, (date, ret) in enumerate(returns.iterrows()):
        if i > 0:                                   # 先吃上一交易日权重对应的收益
            port_ret = float((cur_w * ret.fillna(0.0)).sum())
            nav *= 1.0 + port_ret

        if date in rebalance_map:                   # 收盘调仓：交割成本，权重切到目标
            target = rebalance_map[date]
            turnover = float((target - cur_w).abs().sum())
            nav *= 1.0 - turnover * cost_rate
            cur_w = target.copy()
            # 记录权重漂移前的目标持仓，便于查看
        idx.append(date)
        nav_list.append(nav)
        w_list.append(cur_w.copy())

    nav_s = pd.Series(nav_list, index=pd.DatetimeIndex(idx), name="nav")
    pos = pd.DataFrame(w_list, index=nav_s.index)
    return nav_s, pos


def perf_stats(nav: pd.Series, weights: pd.DataFrame | None = None,
               periods_per_year: int = 252) -> dict:
    """年化收益 / 波动 / 夏普 / 最大回撤 / 卡玛 / 年化换手。"""
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * np.sqrt(periods_per_year)
    sharpe = cagr / vol if vol > 0 else np.nan
    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    turnover = np.nan
    if weights is not None and len(weights) > 1:
        chg = weights.diff().abs().sum(axis=1).iloc[1:]
        turnover = chg.sum() / max(years, 1e-9)     # 年化双边换手

    monthly = nav.resample("ME").last().pct_change().dropna()
    return {
        "累计收益": nav.iloc[-1] / nav.iloc[0] - 1,
        "年化收益": cagr,
        "年化波动": vol,
        "夏普": sharpe,
        "最大回撤": mdd,
        "卡玛": calmar,
        "月胜率": (monthly > 0).mean() if len(monthly) else np.nan,
        "年化换手": turnover,
        "区间": f"{nav.index[0]:%Y-%m-%d} ~ {nav.index[-1]:%Y-%m-%d}",
    }


def format_stats(stats: dict) -> str:
    pct = ("累计收益", "年化收益", "年化波动", "最大回撤", "月胜率", "年化换手")
    lines = []
    for k, v in stats.items():
        if k == "区间":
            lines.append(f"{k}: {v}")
        elif isinstance(v, float) and np.isnan(v):
            lines.append(f"{k}: -")
        elif k in pct:
            lines.append(f"{k}: {v:.2%}")
        else:
            lines.append(f"{k}: {v:.2f}")
    return "\n".join(lines)
