"""Common capacity disclosure and the unchanged micro-cap three-factor formula."""

import numpy as np
import pandas as pd

from qs.micro_cap_data import START, build_market
from research.micro_cap_report import amount, pct, table
from research.micro_cap_study import (Attribution, CAPACITY_SENSITIVITY_START, _cross_mean,
                                      capacity_audit, factors)

FACTOR_NAMES = ["市场", "市值", "20 日动量"]


def three_factors(market):
    """Use exactly micro_cap_study.attribution's universe, cutoffs and missingness."""
    start = market.dates.searchsorted(START)
    if start < 21:
        raise ValueError("Three-factor diagnostics require 21 warmup sessions")
    today = market.returns[start:]
    cap = market.cap[start - 1:len(market.dates) - 1]
    adj = market.adjusted
    trend = adj[start - 1:len(market.dates) - 1] / adj[start - 21:len(market.dates) - 21] - 1
    valid = np.isfinite(today) & np.isfinite(cap)
    with np.errstate(all="ignore"):
        small, large = np.nanquantile(cap, [.2, .8], axis=1)
        weak, strong = np.nanquantile(np.where(valid, trend, np.nan), [.2, .8], axis=1)
    market_ret = _cross_mean(today, valid)
    size = (_cross_mean(today, valid & (cap <= small[:, None]))
            - _cross_mean(today, valid & (cap >= large[:, None])))
    momentum = (_cross_mean(today, valid & (trend >= strong[:, None]))
                - _cross_mean(today, valid & (trend <= weak[:, None])))
    return pd.DataFrame(np.column_stack([market_ret, size, momentum]),
                        index=market.dates[start:], columns=FACTOR_NAMES)


def attribution(nav, gross, factor_returns, *, is_end):
    """No intercept: fit daily gross log returns in IS, freeze betas for all dates."""
    x = factor_returns.reindex(nav.index)
    # The ETF engine has a single non-trading cash anchor; it earns no return.
    if len(x) and x.iloc[0].isna().all() and nav.iloc[0] == gross.iloc[0] == 1:
        x.iloc[0] = 0.
    if x.isna().any().any() or not gross.index.equals(nav.index):
        raise ValueError("Factors and gross returns must cover every NAV date")
    design = x.to_numpy()
    log_gross = np.log(gross.to_numpy())
    log_net = np.log(factors(nav).to_numpy())
    isin = nav.index <= is_end
    if not isin.any():
        raise ValueError("No IS observations for factor exposure estimation")
    betas = np.linalg.lstsq(design[isin], log_gross[isin], rcond=None)[0]
    components = (design * betas).sum(axis=0)
    return Attribution(np.r_[components, log_gross.sum() - components.sum(),
                              (log_net - log_gross).sum()], betas, design.sum(axis=0))


def write_report(title, ledger, result, destination, *, details, is_end, prefix):
    yearly, recent = capacity_audit(ledger)
    counts = yearly[["year", "hold_zero", "hold_unavailable", "hold_empty", "order_zero", "order_unavailable"]]
    rows = [*counts.itertuples(index=False, name=None), ("合计", *counts.iloc[:, 1:].sum())]
    samples = []
    for label, kind, full in (("实际持仓", "holding", ledger.holding_capacity),
                              ("计划订单", "order", ledger.order_capacity.dropna())):
        effective = recent.loc[kind]
        samples.append((label, len(full), amount(full.median()), amount(full.quantile(.1)),
                        int(effective.days), amount(effective["median"] * 10_000),
                        amount(effective.p10 * 10_000)))
    net = result.components.sum()
    labels = ["市场 beta", "市值因子", "其他风格（20 日动量）", "选股残差", "交易摩擦"]
    text = f"""# {title}｜容量与三因子审计

{details}

## 容量（V7）

按 5% × 前一交易日可得的 20 日 ADV ÷ 权重逐日计算。实际持仓中先取 ADV 最低 25 只（不足时取全部），再按实际权重取最小容量；订单取参与率裁切前的绝对计划订单权重。空仓日持仓容量记零，无计划订单日的订单容量留空。逐日取瓶颈后计算分位数。

{table(["口径", "原账本日数", "原账本中位数", "原账本 P10", f"{CAPACITY_SENSITIVITY_START} 起有效日数", "2019 年起中位数", "2019 年起 P10"], samples)}

### 零容量日拆分与 2019 年起的敏感性

真实零指完整 20 日成交额窗口为零；ADV 不可得指不足 20 个可核验观测；空仓单列。与 micro_cap 模板一致，订单侧不统计空仓日，也不统计无计划订单日。全段列保留真实零、不可得导致的零及持仓空仓零；2019 年起列剔除空仓与 ADV 不可得，保留真实零。

{table(["年份", "持仓真实零", "持仓 ADV 不可得", "空仓记零", "订单真实零", "订单 ADV 不可得"], rows)}

逐日瓶颈代码、ADV、实际权重及分类见 [容量诊断](../results/{prefix}_capacity_diag.csv)，逐日两种容量见 [容量序列](../results/{prefix}_capacity_daily.csv)。敏感性列不改全段样本或策略规则。

## 收益归因（V8）

与 micro_cap 使用同一全历史 A 股行情面板及同一公式：市场为股票等权日收益；市值为前一交易日流通市值最低减最高五分位；动量为截至前一交易日的 20 日动量最高减最低五分位。仅要求当日收益与前日市值有限，分位数和缺失值处理完全沿用模板，不叠加策略的选股、状态、流动性或正动量过滤。因子为诊断基准，不是可交易组合。载荷仅用截至 {is_end} 的 IS 毛对数收益进行无截距日频拟合，随后冻结。

{table(["分量", "累计对数贡献", "占净收益"], [(label, pct(value), pct(value / net) if net else "—") for label, value in zip(labels, result.components)])}

{table(["因子", "IS 载荷", "原始日收益累计和", "载荷 × 原始累计和"], [(name, f"{beta:.6f}", pct(raw), pct(beta * raw)) for name, beta, raw in zip(FACTOR_NAMES, result.betas, result.factor_sums)])}

原始因子列为日收益算术累计和，不是复合收益；乘固定载荷后为累计对数贡献。选股残差为同一成交路径毛对数收益减三因子贡献，包含未识别暴露；摩擦为每日净毛对数差。五项未舍入之和为累计净对数收益 {pct(net)}，负残差和摩擦可使因子占比超过 100%。逐日因子见 [三因子序列](../results/{prefix}_factors.csv)。

### 复现

依赖已有 ETF/股票缓存、micro_cap 使用的本地 Tushare 归档和原净值 CSV；不重选参数、不覆盖原净值与权重。报告生成额外需要 pyarrow 读取 parquet，版本约束沿用 requirements.txt。

```bash
uv run --with pytest --with numpy --with pandas --with pyarrow python -m research.{prefix}_capacity
uv run --with pytest --with numpy --with pandas python -m pytest tests/ -q
```
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def export_capacity(ledger, out, prefix):
    out.mkdir(parents=True, exist_ok=True)
    ledger.capacity_diag.to_csv(out / f"{prefix}_capacity_diag.csv", index=False)
    pd.DataFrame({"count": ledger.count, "holding_capacity": ledger.holding_capacity,
                  "order_capacity": ledger.order_capacity}).to_csv(
                      out / f"{prefix}_capacity_daily.csv", index_label="date")


def verify_nav(nav, path):
    archived = pd.read_csv(path, index_col="date", parse_dates=True).nav
    # CSV has no datetime unit/index-name metadata; compare the actual dates and values.
    archived.index = pd.DatetimeIndex(archived.index, dtype=nav.index.dtype, name=nav.index.name)
    archived.index.name = nav.index.name
    pd.testing.assert_series_equal(nav, archived, check_freq=False, rtol=1e-12, atol=1e-12)


def load_factors():
    """Read the same local point-in-time archive as the micro-cap template."""
    return three_factors(build_market())
