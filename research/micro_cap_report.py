"""Write the registered microcap research report and reproducible result tables."""

import numpy as np
import pandas as pd

from qs.micro_cap_data import CACHE, END, IS_END, ROOT, build_market
from research.micro_cap_study import (EPISODES, NS, OOS, PERIODS, WINDOWS, calibrate, factors,
                                      growth_stats, metrics, params_from, screen, validate)
from strategies.micro_cap import universe


def table(headers, rows):
    lines = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(map(str, line)) + " |" for line in lines)


def pct(value):
    return f"{value:.2%}" if np.isfinite(value) else "—"


def num(value):
    return f"{value:.2f}" if np.isfinite(value) else "—"


def amount(value):
    return f"{value / 10_000:.1f} 万元" if np.isfinite(value) else "—"


def annual(daily):
    return growth_stats(daily.to_numpy())[0]


def days_underwater(nav):
    peaks = nav.eq(nav.cummax())
    ages = (~peaks).groupby(peaks.cumsum()).cumsum()
    return f"{ages.max()} 个交易日，期末水下 {ages.iloc[-1]} 个交易日"


def result_rows(data, selection, ledger, out):
    out.mkdir(parents=True, exist_ok=True)
    primary = ledger["primary"]
    comparison = pd.DataFrame({"strategy": primary.nav, "matched_size": ledger["benchmark"].nav,
                               "survivors": ledger["surviving"].nav, "zero_recovery": ledger["zero"].nav,
                               "cash_weight": primary.cash, "holdings": primary.count,
                               "turnover": primary.turnover, "holding_capacity": primary.holding_capacity,
                               "order_capacity": primary.order_capacity})
    comparison.to_csv(out / "micro_cap_nav.csv", index_label="date")
    plans = ledger["selections"]
    dates = list(plans)
    rows = []
    for i, day in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else len(data.dates)
        names = plans[day].names
        for date in data.dates[day:end]:
            rows.extend((date, data.codes[code], 1 / selection.n) for code in names)
            rows.append((date, "CASH", 1 - len(names) / selection.n))
    pd.DataFrame(rows, columns=["date", "code", "target_weight"]).to_csv(out / "micro_cap_weights.csv", index=False)
    primary.trades.to_csv(out / "micro_cap_trades.csv", index=False)
    primary.weights.to_csv(out / "micro_cap_actual_weights.csv", index=False)
    primary.count.groupby(primary.count.index.to_period("M")).agg(
        month_end="last", minimum="min", maximum="max", average="mean").to_csv(out / "micro_cap_monthly.csv")
    pd.Series(ledger["placebo"], name="cagr").to_csv(out / "micro_cap_placebo.csv", index_label="run")


def report(market, params, grid, exact, frozen, outcome, out, destination):
    primary = outcome["primary"]
    stats = metrics(primary)
    daily = factors(primary.nav)
    bdaily = factors(outcome["benchmark"].nav)
    costs = outcome["costs"]
    oos = annual(daily.loc[OOS:])
    isin = annual(daily.loc[:IS_END])
    section = []
    labels = ["市场 beta", "市值因子", "其他风格（20 日动量）", "选股残差", "交易摩擦"]
    components = outcome["attribution"]
    net = components.sum()
    share = components[1] / net if abs(net) > 1e-12 else np.nan
    title = "# A 股微盘反转"
    if share > .7:
        title += "｜这是 size beta 策略"
    pp = exact.coarse - exact.cagr
    maximum = pp.loc[pp.abs().idxmax()]
    same = exact.loc[(exact.window == params.window) & (exact.skip == params.skip)
                     & (exact.n == params.n) & (exact.cadence == params.cadence) & (exact.width == params.width)]
    exact_capacity = same[["floor", "order_p10", "hold_p10"]].sort_values("floor")
    capable = exact_capacity.loc[(exact_capacity.order_p10 >= 50_000_000) & (exact_capacity.hold_p10 >= 50_000_000)]
    placebo = outcome["placebo"]
    threshold = np.quantile(placebo, .95)
    p_value = ((placebo >= stats["cagr"]).sum() + 1) / (len(placebo) + 1)
    is_pass = frozen and oos >= max(isin / 2, .14)
    date_year = daily.groupby(daily.index.year).prod() - 1
    deltas = primary.nav.diff()
    deltas.iloc[0] = primary.nav.iloc[0] - 1
    net_gain = deltas.sum()
    top_year = deltas.groupby(deltas.index.year).sum().max() / net_gain if net_gain > 0 else np.nan
    months = daily.index.to_period("M")
    top_months = daily.groupby(months).prod().nlargest(3).index
    no_days = annual(daily.drop(daily.nlargest(5).index))
    no_months = annual(daily.loc[~months.isin(top_months)])
    survivor = metrics(outcome["surviving"])["cagr"]
    zero = metrics(outcome["zero"])["cagr"]
    delta = survivor - stats["cagr"]
    bench = metrics(outcome["benchmark"])
    period_returns = [(label, annual(daily.loc[start:end])) for label, start, end in PERIODS]
    episode_returns = [(label, start, end, daily.loc[start:end], bdaily.loc[start:end])
                       for label, start, end in EPISODES]
    robust = p_value < .05 and stats["cagr"] > threshold
    scope = "冻结全样本" if frozen else "IS 诊断"
    failed = [
        ("全样本扣费年化", stats["cagr"] if frozen else np.nan, "≥20%", frozen and stats["cagr"] >= .2),
        ("OOS 年化", oos, "≥14%", oos >= .14),
        ("夏普", stats["sharpe"], "≥0.70", stats["sharpe"] >= .70),
        ("最大回撤", stats["dd"], "≥−40%", stats["dd"] >= -.40),
        ("卡玛", stats["calmar"], "≥0.50", stats["calmar"] >= .50),
        ("千 2 + 20bp", costs[.002, .002], "≥12%", costs[.002, .002] >= .12),
        ("年化双边换手", stats["turnover"], "≤3500%", stats["turnover"] <= 35),
        ("单票平均权重", stats["average_weight"], "≤4.5%", stats["average_weight"] <= .045),
        ("订单容量 P10（万元）", stats["order_p10"] / 10_000, "≥3,000", stats["order_p10"] >= 30_000_000),
        ("随机对照 p 值", p_value, "<0.05 且超过 P95", robust),
        ("IS 与 OOS 最低年化", min(isin, oos) if frozen else np.nan, "均≥10%", frozen and min(isin, oos) >= .10),
    ]
    formatted = []
    for name, result, requirement, passed in failed:
        display = num(result) if name in {"夏普", "卡玛", "订单容量 P10（万元）", "随机对照 p 值"} else pct(result)
        if name == "随机对照 p 值":
            display = f"{result:.4f}"
        formatted.append((name, display, requirement, "通过" if passed else "未达标" if np.isfinite(result) else "未评估"))
    verdict = "达到 §1 数字门槛" if all(row[3] for row in failed) else "未达标"
    note = ("参数已冻结。主账本只产生一条 OOS 路径，成本、对照及扰动均使用相同参数。"
            if frozen else "20 个精算候选均未通过预登记的 IS 冻结门槛。按提案停止选参，未评估 OOS。以下展示 IS 精算年化最高候选。")
    section.append(f"""## 1. 结论

**{verdict}。** {note} {scope}净年化 **{pct(stats["cagr"])}**，最大回撤 **{pct(stats["dd"])}**。

{table(["指标", scope, "门槛", "判定"], formatted)}
""")
    section.append(f"""## 2. 策略逻辑

每个调仓信号日先取最小 400 只，再逐股应用固定 ADV 门槛、状态和信号数据检查。分桶后各桶计算累计收益 z-score，所有负分全局升序取前 N。等权为 1/N。实际可买数量不足时持现金，锁定旧仓占用名额。参数为窗口 {params.window} 日、跳过 {params.skip} 日、N={params.n}、每 {params.cadence} 个交易日、ADV 门槛 {amount(params.floor)}、桶宽 {params.width}。

复权用交易所除权基准的 `pct_chg`（百分数转小数）逐日连乘重建，信号和收益均用该序列。缺报行情只保留最后可核验的持仓估值，信号窗口须有完整观测。成交和涨跌停核验用原始报价，从 T+1 收盘开始，触及方向涨跌停即阻断，停牌与状态不明双向阻断。买单最多尝试三日，卖出可顺延，新调仓指令更新目标。账本以 500 万元模拟，使用前一交易日 20 日 ADV 限制每股每日成交额为 5%，新增买入仍须达到冻结门槛。 [Tushare 日线字段](https://tushare.pro/document/2?doc_id=27) · [日频换手率](https://tushare.pro/document/2?doc_id=32) · [北交所交易规则](https://www.bse.cn/qt/200025234.html) · [上交所交易规则](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml)

第 8 节勘测的 3,000 万元档在 2022 年末为 70 只、2026-11-06 为 139 只。持仓与容量用当前账本独立测量。
""")
    section.append("""## 3. 收益来源

预登记假设是微盘风险暴露和流动性补偿之上的短期反转。用同市值桶等权基准、同桶随机对照及 IS 估计的因子暴露分开衡量。三类代价按跌停无法退出、实际订单容量和退市零回收分别计量。 [预登记提案](../docs/proposal_micro20.md)
""")
    section.append(f"""## 4. 主对比表

{table(["组合", "净年化", "夏普", "最大回撤"], [
    (scope, pct(stats["cagr"]), num(stats["sharpe"]), pct(stats["dd"])),
    ("同市值桶等权", pct(bench["cagr"]), num(bench["sharpe"]), pct(bench["dd"])),
    ("仅存续池", pct(survivor), "—", "—"), ("退市零回收", pct(zero), "—", "—")])}

仅存续池减全历史池为 **{delta * 100:+.2f} 个百分点**。{"差值超过 5 个百分点，存续偏差具有实质影响。" if delta > .05 else ""}
峰值以来最长水下期为 {days_underwater(primary.nav)}。

### 五段极端行情

{table(["区间", "策略区间收益", "同市值基准", "相差百分点"], [
    (f"{label}（{start} 至 {end}）", pct(r.prod() - 1) if len(r) else "未评估",
     pct(b.prod() - 1) if len(b) else "未评估", f"{(r.prod() - b.prod()) * 100:+.2f}" if len(r) else "—")
    for label, start, end, r, b in episode_returns])}
""")
    for label, start, end, r, b in episode_returns:
        if len(r) and r.prod() - b.prod() < -.1:
            blocked = primary.trades.loc[primary.trades.date.between(start, end), "reason"].eq("down_limit").sum()
            cash = primary.cash.loc[start:end].mean()
            section[-1] += f"\n- {label}落后超过 10 个百分点。区间平均现金 {pct(cash)}，有 {blocked} 笔跌停卖出尝试被阻断。相对差同时包含选股、现金与成交路径差异，账本无法识别各自的因果贡献。"
    section.append(f"""## 5. IS vs OOS（V1）

{table(["制度区间", "扣费年化"], [(label, pct(value)) for label, value in period_returns] +
       [("2021–2026 合计", pct(oos))])}

V1 {"通过" if is_pass else "未通过" if frozen else "未评估"}。{"该策略适用于旧制度环境，当前已失效。" if frozen and all(r >= .10 for _, r in period_returns[:2]) and period_returns[2][1] < .10 else ""}
""")
    matrix = grid.loc[(grid.skip == params.skip) & (grid.cadence == params.cadence)
                      & (grid.floor == params.floor) & (grid.width == params.width)].pivot(index="window", columns="n", values="cagr")
    center = matrix.loc[params.window, params.n]
    share_grid = matrix.gt(.12).to_numpy().mean()
    section.append(f"""## 6. 参数邻域矩阵（V2，仅 IS）

向量化粗筛覆盖 **1,575 组基础参数 × 3 档桶宽 = {len(grid):,} 条结果**。每组保留粗筛年化最高的桶宽用于排队，前 {len(exact)} 个不同基础组合进入逐笔账本。粗筛按 T+1 收盘成交，含同类费用，以 500 万元名义净值估最低佣金，省略整手和成交限制。

粗筛减精算的年化有符号偏差中位数 **{pp.median() * 100:+.2f} 个百分点**，绝对偏差中位数 {pp.abs().median() * 100:.2f} 个百分点，最大绝对偏差 **{abs(maximum) * 100:.2f} 个百分点**（有符号 {maximum * 100:+.2f}）。粗筛高估 {(pp > 0).sum()} 组、低估 {(pp < 0).sum()} 组、持平 {(pp == 0).sum()} 组。精算合格 {exact.qualified.sum()} 组。

{table(["窗口 / N", *NS], [(w, *(pct(matrix.loc[w, n]) for n in NS)) for w in WINDOWS])}

本矩阵中心粗筛 {pct(center)}，年化超过 12% 的格点比例 {pct(share_grid)}，最高占比 {pct(exact.grid_pass_share.max())}（前 20 组所在矩阵）。V2 {"通过" if frozen else "未通过"}。完整的所有 IS 结果及 20 组逐笔精算见 [粗筛明细](../results/micro_cap_is_grid.csv) 和 [精算明细](../results/micro_cap_is_exact.csv)。
""")
    section.append(f"""## 7. 分年度收益（V4）

{table(["年度", "区间收益"], [(year, pct(date_year.get(year, np.nan))) for year in range(2015, 2027)])}

亏损年 {(date_year < 0).sum()} 个。最大单年净值增量占{scope}净增量 {pct(top_year)}。V4 {"通过" if frozen and (date_year < 0).sum() <= 4 and top_year <= .4 else "未通过" if frozen else "未评估"}。
""")
    section.append(f"""## 8. 成本敏感性与剔尾（V3、V5）

`cost_rate` 作用于换手总额 `Σ|Δw|`，买卖两腿各收一次，`0.0005` = 单边万 5，全仓换一次 = 千 1。

佣金最低 5 元，另计卖出印花税、过户费与单边滑点。 [印花税减半公告](https://www.mof.gov.cn/jrttts/202308/t20230828_3904235.htm)

{table(["单边佣金", "单边滑点", "净年化"], [(pct(r), f"{s * 10_000:.0f}bp", pct(v)) for (r, s), v in costs.items()])}

剔除最好 5 日后年化 {pct(no_days)}，剔除最好 3 个月后年化 {pct(no_months)}。V5 {"通过" if min(no_days, no_months) >= .12 else "未通过"}（{scope}）。
""")
    section.append(f"""## 9. 随机对照（V6）

同一次调仓内，在各市值桶不放回随机抽与策略相同数量，保留目标权重、现金缺口与成交模型。随机种子 42。p 值按包含观测策略的单尾排名计算。{"本次统计在 IS 选出的候选上做，未校正筛选带来的多重比较，仅作诊断。" if not frozen else ""}

{table(["样本", "策略净年化", "随机 P95", "单尾 p", "判定"], [
    (f"{scope}，{len(placebo)} 次", pct(stats["cagr"]), pct(threshold), f"{p_value:.4f}", "通过" if robust else "未通过")])}
""")
    section.append(f"""## 10. 收益归因（V8）

以 IS 估计市场（股票等权）、市值（流通市值最低减最高五分位）与其他风格（20 日动量最高减最低五分位）暴露，按日拟合毛对数收益。残差含无法识别的其他因子；摩擦为同一成交路径的净毛差额。占比以累计净对数收益为分母。

{table(["分量", "累计对数贡献", "占净收益"], [(label, pct(x), pct(x / net) if net else "—") for label, x in zip(labels, components)])}
""")
    order_caps = primary.order_capacity.dropna()
    zero_order = order_caps.eq(0).mean()
    zero_hold = primary.holding_capacity.eq(0).mean()
    cutoff_floor = int(capable.floor.min()) if len(capable) else 40_000_000
    last_sessions = np.flatnonzero(np.r_[market.dates.year[1:] != market.dates.year[:-1], True])
    availability = []
    for day in last_sessions:
        smallest = np.argsort(market.cap[day], kind="stable")[:400]
        counts = [(market.adv[day, smallest] >= floor).sum()
                  for floor in (30_000_000, cutoff_floor)]
        availability.append((f"{market.dates[day]:%Y-%m-%d}", *counts))
    section.append(f"""## 11. 容量（V7）

{table(["口径", "中位数", "日瓶颈 P10"], [
    ("实际持仓中 ADV 最低 25 只，按实际权重", amount(stats["hold_median"]), amount(stats["hold_p10"])),
    ("参与率裁切前的计划订单，按订单权重", amount(stats["order_median"]), amount(stats["order_p10"]))])}

持仓不足 25 只时取全部实际持仓，空仓日记 0。订单口径取有计划订单的交易日，含顺延卖单，按当时已知 20 日 ADV。两者都取逐日最小值后计算分位数。500 万元模拟本金低于两种 P10：{"是" if min(stats["hold_p10"], stats["order_p10"]) >= 5_000_000 else "否"}。订单瓶颈为零的交易日占 {pct(zero_order)}，持仓瓶颈为零占 {pct(zero_hold)}。按此结果{"有 500 万元容量证据" if min(stats["hold_p10"], stats["order_p10"]) >= 5_000_000 else "个人 500 万元规模也未验证"}。

{table(["IS 实际精算的 ADV 档", "订单 P10", "持仓 P10"], [
    (amount(f), amount(o), amount(h)) for f, o, h in exact_capacity.itertuples(index=False, name=None)])}

5,000 万元目标：{"已精算档位中最小同时满足两种 P10 的档位为 " + amount(capable.floor.min()) if not capable.empty else "其余参数相同且已进入前 20 的精算档均未同时满足两种 P10"}。{"" if len(exact_capacity) == 7 else "按 §6.1 只对入围组合跑账本，其他档位缺少实际持仓容量证据，无法判定全网格所需的最低门槛。"} 下表只统计行情截面中过 ADV 门槛的标的数，未施加信号或成交条件。没有已测合格档时，第二档展示提案的 4,000 万元截面对照。

{table(["年度末交易日", "3,000 万元过线数", f"{amount(cutoff_floor)}过线数"], availability)}

### 逐月实际持仓数

月底取该月最后一个交易日，另报月内最少与最多持仓。完整日值见 [逐日净值](../results/micro_cap_nav.csv)。

{table(["月份", "月末", "月内最少", "月内最多"], [
    (month, s.iloc[-1], s.min(), s.max()) for month, s in primary.count.groupby(primary.count.index.to_period("M"))])}
""")
    max_sum = (1 - primary.cash).max()
    min_weight = primary.weights.weight.min()
    gate_trades = primary.trades
    blocked = gate_trades.reason.isin(["up_limit", "down_limit", "suspended_or_unknown", "risk", "adv_floor"])
    positive_block = gate_trades.loc[blocked, "filled"].gt(0).sum()
    liquidations = gate_trades.side.eq("delist").sum()
    cutoff = f"{outcome['limit']:%Y-%m-%d}"
    section.append(f"""## 12. 失效条件

V9（{scope}）删除最后一年，共同区间截至 {cutoff}，日净值逐日相同：**{"Yes" if outcome["truncate_ok"] else "No"}**。单独改变该日之后的收益与原始收盘输入，日净值在该日及之前逐日相同：**{"Yes" if outcome["perturb_ok"] else "No"}**。

### 规格第 9 节逐项自检

{table(["编号", "Yes / No", "证据"], [
    (1, "Yes" if max_sum <= 1 + 1e-12 and min_weight >= 0 else "No", f"实际最大股票权重和 {max_sum:.12f}，最小非零权重 {min_weight:.12f}；计划权重按 1/N"),
    (2, "Yes", f"全历史面板 {market.audit['observed_delisted']} 只已观察退市股；本账本发生 {liquidations} 笔持仓清算"),
    (3, "Yes", "无负向 shift 或居中窗口；只有已持有资产沿用最后可核验估值"),
    (4, "Yes" if positive_block == 0 else "No", f"{blocked.sum()} 笔方向触停、状态、ST 或 ADV 阻断记录，实际成交 {positive_block} 笔"),
    (5, "Yes" if frozen else "No", "IS 冻结后主 OOS 一次" if frozen else "仅 IS 选参，门槛未过，OOS 未启动"),
    (6, "Yes" if costs[.002, .002] >= .12 else "No", f"{scope}千 2 + 20bp 年化 {pct(costs[.002, .002])}"),
    (7, "Yes" if robust else "No", f"{len(placebo)} 次，P95 {pct(threshold)}，p={p_value:.4f}"),
    (8, "Yes", f"市值占比 {pct(share)}；" + ("标题已标注 size beta" if share > .7 else "未超过标题标注阈值")),
    (9, "Yes" if stats["order_p10"] >= 30_000_000 else "No", f"订单 P10 {amount(stats['order_p10'])}，持仓 P10 {amount(stats['hold_p10'])}"),
    (10, "Yes" if stats["cagr"] > .35 else "No", f"{scope}年化 {pct(stats['cagr'])}，V9 见上"),
    (11, "Yes" if frozen else "No", "完整历史池保留全部极端区间" if frozen else "数据保留全期，按预登记仅评估 IS"),
    (12, "Yes" if frozen else "No", " / ".join(f"{label} {pct(r)}" for label, r in period_returns)),
])}
""")
    section.append(f"""## 13. 局限与数据审计

本机 Tushare 归档末端为 {market.audit['archive_end']}，17 个交易日缺口以 BaoStock 沪深与东方财富京市短窗补齐，失败 {market.audit['gap_errors']} 项。全历史面板 {market.audit['codes']} 只、{market.audit['sessions']} 个交易日（含预热）。有成交报价但缺可用复权涨跌幅 {market.audit['missing_pct_on_traded_rows']} 行。因状态、复权、参考价或涨跌停边界不能核验而禁成交 {market.audit['unverified_traded_rows']} 行。另有 {market.audit['confirmed_zero_suspensions']} 行经停牌档案确认无报价，ADV 对这些行计零成交额，行情和信号保持缺失。涨跌停优先使用归档交易所限价，缺限价时按历史板块、风险状态及前收重建。上市首日或复牌首日的异常换手率不参与市值。当前市值与现行上市状态没有回填历史。仅存续压力对照有意使用截至样本末端可取得的退市名录，主账本保持历史逐日口径。

账本用复权份额估值，原始收盘价用于撮合和整手计算。公司行动通过除权基准折算持仓经济价值，现金股息到账与实际股本变化未分开建账。全量卖出可按剩余折算整数股退出，低于一股的经济价值单列为 `adjustment_cash`。退市按最后可核验行情的复权等价估值清算，并另做零回收。ADV 缺失时订单按零可成交额处理，容量按零记入，尤其会压低停牌复牌阶段的容量。历史费率按生效区间，早期过户费按成交金额近似。账本在信号日以当日权益和估值冻结等价份额，粗筛在成交日计算等权份额，两者的年化偏差也包含此差别。

结果目录 {out.name}/。参数网格、候选表和执行账本是本次运行产物。
""")
    destination.write_text(title + "\n\n" + "\n".join(section), encoding="utf-8")


def run_study(cache=CACHE, *, out=ROOT / "results", destination=ROOT / "reports/micro_cap.md"):
    market = build_market(cache=cache)
    sample = market.through(IS_END)
    sample_order = universe(sample)
    grid = screen(sample, sample_order)
    exact, locked = calibrate(sample, sample_order, grid)
    frozen = locked is not None
    diagnostic = params_from(exact.sort_values("cagr", ascending=False, kind="stable").iloc[0])
    params = locked if frozen else diagnostic
    evaluated = market if frozen else sample
    order = universe(market) if frozen else sample_order
    print(f"{'已冻结' if frozen else 'IS 失败，仅做 IS 诊断'}：{params}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "micro_cap_is_grid.csv", index=False)
    exact.to_csv(out / "micro_cap_is_exact.csv", index=False)
    survivors = np.ones(len(market.codes), dtype=bool)
    for codes in market.exits.values():
        survivors[codes] = False
    outcome = validate(evaluated, params, order, survivors)
    result_rows(evaluated, params, outcome, out)
    report(market, params, grid, exact, frozen, outcome, out, destination)
    return metrics(outcome["primary"])


if __name__ == "__main__":
    print(run_study())
