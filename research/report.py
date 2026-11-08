"""Render the frozen protocol's report and auditable CSV outputs."""

from pathlib import Path

import numpy as np

from research.study import CENTERS, PRIMARY_COST, analyze
from strategies import candidate_b, high_return

ROOT = Path(__file__).resolve().parents[1]


def percent(value):
    return f"{value:.2%}"


def table(headers, rows):
    lines = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in lines)


def verdict(passed):
    return "通过" if passed else "未通过"


def comparison(stats):
    return [
        percent(stats["年化收益"]), percent(stats["年化波动"]), f'{stats["夏普"]:.2f}',
        percent(stats["最大回撤"]), f'{stats["卡玛"]:.2f}', percent(stats["年化换手"]),
    ]


def grid_table(grid):
    matrix = grid.pivot(index="lookback", columns=["frequency", "top_n"], values="cagr")
    columns = sorted(matrix.columns, key=lambda item: (item[0] != "monthly", item[1]))
    return table(
        ["动量窗口"] + [
            f'{"月度" if frequency == "monthly" else "双周"} / 前 {count}'
            for frequency, count in columns
        ],
        ([days, *(percent(row[column]) for column in columns)] for days, row in matrix.iterrows()),
    )


def render(study):
    winner = high_return.SELECTED_CANDIDATE
    params = CENTERS[winner]
    panel, grid, weights = (study[key] for key in ("panel", "grid", "weights"))
    runs, oos, costs = (study[key] for key in ("runs", "oos", "costs"))
    _, positions, stats = runs[winner]
    cagr = stats["年化收益"]
    primary = [
        ("全样本年化", percent(cagr), "≥ 13.0%", cagr >= 0.13),
        ("OOS 年化", percent(oos[winner]["年化收益"]), "≥ 10.0%", oos[winner]["年化收益"] >= 0.10),
        ("夏普", f'{stats["夏普"]:.2f}', "≥ 0.85", stats["夏普"] >= 0.85),
        ("最大回撤", percent(stats["最大回撤"]), "≥ −30%", stats["最大回撤"] >= -0.30),
        ("卡玛", f'{stats["卡玛"]:.2f}', "≥ 0.50", stats["卡玛"] >= 0.50),
        ("年化双边换手", percent(stats["年化换手"]), "≤ 700%", stats["年化换手"] <= 7.0),
        ("单边千 2 年化", percent(costs[0.002]["年化收益"]), "> 0%", costs[0.002]["年化收益"] > 0),
    ]
    conclusion = "达标" if all(item[-1] for item in primary) else "未达标"
    sections = [
        "# ETF 高收益策略研究",
        "## 1. 结论",
        f'**{conclusion}。** 主推候选 {winner} 用正动量和趋势筛选后按波动率倒数配置，'
        f'全样本扣费年化 {percent(cagr)}，最大回撤 {percent(stats["最大回撤"])}，'
        f'OOS 年化 {percent(oos[winner]["年化收益"])}。',
        table(["指标", "结果", "及格线", "判定"],
              ((*row[:3], verdict(row[3])) for row in primary)),
        f'两套锁定方案的最高全样本年化为 '
        f'{percent(max(runs[name][2]["年化收益"] for name in CENTERS))}。'
        f'现有动量同口径年化为 {percent(runs["现有 ETF 动量（250/前3）"][2]["年化收益"])}。'
        "主推的主要瓶颈是样本外收益、换手和随机对照表现。在当前数据与成本约束下，"
        "本次测试的 ETF 日线月度轮动方案未能稳定超过 13%。",
        "## 2. 策略逻辑",
        f"""- 数据为 AkShare `fund_etf_hist_em` 的东方财富 ETF 后复权日行情（`hfq`）。2015 年用于预热，主样本请求区间为 2016-01-01 至 2026-11-07，最后实际交易日为 {panel.index[-1]:%Y-%m-%d}。日收益以相邻两笔实际收盘价计算，停牌期间保留缺失，复牌时计入跨缺口收益。
- 两个候选共用附录的固定池。每个标的以 `收盘价(t) / 收盘价(t−L) − 1` 打分，回看窗口按面板交易日计。无当日或回看端点价格时不打分，上市前不补值。
- A 取 L={CENTERS["A"].lookback}。仅保留动量大于零、收盘价高于 200 日简单均线、60 日样本波动率为正的标的。按动量取前 {CENTERS["A"].top_n}，再以 `(1/波动率) / Σ(1/波动率)` 分配。黄金和债券使用相同门槛。若风险资产都落选，合格的黄金或债券可承接配置，无合格标的时持现金。
- B 取 L={CENTERS["B"].lookback}。池内可打分标的中，正动量比例达到 60% 时取前 {CENTERS["B"].top_n} 等权。低于 60% 时，70% 配给最多 5 只正动量非债券，30% 配给一只正动量债券，对应篮子为空时预算留在现金。每个篮子先留住仍在入选名次加 1 以内的原持仓，再按分数补足。
- 两套中心方案均在每月第一个实际交易日收盘调仓。邻域的双周方案取自 2015-01-05 起每 14 个自然日桶的首个交易日。调仓当天扣费，新权重从下一交易日起计收益，未分配部分计为零收益现金。
- 复现时先设置 `export no_proxy='*' NO_PROXY='*'`，再执行 `python run.py --strategy high`，会从缓存或同一数据源重算报告和两张 CSV。""",
        "## 3. 主对比表",
        "全样本统一使用现有回测引擎、同一行情口径和单边万 5。"
        "净值从 2016-01-01 的现金锚点 1.0 起算，年化双边换手包含首次建仓。"
        "现有动量、双均线复用原权重算法，因此与 README 中历史前复权数字存在口径差异。",
        table(
            ["策略", "年化", "波动", "夏普", "最大回撤", "卡玛", "年化双边换手"],
            ([label, *comparison(runs[key][2])] for label, key in (
                ("买入持有 510300", "买入持有 510300"), ("双均线 MA20/60", "双均线 MA20/60"),
                ("现有动量（250/前 3）", "现有 ETF 动量（250/前3）"),
                ("候选 A", "A"), ("候选 B", "B"), (f"主推（{winner}）", winner),
            )),
        ),
    ]

    centers = grid.loc[grid["center"]].set_index("candidate")
    median = grid.groupby("candidate")["sharpe"].median()
    ratio = oos[winner]["年化收益"] / centers.loc[winner, "cagr"]
    sections += [
        "## 4. IS vs OOS（V1）",
        f"参数在 IS（2016-01-01 至 2021-12-31）选定，锁定后 OOS 仅跑一次。"
        f"两套中心参数在跑 IS 前预设，按 IS 邻域夏普中位数选主推（A {median['A']:.3f}，"
        f"B {median['B']:.3f}，并列时看中心年化），不挑邻域收益最高的点。"
        f"主推为 {winner} 的 {params.lookback} 日 / 前 {params.top_n} / 月度。"
        "OOS 从 2022-01-01 起，继承 2021 年末仓位，沿用冻结策略的同一条净值路径。",
        table(
            ["候选", "IS 年化", "IS 夏普", "IS 最大回撤", "OOS 年化", "OOS 夏普", "OOS 最大回撤"],
            ([name, percent(centers.loc[name, "cagr"]), f'{centers.loc[name, "sharpe"]:.2f}',
              percent(centers.loc[name, "mdd"]), percent(oos[name]["年化收益"]),
              f'{oos[name]["夏普"]:.2f}', percent(oos[name]["最大回撤"])] for name in CENTERS),
        ),
        f"V1 {verdict(ratio >= 0.5 and oos[winner]['年化收益'] >= 0.10)}。"
        f"主推 OOS 年化为 IS 的 {percent(ratio)}，门槛为 IS 的 50% 且年化至少 10%。",
        "## 5. 参数稳健性矩阵（V2，仅 IS）",
    ]
    for name, center in CENTERS.items():
        neighborhood = grid.loc[grid["candidate"].eq(name)]
        above = neighborhood["cagr"].gt(0.10).sum()
        minimum = neighborhood["cagr"].min()
        cliff = minimum < centers.loc[name, "cagr"] / 2
        sections.extend([
            f"**候选 {name}**（中心 {center.lookback} 日 / 前 {center.top_n} / 月度）",
            grid_table(neighborhood),
            f"年化超过 10% 的组合 {above}/{len(neighborhood)}（{percent(above / len(neighborhood))}），"
            f"最差 {percent(minimum)}。V2 {verdict(above / len(neighborhood) >= 0.70 and not cliff)}。"
            "把年化跌破中心的一半视为悬崖式下降。",
        ])

    groups = study["market_months"]
    up = groups.loc[groups["market"].gt(0)]
    down = groups.loc[groups["market"].lt(0)]
    down_strategy, down_benchmark = down[["strategy", "benchmark"]].mean()
    contribution = study["largest_contribution"]
    v4 = (down_strategy >= -0.01 or down_strategy - down_benchmark >= 0.01) and contribution <= 0.60
    sections.extend([
        "## 6. 分年度与市场状态（V4）",
        table(["年份", "主推扣费收益"], ((year, percent(rate)) for year, rate in study["yearly"].items())),
        "按沪深 300 指数（000300，AkShare 东方财富日行情）当月价格收益的正负分组，零收益月不进两组。"
        "每月收益由当月每日净值变动复合，包含首末不完整月份。",
        table(["沪深 300 月份", "月份数", "主推平均月收益", "买入持有平均月收益"],
              ([label, len(rows), percent(rows["strategy"].mean()), percent(rows["benchmark"].mean())]
               for label, rows in (("上涨", up), ("下跌", down)))),
        f"V4 {verdict(v4)}。下跌月主推相对基准改善 {100 * (down_strategy - down_benchmark):.2f} 个百分点，"
        f"以至少 1 个百分点为明显改善。{study['largest_year']} 年贡献最大，"
        f"占全样本累计收益 {percent(contribution)}，按当年净值增量除以全样本净值增量计算。",
        "## 7. 成本敏感性（V3）",
        table(["单边成本", "年化", "最大回撤", "夏普"],
              ([percent(cost), percent(row["年化收益"]), percent(row["最大回撤"]), f'{row["夏普"]:.2f}']
               for cost, row in costs.items())),
        f"V3 {verdict(costs[0.002]['年化收益'] > 0)}。三档使用相同目标权重。",
    ])

    random_cagr = study["random_cagr"]
    quantiles = (0, 0.05, 0.25, 0.50, 0.75, 0.95, 1)
    values = np.quantile(random_cagr, quantiles)
    p = (1 + (random_cagr >= cagr).sum()) / (1 + len(random_cagr))
    percentile = (random_cagr < cagr).mean()
    sections.extend([
        "## 8. 同换手随机对照（V6）",
        f"用固定种子 42，从符合历史数据可用性约束的标的排列中等概率抽取 {len(random_cagr)} 次。"
        "每次把整条权重轨迹重新分配给同池标的。这样每次调仓的持仓数、现金比例、单标的权重大小及"
        "双边换手均与主推完全相同，保留原持仓持续性。按相同成本跑现有引擎。"
        "这是以原策略持仓形状为条件的随机检验。",
        table(["分位", *(f"{q:.0%}" for q in quantiles)],
              [["随机年化", *(percent(value) for value in values)]]),
        f"V6 {verdict(cagr > values[-2] and p < 0.05)}。主推年化 {percent(cagr)}，"
        f"严格超过 {percent(percentile)} 的随机结果，单尾 p = {p:.4f}"
        "（等于或高于主推的次数加 1，再除以总次数加 1）。",
    ])

    holdings = positions.iloc[1:]
    averages = holdings.mean()
    latest = weights.iloc[-1]
    name_map = candidate_b.ETF_UNIVERSE
    trimmed = study["trimmed_cagr"]
    best_months = "、".join(f"{month}（{percent(value)}）" for month, value in study["best_months"].items())
    oos_holdings = holdings.loc["2022-01-01":]
    sections.extend([
        "## 9. 持仓、换手与尾部（V5）",
        f"- 按交易日收盘后的目标权重统计，平均持仓 {holdings.gt(0).sum(axis=1).mean():.2f} 只。"
        f"最大单标的平均权重为 {name_map[averages.idxmax()][0]}（{averages.idxmax()}）"
        f"的 {percent(averages.max())}。年化双边换手为 {percent(stats['年化换手'])}。\n"
        f"- 最近一期实际调仓日 {weights.index[-1]:%Y-%m-%d}，持仓为 "
        + "、".join(f"{name_map[code][0]} {percent(weight)}" for code, weight in latest.items() if weight > 0)
        + f"，现金 {percent(max(0, 1 - latest.sum()))}。\n"
        f"- 剔除净收益最好的三个月 {best_months} 后，按剩余月份数除以 12 年化，"
        f"得到 {percent(trimmed)}。V5 {verdict(trimmed > 0.05)}。",
        "## 10. 标的池附录",
        "起止日为本次行情窗口内的首末非空收盘日期。起点晚于 2016-01-01 的标的保留上市前缺失，"
        "只有满足信号历史长度后才可入选。",
        table(["代码", "名称", "类别", "数据起始日", "数据末日"],
              ([code, name, group, f"{panel[code].first_valid_index():%Y-%m-%d}",
                f"{panel[code].last_valid_index():%Y-%m-%d}"] for code, (name, group) in name_map.items())),
        "## 11. 已知局限",
        f"""- 跨境 ETF 的场内价格包含 QDII 溢价。回测没有对极端溢价、申赎限制或成交冲击单独建模。OOS 纳指 ETF 平均权重为 {percent(oos_holdings["513100"].mean())}，标普 500 ETF 为 {percent(oos_holdings["513500"].mean())}。
- 池子按资产覆盖预先设定，晚上市标的保留了时间门槛，但没有纳入历史退市基金，仍存在池子选择与存续偏差。
- 引擎按固定目标权重计算两次正式调仓间的日收益，等价于每日恢复目标权重，未计这些隐含调整的交易成本。
- 后复权使用已公布的除权除息调整；本研究缓存的是当前供应商行情，并非逐日存档的当时数据快照。供应商补录和舍入修订可能影响独立重新下载的结果。
- 随机对照保留了策略的现金时间、持仓持续性和换手，未额外约束资产类别暴露。因此该检验反映固定池内的条件随机排名。""",
    ])
    return "\n\n".join(sections) + "\n"


def write_study(study, root=ROOT):
    reports = root / "reports"
    results = root / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "high_return.md").write_text(render(study), encoding="utf-8")
    nav = study["runs"][high_return.SELECTED_CANDIDATE][0]
    nav.to_csv(results / "high_return_nav.csv", index_label="date")
    study["weights"].to_csv(results / "high_return_weights.csv", index_label="date")


def run_study():
    study = analyze()
    write_study(study)
    return study["costs"][PRIMARY_COST]
