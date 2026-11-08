"""Render the study without substituting estimates for failed validation."""

from pathlib import Path

import numpy as np

from research.lowvol_momentum_study import analyze
from research.report import table

ROOT = Path(__file__).resolve().parents[1]
COST_DECLARATION = (
    "成本口径按单边万 5：`cost_rate` 作用于换手总额 `Σ|Δw|`，"
    "买卖两腿各收一次，全仓换一次 = 千 1。"
)


def pct(value):
    return f"{value:.2%}" if np.isfinite(value) else "未定义"


def outcome(passed):
    return "通过" if passed else "未通过"


def performance(stats):
    return [pct(stats["年化收益"]), f'{stats["夏普"]:.2f}', pct(stats["最大回撤"])]


def render(study):
    main, stats = study["main"], study["costs"][0.0005]
    engine = study["engine_stats"]
    is_stats, oos, engine_oos = study["is"], study["oos"], study["engine_oos"]
    capacity = study["capacity"]
    annual = min(stats["年化收益"], engine["年化收益"])
    outside = min(oos["年化收益"], engine_oos["年化收益"])
    sharpe = min(stats["夏普"], engine["夏普"])
    mdd = min(stats["最大回撤"], engine["最大回撤"])
    calmar = min(stats["卡玛"], engine["卡玛"])
    average = main.weights.mean().max() if len(main.weights.columns) else 0.0
    held = main.weights.gt(0).sum(axis=1)
    peak_day = main.weights.max(axis=1).idxmax() if len(main.weights.columns) else None
    peak_code = main.weights.loc[peak_day].idxmax() if peak_day is not None else None
    years = (main.nav.index[-1] - main.nav.index[0]).days / 365.25
    gross_log = study["attribution"].drop("交易摩擦").sum()
    gross_annual = np.expm1(gross_log / years)
    costs = study["costs"]
    stress_cagr = costs[0.002]["年化收益"]
    matched, draws = study["matched"], study["controls"]
    matched_ready = len(matched) == 200
    quantile = np.quantile(matched, 0.95) if len(matched) else np.nan
    checks = [
        ("全样本年化", pct(annual), "≥18%", annual >= 0.18),
        ("OOS 年化", pct(outside), "≥12%", outside >= 0.12),
        ("夏普", f"{sharpe:.2f}", "≥0.85", sharpe >= 0.85),
        ("最大回撤", pct(mdd), "≥−40%", mdd >= -0.40),
        ("卡玛", f"{calmar:.2f}", "≥0.50", calmar >= 0.50),
        ("年化双边换手", pct(stats["年化换手"]), "≤1200%", stats["年化换手"] <= 12),
        ("千 2 净年化", pct(stress_cagr), "≥10%", stress_cagr >= 0.10),
        ("单票平均权重最大值", pct(average), "≤8%", average <= 0.08),
        ("任意日单票权重最大值", pct(main.max_weight), "≤8%", main.max_weight <= 0.08 + 1e-9),
    ]
    v1 = outside >= 0.12 and outside >= is_stats["年化收益"] / 2
    v4 = np.isfinite(study["contribution"]) and study["contribution"] <= 0.60
    v6 = matched_ready and stats["年化收益"] > quantile
    complete = (
        all(row[-1] for row in checks) and v1 and study["stable"] and v4 and study["trimmed"] >= 0.10
        and v6 and study["underwater"] <= 365.25 * 3 and main.unresolved_exposure == 0
        and np.isfinite(study["style_share"]) and len(capacity) > 0
        and not (annual > 0.35)
    )
    achieved = complete and 0.20 <= annual <= 0.30
    lock = study["lock"]
    grid = study["grid"].pivot(index="lookback", columns="top_n", values="cagr")
    sections = [
        "# 沪深主板低波动动量",
        "## 1. 结论",
        f'**{"达标" if achieved else "未达标"}。** 逐笔账本扣费年化 {pct(stats["年化收益"])}，'
        f'OOS 年化 {pct(oos["年化收益"])}，最大回撤 {pct(stats["最大回撤"])}。'
        "验收对引擎和逐笔账本取较保守的口径。该结果仅适用于预登记方案及其邻域。",
        table(["指标", "结果", "门槛", "判定"],
              ([*row[:3], outcome(row[3])] for row in checks)),
        f"这轮研究中唯一锁定的全样本方案净年化为 {pct(stats['年化收益'])}。"
        f"保持成交路径不变，加回已计摩擦后的归因年化为 {pct(gross_annual)}，"
        "说明收益缺口主要来自信号与风险暴露，成本进一步侵蚀结果。"
        "这只能界定本轮方案，无法估计 A 股策略总体的收益上限。",
        "## 2. 策略逻辑",
        f"沪深主板历史池含在市、退市和摘牌。原中小板股票自交易所合并日纳入。"
        f"使用过去 20 日成交额均值，至少 2,000 万元且位于当日基础池的 30–80 分位。"
        f"上市至少 250 个交易日，历史风险警示和停牌状态须可核验。跳过近 20 日的 "
        f"{lock['lookback']} 日累计收益与 60 日低波动排名等权，只买正动量。"
        f"月末生成指令，最多 {lock['top_n']} 只，目标单票为 1/{lock['top_n']}。"
        "从下一交易日起按可成交收盘模拟，方向性触停、停牌或状态不明不成交。"
        "买单三日取消，卖单顺延。每笔受现金、整手及日成交额约束。"
        " [深交所两板合并](https://www.szse.cn/aboutus/trends/news/t20210406_585442.html)"
        " · [上交所交易规则](https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml)"
        " · [深交所风险警示规则](https://www.szse.cn/lawrules/service/member/t20260630_621404.html)",
        "股息和送转按已公告的股权登记日、除权日和到账日记账。"
        "停牌旧仓仅在账本以最后可核验价格计估值，报价、信号和撮合保持缺失。"
        "退市按最后可核验收盘价转入不可再投资的清算价值，并另做零回收压力。",
        "## 3. 收益来源",
        "预登记假设为结构性错误定价。低波动和正动量筛选是否带来可交易的选股增量，"
        "由同池随机对照和风格归因检验。流动性分层衡量的是交易容量与流动性暴露。",
        "## 4. 主对比表",
        "2015-01-01 至 2026-11-07。基准复用原引擎与原策略权重，使用同区间后复权行情。"
        "既有 ETF 动量先前在不同区间及成本口径下为 12.23%。"
        "股票引擎复核使用实际收盘权重，权重差包含被动漂移。真实成交换手与法定税费以逐笔账本为准。",
        table(["策略", "年化", "夏普", "最大回撤"], (
            [name, *performance(row)] for name, row in (
                *study["baselines"].items(), ("低波动动量（逐笔账本）", stats),
                ("低波动动量（现有引擎复核）", engine),
            )
        )),
        "## 5. IS vs OOS（V1）",
        "仅在 2015–2020 年选参。参数冻结后，OOS 从 2021-01 起沿用同一条持仓路径。"
        "成本与零回收压力使用冻结规则。",
        table(["区间", "年化", "夏普", "最大回撤"], (
            [name, *performance(row)] for name, row in (
                ("IS 逐笔", is_stats), ("OOS 逐笔", oos), ("OOS 引擎复核", engine_oos),
            )
        )),
        f"V1 {outcome(v1)}。较保守的 OOS 年化须达到 IS 的一半且至少 12%。",
        "## 6. 参数邻域矩阵（V2，仅 IS）",
        table(["窗口 / 持仓数", *grid.columns],
              ([window, *(pct(value) for value in row)] for window, row in grid.iterrows())),
        f"净年化超过 12% 的组合占 {pct(study['neighborhood_rate'])}。"
        f"IS 最佳单点年化为 {pct(grid.to_numpy().max())}。"
        f"V2 {outcome(study['stable'])}。"
        + ("选用合格邻域中心。" if study["stable"] else "按提案保留预设参数作为失败对照。"),
        "## 7. 分年度收益（V4）",
        table(["年份", "账本净收益"], ((year, pct(value)) for year, value in study["yearly"].items())),
        f"最大贡献年为 {study['largest_year']}，占全样本净值增量 {pct(study['contribution'])}。"
        f"V4 {outcome(v4)}。",
        "## 8. 成本敏感性与剔尾（V3、V5）",
        COST_DECLARATION,
        "双边换手率统计买腿和卖腿的成交额。三档均另计单边 10 bp 滑点，"
        "逐笔账本计佣金最低收费、卖出印花税与过户费。2015 年统一前使用保守过户费估算。"
        "股息税按 20% 保守预留。"
        " [印花税调整](https://www.mof.gov.cn/jrttts/202308/t20230828_3904235.htm)"
        " · [股息红利税](https://shanghai.chinatax.gov.cn/zcfw/zcfgk/grsds/201509/t418999.html)",
        table(["单边佣金", "账本净年化", "夏普", "最大回撤"],
              ([pct(rate), *performance(row)] for rate, row in costs.items())),
        f"V3 {outcome(stress_cagr >= 0.10)}。单边 20 bp 滑点压力下净年化 "
        f"{pct(study['stress_slip']['年化收益'])}。",
        f"最佳五日为 {'、'.join(f'{day:%Y-%m-%d}（{pct(value)}）' for day, value in study['best_days'].items())}。"
        f"剔除后净年化为 {pct(study['trimmed'])}。V5 {outcome(study['trimmed'] >= 0.10)}。",
        "## 9. 随机对照（V6）",
        f"固定种子 42，使用同日可选池、目标持仓数及逐笔成交模型模拟 {len(draws)} 次，"
        f"逐日实际换手在 1 bp 容差内的有 {len(matched)} 次。"
        f"V6 {'通过' if v6 else '未通过' if matched_ready else '未验证'}。",
        f"同换手合格样本的第 95 分位为 {pct(quantile)}。"
        + (f"200 次合格对照的单尾 p 为 {(1 + (matched >= stats['年化收益']).sum()) / 201:.4f}。"
           if matched_ready else "未达到预登记样本量，无法据此判断同换手排名。"),
        table(["分位", "5%", "50%", "95%"],
              [["全部随机试验（描述性）", *(pct(v) for v in np.quantile(draws, [0.05, 0.5, 0.95]))]]),
        "## 10. 收益归因（V8）",
        "市场因子用沪深 300 指数。风格因子为低波、流动性和动量的月末分层价差。"
        "系数仅在 IS 估计，四项按累计对数收益归一，以便相加。",
        table(["来源", "累计对数收益贡献"],
              ((name, pct(value)) for name, value in study["attribution"].items())),
        f"风格贡献占总净收益 {pct(study['style_share'])}。"
        + ("风格收益被负的选股残差和交易摩擦抵消，风格占比可超过 100%。"
           "归类为风格策略。" if study["style_share"] > 0.70 else "选股残差需与 V6 一并解读。"),
        "## 11. 容量（V7）",
        f"以 500 万元初始账本逐笔估算，按 5% × 前 20 日均成交额 ÷ 计划订单占净值比例。"
        f"成交订单的严格瓶颈为 {capacity.min():,.0f} 元，日瓶颈第 10 分位为 {capacity.quantile(.10):,.0f} 元。"
        + ("个人可跑、机构不可用。" if capacity.min() < 10_000_000 else ""),
        "## 12. 失效条件",
        f"最长水下期为 {study['underwater']:.0f} 个自然日。"
        "高波动股票快速反弹、跌停无法退出和成交额萎缩会分别削弱排名、退出与容量。"
        f"历史上超过 10% 的独立回撤有 {len(study['drawdown_episodes'])} 次。"
        "下表的指数收益覆盖策略峰值至谷底，可区分指数下跌和策略独自回撤。",
        table(["峰值", "谷底", "收复前高", "策略回撤", "同期沪深 300"],
              ([f"{start:%Y-%m-%d}", f"{bottom:%Y-%m-%d}",
                f"{recovery:%Y-%m-%d}" if not np.isnat(recovery.to_datetime64()) else "期末未收复",
                pct(decline), pct(market)]
               for start, bottom, recovery, decline, market in study["drawdown_episodes"])),
        f"按退市零回收且未知公司行动持仓当日零估值，净年化为 "
        f"{pct(study['stress_recovery']['年化收益'])}。账本持仓遇到未解决行情或公司行动冲突 "
        f"{main.unresolved_exposure} 个证券日。",
        f"实际同时持有最多 {held.max()} 只，未退出的旧仓可与新目标重叠。"
        + (f"{peak_day:%Y-%m-%d} 的 {peak_code} 权重达到 {pct(main.max_weight)}，"
           "买入时的单票限制无法约束后续价格漂移，实盘式单票硬上限未通过。"
           if peak_day is not None and main.max_weight > 0.08 + 1e-9 else ""),
        "以放宽回撤或使用杠杆追求目标，会突破已设风险约束。放宽流动性筛选可能进一步压缩容量。"
        "本轮没有证据表明放宽这些约束能产生稳定的选股收益。",
        "## 13. 局限与数据审计",
    ]
    audit = study["audit"].sum(numeric_only=True)
    exits = study["audit"]["exit_intervals"].gt(0).sum()
    exit_gaps = study["audit"].loc[study["audit"]["unverified_exits"].gt(0), "code"]
    survivor_delta = study["is_survivors"]["年化收益"] - is_stats["年化收益"]
    sections += [
        f"- 全样本合并 {len(study['audit']):,} 个主板代码，其中 {exits} 个出现在退市名录。"
        f"两源冲突或状态缺失分别为 {audit['close_disagreements']:,} 和 {audit['missing_status']:,} 个证券日，"
        f"未解公司行动 {audit['unresolved_actions']:,} 个证券日。"
        f"日线和状态文件均齐全，{audit['missing_actions_file']:,} 个代码缺少供应商公司行动文件，"
        f"{audit['unverified_exits']:,} 次退出的末价未能交叉核实，按零价值处理。"
        + (f"涉及代码 {'、'.join(exit_gaps)}。" if len(exit_gaps) else "")
        + f"仅存续池在 IS 的净年化差为 {survivor_delta * 100:+.2f} 个百分点。"
        "该差值针对本策略和本区间。",
        "- 日线未提供收盘竞价队列与盘口深度。即使未触停且单量低于成交额上限，成交仍属于日线估算。",
        "- 退市末价不是到账凭证。零回收压力单独报告该假设对绩效的影响。",
        "- 当前归档数据可受补录与修订影响。双源核对与独立账本仍无法替代逐日交易所快照。",
        "交易所名录的分年覆盖如下。数量为当年与主板样本重叠的证券记录数。",
        table(["年份", "覆盖记录", "终止上市记录"],
              ([row["year"], row["registry_intervals"], row["exits"]]
               for row in study["scan"].get("years", []))),
        "### 复现",
        "```bash\nexport no_proxy='*' NO_PROXY='*'\n"
        "python -m research.lowvol_momentum_scan\npython -m research.lowvol_momentum_fetch --resume\n"
        "python run.py --strategy lowvol_momentum\n```",
    ]
    return "\n\n".join(sections) + "\n"


def write_study(study, root=ROOT):
    reports = root / "reports"
    results = root / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "lowvol_momentum.md").write_text(render(study), encoding="utf-8")
    main = study["main"]
    main.nav.to_csv(results / "lowvol_momentum_nav.csv", index_label="date")
    main.weights.to_csv(results / "lowvol_momentum_weights.csv", index_label="date", float_format="%.10g")


def run_study(cache):
    study = analyze(cache)
    write_study(study)
    return study["costs"][0.0005]
