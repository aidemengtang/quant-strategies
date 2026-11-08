# 夜间执行记录

起点：`eee95a80314eef150898122b4faa3f077624a088`，工作区原本干净。

## 模板与环境核对

- 已阅读 `research/micro_cap_report.py`、`research/micro_cap_study.py` 的容量审计与三因子归因、`qs/micro_cap_execution.py` 以及 `tests/test_micro_cap.py`，并核对 `reports/micro_cap.md` 的披露格式。
- 基线执行 `uv run --with pytest --with numpy --with pandas python -m pytest tests/ -q`：55 passed。
- 系统 `python` 缺少 pytest 和 numpy，所有 pytest 均经上述 uv 命令执行。
- uv 实测 numpy 2.5.4、pandas 3.1.1，符合现有 `numpy>=1.24`、`pandas>=2.0`；没有修改 requirements.txt 或任何锁定版本。报告生成另通过 uv 加载 pyarrow 26.0.1，符合 `pyarrow>=14.0`。

## high_return

- 新增 `qs/capacity.py`：只读日瓶颈快照，沿用 micro_cap 的 5% 参与率、最低 ADV 25 只持仓范围及真实零/不可得分类。
- 新增 `research/capacity_report.py`：直接复用 micro_cap 的 `capacity_audit`；三因子计算逐式对齐其市场/市值/20 日动量公式，保持原分位数、缺失值与股票截面，不添加策略筛选。提供共用报告格式、载荷、原始因子累计和、净值核对与结果导出。
- 新增 `research/high_return_capacity.py`：重放冻结候选 A，从 ETF parquet 缓存直接读取人民币成交额，逐日计算滞后 20 日 ADV，观察引擎实际权重和正式调仓订单；排除非交易日现金锚点的容量计数。报告明确原引擎未记录的隐含日调权及跨资产三因子的解释边界。
- 更新 `tests/test_high_return.py`，增加 4 项：三类容量计数与 2019 样本、三因子与 micro_cap 数值一致及载荷符号、IS 冻结边界、CSV 净值核对。
- 生成 `reports/high_return_capacity.md`，以及被原 .gitignore 忽略的 `results/high_return_capacity_diag.csv`、`results/high_return_capacity_daily.csv`、`results/high_return_factors.csv`。不覆盖原净值和权重 CSV。
- 归档净值重放逐日通过，容差 rtol/atol 均为 1e-12。全段 11 个持仓 ADV 不可得日，真实零与空仓均为 0。

## 遇到的问题与修正

- ETF 的公共 `load_etf` 返回值不暴露 amount，但已有 parquet 保留 amount，因此直接读取缓存成交额，没有用成交量代替，也未改变数据加载器接口。
- pandas 的 CSV 日期解析默认微秒单位与行情索引的纳秒单位不同，且索引名可能从 CSV 继承。核对函数仅统一日期单位和索引名后严格比较日期与净值；新增测试确认实际净值变化仍会报错。
- 容量敏感性测试最初漏计一日真实零订单，修正期望值：真实零应保留在有效样本，而 ADV 不可得才剔除。

## 验证进度

- 两个策略修改后的完整测试已通过：62 passed（high_return 新增 4 项，lowvol_momentum 新增 3 项）。
- high_return 与共享诊断支持作为第一个独立提交；lowvol_momentum 的变更与最终记录随后作为第二个独立提交。
