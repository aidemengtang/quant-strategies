"""策略一：双均线趋势择时（基线对照组）。

逻辑：收盘价 MA(fast) 在 MA(slow) 之上则满仓，否则空仓（现金）。
标的：510300 沪深300ETF —— 用宽基 ETF 而非指数，避免"指数不能交易"的失真。
"""

from __future__ import annotations

import pandas as pd

from qs.data import load_etf


def build_weights(symbol: str = "510300", fast: int = 20, slow: int = 60,
                  start: str = "2015-01-01", end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = load_etf(symbol, start=start, end=end)
    close = px["close"]
    rets = close.pct_change(fill_method=None).to_frame(name=symbol)

    ma_f = close.rolling(fast, min_periods=fast).mean()
    ma_s = close.rolling(slow, min_periods=slow).mean()
    signal = (ma_f > ma_s).astype(float)
    signal[ma_s.isna()] = 0.0                     # 均线未成形前空仓

    weights = signal.to_frame(name=symbol)        # 目标权重：1=满仓，0=现金
    return rets.dropna(how="all"), weights
