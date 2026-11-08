"""行情数据获取与本地缓存（数据源：akshare，免 token）。"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(os.environ.get("QS_CACHE_DIR", Path(__file__).resolve().parent.parent / "data"))

_COLMAP = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
           "最低": "low", "成交量": "volume", "成交额": "amount"}


def load_etf(symbol: str, start: str = "2015-01-01", end: str | None = None,
             adjust: str = "qfq", use_cache: bool = True) -> pd.DataFrame:
    """读取单只 ETF 日线（前复权），返回 index=date 的 OHLCV 表。

    symbol 传 6 位代码，例如 510300（沪深300ETF）。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    start_c = start.replace("-", "")
    end_c = (end or pd.Timestamp.today().strftime("%Y%m%d")).replace("-", "")
    cache = CACHE_DIR / f"{symbol}_{start_c}_{end_c}_{adjust}.parquet"

    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        import akshare as ak

        raw = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_c,
                                  end_date=end_c, adjust=adjust)
        df = raw.rename(columns=_COLMAP)[list(_COLMAP.values())]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df.to_parquet(cache)

    df = df.loc[pd.Timestamp(start):pd.Timestamp(end)] if end else df.loc[pd.Timestamp(start):]
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def load_panel(symbols: list[str], start: str = "2015-01-01",
               end: str | None = None) -> pd.DataFrame:
    """读取多只标的收盘价，返回宽表 index=date, columns=symbol。"""
    frames = {}
    for s in symbols:
        try:
            frames[s] = load_etf(s, start=start, end=end)["close"]
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {s} 读取失败：{type(exc).__name__} {exc}")
    if not frames:
        raise RuntimeError("全部标的数据读取失败")
    panel = pd.DataFrame(frames)
    return panel.sort_index()


ETF_POOL = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
    "518880": "黄金ETF",
    "511010": "国债ETF",
    "513100": "纳指ETF",
}
