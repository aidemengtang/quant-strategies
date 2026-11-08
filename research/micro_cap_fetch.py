"""Resume the small gap after the local market archive, by date and exchange."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from pathlib import Path
import socket

import pandas as pd

from qs.micro_cap_data import market_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = Path("/Users/mac/PycharmProjects/tushare_data")
DEFAULT_CACHE = ROOT / "data" / "micro_cap"
DEFAULT_SCAN = Path("/Users/mac/Documents/Codex/2026-11-08/github/work/micro-cap-scan")
END = "2026-11-06"
CALENDAR = ROOT / "data" / "lowvol_momentum" / "calendar.parquet"
FIELDS = ["ts_code", "close", "pre_close", "high", "low", "pct_chg", "amount", "turnover_rate", "risk"]
_connected = False


def write(frame, path):
    temp = path.with_suffix(".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def bao_day(job):
    import baostock as bs

    global _connected
    day, path = job
    socket.setdefaulttimeout(25)
    for _ in range(2):
        try:
            if not _connected:
                if bs.login().error_code != "0":
                    raise ConnectionError("BaoStock login failed")
                _connected = True
            result = bs.query_daily_history_k_AStock(day)
            rows = []
            while result.error_code == "0" and result.next():
                rows.append(result.get_row_data())
            if result.error_code != "0" or len(rows) < 5000:
                raise ValueError(f"BaoStock {result.error_msg}, {len(rows)} records")
            frame = pd.DataFrame(rows, columns=result.fields)
            if not frame["date"].eq(day).all():
                raise ValueError("incorrect session")
            frame["ts_code"] = frame["code"].str[3:] + "." + frame["code"].str[:2].str.upper()
            frame = frame.rename(columns={"preclose": "pre_close", "pctChg": "pct_chg",
                                          "turn": "turnover_rate", "isST": "risk"})
            frame = frame.loc[frame["tradestatus"].eq("1"), FIELDS]
            write(frame, path)
            return day, ""
        except (OSError, ValueError) as exc:
            _connected = False
            error = str(exc)
    return day, error


def bj_stock(job):
    import akshare as ak

    code, first, last, path, previous = job
    if previous.is_file() and first >= "20261012":
        frame = pd.read_parquet(previous)
    else:
        for _ in range(2):
            try:
                frame = ak.stock_zh_a_hist(code, start_date=first, end_date=last, adjust="", timeout=12)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        else:
            return code, error
    if frame.empty:
        # An empty result is unresolved and is never used as a zero-activity record.
        return code, "empty response"
    frame = frame.rename(columns={"日期": "date", "收盘": "close", "最高": "high", "最低": "low",
                                  "涨跌幅": "pct_chg", "成交额": "amount", "换手率": "turnover_rate"})
    frame["pre_close"] = frame["close"] - frame["涨跌额"]
    frame["ts_code"] = code + ".BJ"
    frame["risk"] = pd.NA
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame = frame.loc[frame.date.between(pd.Timestamp(first).strftime("%Y-%m-%d"),
                                        pd.Timestamp(last).strftime("%Y-%m-%d")), ["date", *FIELDS]]
    if frame.empty:
        return code, "no record in the terminal window"
    write(frame, path)
    return code, ""


def sessions(archive, calendar=CALENDAR):
    dates = pd.DatetimeIndex(pd.read_parquet(calendar)["date"])
    return pd.DatetimeIndex([
        day for day in dates[(dates >= "2014-11-01") & (dates <= END)]
        if not ((market_file(archive, "daily", day) or market_file(archive, "stk_factor", day))
                and market_file(archive, "daily_basic", day))
    ])


def fetch(archive, cache, scan=DEFAULT_SCAN, *, resume=False):
    import akshare as ak

    os.environ.update(no_proxy="*", NO_PROXY="*")
    dates = sessions(archive)
    if dates.empty:
        return
    day_dir, bj_dir = cache / "shsz", cache / "bj"
    day_dir.mkdir(parents=True, exist_ok=True)
    bj_dir.mkdir(parents=True, exist_ok=True)
    if (scan / "bj-live.parquet").is_file():
        listings = pd.read_parquet(scan / "bj-live.parquet")
    else:
        listings = ak.stock_info_bj_name_code()
    write(listings, cache / "bj_listings.parquet")
    jobs = [(day, day_dir / f"{day}.parquet") for day in dates.strftime("%Y-%m-%d")]
    if resume:
        jobs = [(day, path) for day, path in jobs if not path.exists()]
    errors = []
    print(f"沪深补齐 {len(jobs)} 个交易日", flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        for number, (item, error) in enumerate(pool.map(bao_day, jobs), 1):
            if error:
                errors.append(("沪深", item, error))
            print(f"沪深 {number}/{len(jobs)}", flush=True)
    jobs = []
    listing_days = pd.to_datetime(listings["上市日期"]).groupby(
        listings["证券代码"].astype(str).str.zfill(6)).min()
    scan_start = pd.Timestamp("2026-10-12")
    for code, listed in listing_days.items():
        path = bj_dir / f"{code}.parquet"
        if resume and path.is_file():
            cached = pd.read_parquet(path, columns=["date"]).date
            needed_from = max(dates[0], listed)
            has_early_window = needed_from >= scan_start or cached.lt(scan_start.strftime("%Y-%m-%d")).any()
            if has_early_window:
                continue
        jobs.append((code, dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d"),
                     path, scan / "bj-window" / f"{code}.parquet"))
    print(f"京市检查 {len(jobs)} 个短窗口，优先读现有样本", flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        for number, (item, error) in enumerate(pool.map(bj_stock, jobs), 1):
            if error:
                errors.append(("京市", item, error))
            if number % 25 == 0 or number == len(jobs):
                print(f"京市 {number}/{len(jobs)}", flush=True)
    pd.DataFrame(errors, columns=["market", "item", "error"]).to_csv(cache / "fetch_failures.csv", index=False)
    if errors:
        print(f"{len(errors)} 项缺口写入 fetch_failures.csv", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--scan-cache", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    fetch(args.archive, args.cache, args.scan_cache, resume=args.resume)
