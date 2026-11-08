"""Download resumable, separate caches for prices, trade status and distributions."""

import argparse
import json
import os
import socket
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import sleep

import pandas as pd

from research.lowvol_momentum_scan import DEFAULT_CACHE, END, WARMUP_START

START_KEY = WARMUP_START.replace("-", "")
END_KEY = END.replace("-", "")
STATUS_FIELDS = "date,close,preclose,tradestatus,isST"
COMPONENTS = ("raw", "status", "actions")
_status_connected = False


def atomic_parquet(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def raw_prices(code):
    import akshare as ak

    raw = ak.stock_zh_a_hist(code, start_date=START_KEY, end_date=END_KEY, adjust="", timeout=20)
    if raw.empty:
        raise ValueError("AkShare returned no historical prices")
    fields = {
        "日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
        "成交量": "volume", "成交额": "amount", "涨跌额": "change",
    }
    frame = raw[list(fields)].rename(columns=fields)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date")


def status_prices(exchange, code):
    import baostock as bs

    global _status_connected
    if not _status_connected:
        login = bs.login()
        if login.error_code != "0":
            raise ConnectionError(f"BaoStock login: {login.error_msg}")
        _status_connected = True
    result = bs.query_history_k_data_plus(
        f"{exchange}.{code}", STATUS_FIELDS, WARMUP_START, END, "d", "3",
    )
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    frame = pd.DataFrame(rows, columns=STATUS_FIELDS.split(","))
    if result.error_code != "0":
        _status_connected = False
    if result.error_code != "0" or frame.empty:
        raise ValueError(f"BaoStock status {result.error_code}: {result.error_msg}; rows={len(frame)}")
    frame["date"] = pd.to_datetime(frame["date"])
    numeric = frame.columns.difference(["date"])
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame


def corporate_actions(code):
    import akshare as ak

    raw = ak.stock_dividend_cninfo(code)
    fields = {
        "实施方案公告日期": "announced", "股权登记日": "record", "除权日": "date",
        "派息日": "pay", "股份到账日": "share_pay",
        "送股比例": "bonus", "转增比例": "transfer", "派息比例": "cash",
    }
    frame = raw[list(fields)].rename(columns=fields)
    for field in ("announced", "record", "date", "pay", "share_pay"):
        frame[field] = pd.to_datetime(frame[field])
    return frame.loc[frame["date"].between(WARMUP_START, END)].sort_values("date")


def fetch_one(job):
    exchange, code, root, resume, components = job
    failures = {}
    for component, download in (
        ("raw", lambda: raw_prices(code)),
        ("status", lambda: status_prices(exchange, code)),
        ("actions", lambda: corporate_actions(code)),
    ):
        if component not in components:
            continue
        target = root / component / f"{code}.parquet"
        if resume and target.exists():
            continue
        for attempt in range(2):
            try:
                atomic_parquet(download(), target)
                break
            except KeyError as exc:
                failures[component] = f"provider schema or empty response: {exc}"
                break
            except Exception as exc:
                if attempt == 1:
                    failures[component] = f"{type(exc).__name__}: {exc}"
                else:
                    sleep(0.5)
    return code, failures


def worker_setup():
    import requests
    from baostock.common import contants

    base_request = requests.Session.request

    def request_with_timeout(session, *args, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = 20
        return base_request(session, *args, **kwargs)

    socket.setdefaulttimeout(25)
    requests.Session.request = request_with_timeout
    # The pinned provider accepts the complete study window in a 4096-row request.
    contants.BAOSTOCK_PER_PAGE_COUNT = 4096


def fetch(root, *, resume=False, workers=4, limit=None, components=COMPONENTS):
    records = pd.read_parquet(root / "listings.parquet")
    # Probe every in-scope exit first, then download the remaining listings.
    codes = records.sort_values(["exit_registry", "code"], ascending=[False, True])
    codes = codes.drop_duplicates(["exchange", "code"])
    for component in components:
        (root / component).mkdir(exist_ok=True)
    suffix = "" if tuple(components) == COMPONENTS else "_" + "_".join(components)
    ledger = root / f"fetch{suffix}_progress.jsonl"
    if not resume:
        ledger.write_text("")
    if limit is not None:
        codes = codes.head(limit)
    jobs = [(row.exchange, row.code, root, resume, components) for row in codes.itertuples()]
    failures = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_setup) as pool, ledger.open("a") as stream:
        for number, (code, errors) in enumerate(pool.map(fetch_one, jobs), 1):
            stream.write(json.dumps({"code": code, "errors": errors}, ensure_ascii=False) + "\n")
            stream.flush()
            if errors:
                failures[code] = errors
            if number % 25 == 0 or number == len(jobs):
                print(f"{number}/{len(jobs)}  complete; codes with source gaps: {len(failures)}", flush=True)
    (root / f"fetch{suffix}_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n"
    )
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=COMPONENTS)
    parser.add_argument("--limit", type=int, help="Audit only this many unique codes")
    args = parser.parse_args()
    os.environ.update(no_proxy="*", NO_PROXY="*")
    if not 1 <= args.workers <= 8 or (args.limit is not None and args.limit < 1):
        parser.error("workers must be 1–8 and limit must be positive")
    fetch(args.cache, resume=args.resume, workers=args.workers, limit=args.limit,
          components=args.components)


if __name__ == "__main__":
    main()
