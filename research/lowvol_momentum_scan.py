"""Audit exchange listings and sample historical prices before bulk fetching."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "lowvol_momentum"
WARMUP_START = "2014-01-01"
START = "2015-01-01"
IS_END = "2020-12-31"
OOS_START = "2021-01-01"
END = "2026-11-07"

PREFIXES = {"sh": ("600", "601", "603", "605"), "sz": ("000", "001", "002", "003")}


def exchange_listings():
    import akshare as ak

    sources = [
        ("sh", False, ak.stock_info_sh_name_code, ["证券代码", "证券简称", "上市日期"]),
        ("sz", False, ak.stock_info_sz_name_code, ["A股代码", "A股简称", "A股上市日期"]),
        ("sh", True, lambda: ak.stock_info_sh_delist("沪市"),
         ["公司代码", "公司简称", "上市日期", "暂停上市日期"]),
        ("sz", True, ak.stock_info_sz_delist, ["证券代码", "证券简称", "上市日期", "终止上市日期"]),
    ]
    frames = []
    counts = {}
    for exchange, delisted, fetch, columns in sources:
        raw = fetch()
        counts[f"{exchange}_{'exit' if delisted else 'listed'}"] = len(raw)
        if raw.empty:
            raise ValueError(f"Empty {exchange} {'exit' if delisted else 'listed'} registry")
        frame = raw[columns].copy()
        frame.columns = ["code", "name", "listed"] + (["delisted"] if delisted else [])
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame = frame.loc[frame["code"].str.startswith(PREFIXES[exchange])].copy()
        frame["exchange"] = exchange
        frame["listed"] = pd.to_datetime(frame["listed"], errors="coerce")
        frame["delisted"] = pd.to_datetime(frame["delisted"], errors="coerce") if delisted else pd.NaT
        frame["exit_registry"] = delisted
        frames.append(frame)
    records = pd.concat(frames, ignore_index=True)
    invalid = records["listed"].isna() | (records["exit_registry"] & records["delisted"].isna())
    failures = records.loc[invalid, ["exchange", "code", "name"]].assign(issue="missing listing dates")
    records = records.loc[~invalid]
    active = records.loc[
        records["listed"].le(pd.Timestamp(END))
        & (~records["exit_registry"] | records["delisted"].ge(pd.Timestamp(START)))
    ].sort_values(["exchange", "code", "listed"])
    # The SSE company-level table can repeat the same exit on its A/B records.
    active = active.drop_duplicates(["exchange", "code", "delisted"])
    return active.reset_index(drop=True), failures, counts


def probe_history(code, start, end):
    import akshare as ak

    try:
        frame = ak.stock_zh_a_hist(
            symbol=code, start_date=start.replace("-", ""), end_date=end.replace("-", ""),
            adjust="", timeout=15,
        )
        if frame.empty:
            raise ValueError("no daily bars")
        dates = pd.to_datetime(frame["日期"])
        return {"code": code, "rows": len(frame), "first": str(dates.min().date()),
                "last": str(dates.max().date()), "error": ""}
    except Exception as exc:
        return {"code": code, "rows": 0, "error": f"{type(exc).__name__}: {exc}"}


def scan(cache, sample_per_group=3):
    import akshare as ak

    cache.mkdir(parents=True, exist_ok=True)
    records, failures, counts = exchange_listings()
    records.to_parquet(cache / "listings.parquet", index=False)
    failures.to_csv(cache / "registry_failures.csv", index=False)
    calendar = pd.to_datetime(ak.tool_trade_date_hist_sina()["trade_date"])
    calendar_end = pd.Timestamp(END) + pd.offsets.MonthEnd(0)
    calendar = calendar.loc[calendar.between(WARMUP_START, calendar_end)].sort_values()
    if calendar.empty or calendar.iloc[-1] < pd.Timestamp(END) - pd.Timedelta(days=7):
        raise ValueError("Trading calendar does not cover the study")
    calendar.to_frame(name="date").to_parquet(cache / "calendar.parquet", index=False)
    coverage = []
    for year in range(2015, 2027):
        beginning = pd.Timestamp(f"{year}-01-01")
        end = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(END))
        listed = records["listed"].le(end)
        exited = records["delisted"].between(beginning, end)
        alive = records["delisted"].isna() | records["delisted"].ge(beginning)
        old_sme = records["exchange"].eq("sz") & records["code"].str.startswith("002")
        if end < pd.Timestamp("2021-04-06"):
            listed &= ~old_sme
        coverage.append({"year": year, "registry_intervals": int((listed & alive).sum()),
                         "exits": int((listed & exited).sum())})
    groups = records.assign(exit_year=records["delisted"].dt.year.fillna(0))
    sample = groups.groupby(["exchange", "exit_year"]).head(sample_per_group)
    with ThreadPoolExecutor(max_workers=4) as workers:
        probes = list(workers.map(lambda code: probe_history(code, WARMUP_START, END), sample["code"]))
    pd.DataFrame(probes).to_csv(cache / "sample_coverage.csv", index=False)
    summary = {
        "source_rows": counts, "registry_intervals": len(records),
        "unique_codes": int(records["code"].nunique()),
        "exit_intervals": int(records["exit_registry"].sum()),
        "registry_failures": len(failures), "sample_size": len(probes),
        "sample_price_failures": sum(bool(row["error"]) for row in probes),
        "years": coverage,
    }
    (cache / "scan.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sample-per-group", type=int, default=3)
    args = parser.parse_args()
    os.environ.update(no_proxy="*", NO_PROXY="*")
    print(json.dumps(scan(args.cache, args.sample_per_group), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
