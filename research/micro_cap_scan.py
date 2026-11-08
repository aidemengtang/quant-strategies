"""Inspect the local archive required by the approved microcap data scan."""

import argparse
from pathlib import Path

import pandas as pd

from qs.micro_cap_data import ARCHIVE, CACHE, CALENDAR
from research.micro_cap_fetch import sessions


def inspect(archive, cache):
    gap = sessions(archive, CALENDAR)
    missing = [f"{day:%Y-%m-%d}" for day in gap if not (cache / "shsz" / f"{day:%Y-%m-%d}.parquet").is_file()]
    print(f"存档需补 {len(gap)} 个交易日，沪深缓存未覆盖 {len(missing)} 个：{' '.join(missing) or '无'}")
    failures = cache / "fetch_failures.csv"
    failed = len(pd.read_csv(failures)) if failures.is_file() else 0
    print(f"定点补齐失败记录 {failed} 项")
    print("原第 8 节抽样、三档 ADV 及容量勘测见 reports/micro_cap_scan.md。")
    return not missing and not failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    raise SystemExit(0 if inspect(args.archive, args.cache) else 1)
