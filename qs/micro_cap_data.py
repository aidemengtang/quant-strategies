"""Point-in-time microcap panels from the local Tushare archive and short gap cache."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path("/Users/mac/PycharmProjects/tushare_data")
CACHE = ROOT / "data/micro_cap"
CALENDAR = ROOT / "data/lowvol_momentum/calendar.parquet"
START = "2015-01-05"
IS_END = "2020-12-31"
END = "2026-11-06"
BJ_OPEN = "2021-11-15"
QUOTE = ["ts_code", "close", "high", "low", "pre_close", "pct_chg", "amount"]
PANELS = ("returns", "adjusted", "close", "amount", "adv", "cap", "buy", "sell", "risk", "known")


@dataclass
class Market:
    dates: pd.DatetimeIndex
    codes: pd.Index
    returns: np.ndarray
    adjusted: np.ndarray
    close: np.ndarray
    amount: np.ndarray
    adv: np.ndarray
    cap: np.ndarray
    buy: np.ndarray
    sell: np.ndarray
    risk: np.ndarray
    known: np.ndarray
    exits: dict
    audit: dict

    def through(self, end):
        stop = self.dates.searchsorted(pd.Timestamp(end), side="right")
        return replace(self, dates=self.dates[:stop], exits={d: x for d, x in self.exits.items() if d < stop},
                       **{field: getattr(self, field)[:stop] for field in PANELS})


def adjusted_prices(returns):
    """Compound observed exchange returns; missing quotes leave only the valuation mark."""
    return np.cumprod(1 + np.nan_to_num(returns, nan=0), axis=0, dtype=np.float64)


def rolling_adv(amount):
    return pd.DataFrame(amount).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)


def market_file(archive, table, day):
    relative = f"{table}/data/{table}__trade_date={day:%Y%m%d}.parquet"
    for folder in ("pkg_parquet", "pkg_parquet_incr"):
        path = archive / folder / relative
        if path.is_file():
            return path
    return None


def optional_table(archive, table, day, columns):
    path = market_file(archive, table, day)
    return pd.read_parquet(path, columns=columns) if path is not None else None


def accepted_codes(codes):
    return codes.str.fullmatch(r"(?:60[0135]\d{3}|68[89]\d{3})\.SH|(?:00[0-3]\d{3}|30[01]\d{3})\.SZ|\d{6}\.BJ")


def read_quotes(archive, cache, day, bj):
    frame = optional_table(archive, "daily", day, QUOTE)
    if frame is None:
        factor_fields = ["pct_change" if f == "pct_chg" else f for f in QUOTE]
        frame = optional_table(archive, "stk_factor", day, factor_fields)
        if frame is not None:
            frame = frame.rename(columns={"pct_change": "pct_chg"})
    basics = optional_table(archive, "daily_basic", day, ["ts_code", "turnover_rate"])
    if frame is None or basics is None:
        path = cache / "shsz" / f"{day:%Y-%m-%d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"{day:%Y-%m-%d} missing quote snapshot; use micro_cap_fetch --resume")
        pieces = [pd.read_parquet(path)]
        if len(bj):
            pieces.append(bj.loc[bj.date.eq(day.strftime("%Y-%m-%d"))].drop(columns="date"))
        return pd.concat(pieces, ignore_index=True)
    frame["amount"] *= 1000
    return frame.merge(basics, on="ts_code", how="left", validate="one_to_one")


def lot_rules(code):
    if code.endswith(".BJ"):
        return 100, 1
    if code.startswith(("688", "689")):
        return 200, 1
    return 100, 100


def historical_rate(day, codes, risk):
    bj = codes.str.endswith(".BJ")
    star = codes.str.startswith(("688", "689"))
    growth = codes.str.startswith(("300", "301")) & (day >= pd.Timestamp("2020-08-24"))
    rate = np.full(len(codes), 0.10)
    if day < pd.Timestamp("2026-07-06"):
        rate[risk == 1] = 0.05
    rate[star | growth] = 0.20
    rate[bj] = 0.30
    return rate


def limit_masks(close, high, low, up, down, verified):
    # Missing band and bars outside the official/modelled band are ambiguous.
    good = verified & np.isfinite(up + down + close + high + low)
    good &= (high <= up + 1e-4) & (low >= down - 1e-4) & (up > down)
    return good & (high < up - 1e-4), good & (low > down + 1e-4), good


def names_for(archive, codes, dates):
    flags = np.full((len(dates), len(codes)), -1, dtype=np.int8)
    folder = archive / "pkg_parquet/namechange/data"
    for column, code in enumerate(codes):
        path = folder / f"namechange__ts_code={code}.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=["name", "start_date", "end_date", "ann_date"])
        frame["start_date"] = pd.to_datetime(frame.start_date, errors="coerce")
        frame["end_date"] = pd.to_datetime(frame.end_date, errors="coerce")
        frame["ann_date"] = pd.to_datetime(frame.ann_date, errors="coerce")
        for name, start, end, announced in frame.sort_values("start_date").itertuples(index=False, name=None):
            if pd.isna(start) or not isinstance(name, str):
                continue
            first = dates.searchsorted(max(start, announced) if pd.notna(announced) else start)
            last = len(dates) if pd.isna(end) else dates.searchsorted(end, side="right")
            risk = name.upper().startswith(("ST", "*ST", "SST", "S*ST")) or "退" in name
            flags[first:last, column] = int(risk)
    return flags


def build_market(archive=ARCHIVE, cache=CACHE, *, end=END):
    archive, cache = Path(archive), Path(cache)
    dates = pd.DatetimeIndex(pd.read_parquet(CALENDAR)["date"])
    dates = dates[(dates >= "2014-11-01") & (dates <= end)]
    basics = pd.read_parquet(
        archive / "pkg_parquet/stock_basic/data/stock_basic__snapshot=20261012.parquet",
        columns=["ts_code", "list_date", "delist_date"],
    ).set_index("ts_code", verify_integrity=True)
    official = cache / "bj_listings.parquet"
    if official.is_file():
        new = pd.read_parquet(official)
        new["ts_code"] = new["证券代码"].astype(str).str.zfill(6) + ".BJ"
        new = new.set_index("ts_code")
        basics = basics.combine_first(new["上市日期"].rename("list_date").to_frame())
    listed = pd.to_datetime(basics.list_date, format="mixed", errors="coerce")
    delisted = pd.to_datetime(basics.delist_date, format="mixed", errors="coerce")
    bj_files = sorted((cache / "bj").glob("*.parquet"))
    bj = pd.concat([pd.read_parquet(p) for p in bj_files], ignore_index=True) if bj_files else pd.DataFrame()
    quotes = []
    observed = set()
    for day in dates:
        frame = read_quotes(archive, cache, day, bj)
        frame = frame.loc[accepted_codes(frame.ts_code)].set_index("ts_code", verify_integrity=True)
        board = frame.index.str.endswith(".BJ")
        if day < pd.Timestamp(BJ_OPEN):
            frame = frame.loc[~board]
        else:
            valid_bj = listed.reindex(frame.index).le(day)
            frame = frame.loc[~board | valid_bj]
        frame = frame.loc[~delisted.reindex(frame.index).le(day)]
        quotes.append(frame)
        observed.update(frame.index)
    codes = pd.Index(sorted(observed), name="code")
    shape = (len(dates), len(codes))
    returns, close, amount, cap = [np.full(shape, np.nan, dtype=np.float32) for _ in range(4)]
    buy, sell, risk, known = [np.zeros(shape, dtype=bool) for _ in range(4)]
    historical_names = names_for(archive, codes, dates)
    old = np.flatnonzero(dates < pd.Timestamp("2016-08-09"))
    for column, code in enumerate(codes):
        path = ROOT / "data/lowvol_momentum/status" / f"{code.split('.')[0]}.parquet"
        if len(old) and path.is_file():
            status = pd.read_parquet(path, columns=["date", "isST"]).set_index("date")["isST"]
            values = pd.to_numeric(status.reindex(dates[old]), errors="coerce").fillna(-1)
            existing = historical_names[old, column]
            historical_names[old, column] = np.where(values.ge(0), values, existing)
    missing_pct = confirmed_zero = 0
    for i, (day, frame) in enumerate(zip(dates, quotes)):
        indices = codes.get_indexer(frame.index)
        numeric = frame[QUOTE[1:] + ["turnover_rate"]].apply(pd.to_numeric, errors="coerce")
        p, a, pct = (numeric[k].to_numpy() for k in ("close", "amount", "pct_chg"))
        traded = (p > 0) & (a > 0)
        close[i, indices] = np.where(traded, p, np.nan)
        amount[i, indices] = np.where(a >= 0, a, np.nan)
        valid_returns = traded & np.isfinite(pct) & (pct > -100)
        missing_pct += int(np.count_nonzero(traded & ~valid_returns))
        returns[i, indices] = np.where(valid_returns, pct / 100, np.nan)

        flags = historical_names[i, indices].copy()
        st = optional_table(archive, "stock_st", day, ["ts_code"])
        if st is not None:
            flags[:] = frame.index.isin(st.ts_code).astype(np.int8)
        if "risk" in frame:
            supplemental = pd.to_numeric(frame["risk"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            flags = np.where(np.isfinite(supplemental), supplemental, flags)
        risk[i, indices] = flags == 1
        suspension = optional_table(archive, "suspend_d", day, ["ts_code", "suspend_type"])
        suspended_codes = (suspension.loc[suspension.suspend_type.eq("S"), "ts_code"]
                           if suspension is not None else [])
        suspended = frame.index.isin(suspended_codes)
        # An exchange-confirmed no-quote suspension has zero turnover, not an imputed price.
        suspended_indices = codes.get_indexer(pd.Index(suspended_codes).unique())
        suspended_indices = suspended_indices[suspended_indices >= 0]
        no_quote = suspended_indices[np.isnan(amount[i, suspended_indices])]
        amount[i, no_quote] = 0.
        confirmed_zero += len(no_quote)
        ref = numeric["pre_close"].to_numpy()
        limits = optional_table(archive, "stk_limit", day, ["ts_code", "up_limit", "down_limit"])
        rate = historical_rate(day, frame.index, flags)
        up = np.maximum(np.floor(ref * (1 + rate) * 100 + .50000001) / 100, ref + .01)
        down = np.maximum(.01, np.minimum(np.floor(ref * (1 - rate) * 100 + .50000001) / 100, ref - .01))
        if limits is not None:
            limit = limits.set_index("ts_code").reindex(frame.index)
            actual_up = pd.to_numeric(limit.up_limit, errors="coerce").to_numpy()
            actual_down = pd.to_numeric(limit.down_limit, errors="coerce").to_numpy()
            published = np.isfinite(actual_up + actual_down) & (actual_up > actual_down) & (actual_down > 0)
            up = np.where(published, actual_up, up)
            down = np.where(published, actual_down, down)
        mask = valid_returns & ~suspended & (flags >= 0) & (ref > 0)
        b, s, valid = limit_masks(p, numeric["high"].to_numpy(), numeric["low"].to_numpy(), up, down, mask)
        buy[i, indices], sell[i, indices], known[i, indices] = b & (flags == 0), s, valid
        turn = numeric["turnover_rate"].to_numpy()
        recent = np.isfinite(close[i - 1, indices]) if i else np.ones(len(frame), dtype=bool)
        good_cap = traded & (turn > 0) & (recent | ((turn >= .01) & (turn <= 80)))
        cap[i, indices[good_cap]] = a[good_cap] * 100 / turn[good_cap]

    exits = {}
    for column, day in enumerate(delisted.reindex(codes)):
        if pd.notna(day):
            i = dates.searchsorted(day)
            if 0 <= i < len(dates):
                exits.setdefault(int(i), []).append(column)
    audit = {
        "codes": len(codes), "observed_delisted": int(delisted.reindex(codes).notna().sum()),
        "sessions": len(dates), "missing_pct_on_traded_rows": missing_pct,
        "confirmed_zero_suspensions": confirmed_zero,
        "unverified_traded_rows": int((np.isfinite(close) & ~known).sum()),
        "archive_end": max(str(p.stem.split("=")[-1]) for p in
                           (archive / "pkg_parquet/stk_factor/data").glob("*.parquet")),
        "gap_errors": len(pd.read_csv(cache / "fetch_failures.csv")) if (cache / "fetch_failures.csv").is_file() else 0,
    }
    return Market(dates, codes, returns, adjusted_prices(returns), close, amount, rolling_adv(amount),
                  cap, buy, sell, risk, known, exits, audit)
