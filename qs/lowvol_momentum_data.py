"""Point-in-time data for the Shanghai and Shenzhen main-board stock study."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research.lowvol_momentum_scan import DEFAULT_CACHE, END, START

SME_MERGER = pd.Timestamp("2021-04-06")
RISK_LIMIT_CHANGE = pd.Timestamp("2026-07-06")


class CoverageError(RuntimeError):
    """Historical data are insufficient to certify the study."""


@dataclass(frozen=True)
class Distribution:
    column: int
    record: int
    ex_date: int
    pay: int
    share_pay: int
    cash: float
    shares: float


@dataclass
class Market:
    dates: pd.DatetimeIndex
    codes: pd.Index
    close: np.ndarray
    amount: np.ndarray
    returns: np.ndarray
    eligible: np.ndarray
    buy: np.ndarray
    sell: np.ndarray
    risk: np.ndarray
    unresolved: np.ndarray
    month_ends: pd.DatetimeIndex
    exits: dict
    distributions: list
    audit: pd.DataFrame


def limit_prices(reference, risk, dates):
    """Compute main-board tick prices from the contemporaneous reference."""
    old_risk = risk & (dates < RISK_LIMIT_CHANGE)
    rate = np.where(old_risk, 0.05, 0.10)
    up = np.floor(reference * (1 + rate) * 100 + 0.50000001) / 100
    down = np.floor(reference * (1 - rate) * 100 + 0.50000001) / 100
    return np.maximum(up, reference + 0.01), np.maximum(0.01, np.minimum(down, reference - 0.01))


def execution_masks(close, high, low, reference, risk, dates, verified):
    up, down = limit_prices(reference, risk, dates)
    # A bar outside the rule-based band needs exchange-level confirmation.
    in_band = (high <= up + 0.0001) & (low >= down - 0.0001)
    tradable = verified & np.isfinite(close) & in_band
    return tradable & (high < up - 0.0001) & ~risk, tradable & (low > down + 0.0001)


def _cached(root, component, code, dates, columns):
    path = root / component / f"{code}.parquet"
    if path.is_file():
        frame = pd.read_parquet(path, columns=["date", *columns])
        return frame.set_index("date", verify_integrity=True).reindex(dates)
    return pd.DataFrame(np.nan, index=dates, columns=columns)


def _distributions(root, code, column, dates):
    path = root / "actions" / f"{code}.parquet"
    if not path.is_file():
        return []
    rows = pd.read_parquet(path)
    rows = rows.loc[
        rows["announced"].le(rows["record"])
        & rows["record"].lt(rows["date"])
        & rows["date"].isin(dates)
    ].copy()
    rows[["cash", "bonus", "transfer"]] = rows[["cash", "bonus", "transfer"]].fillna(0) / 10
    events = []
    for row in rows.itertuples():
        cash, shares = row.cash, row.bonus + row.transfer
        if not cash and not shares:
            continue
        record, ex_date = dates.get_indexer([row.record, row.date])
        if record < 0:
            continue
        # Unverified payment timing leaves the entitlement unavailable for trading.
        pay = len(dates) if pd.isna(row.pay) else int(dates.searchsorted(max(row.pay, row.date)))
        share_pay = len(dates) if pd.isna(row.share_pay) else int(
            dates.searchsorted(max(row.share_pay, row.date))
        )
        events.append(Distribution(column, record, ex_date, pay, share_pay, cash, shares))
    return events


def _membership(intervals, dates):
    listed = np.zeros(len(dates), dtype=bool)
    mature = listed.copy()
    exits = []
    for row in intervals.itertuples():
        first = dates.searchsorted(row.listed)
        last = len(dates) if pd.isna(row.delisted) else dates.searchsorted(row.delisted)
        board_start = dates.searchsorted(SME_MERGER) if row.code.startswith("002") else 0
        listed[max(first, board_start):last] = True
        # Dates before the cached calendar have enough history by the study start.
        mature[max(first + 249, board_start):last] = True
        if last < len(dates):
            exits.append(int(last))
    return listed, mature, exits


def _load_symbol(root, code, intervals, dates, column):
    raw = _cached(root, "raw", code, dates, ["close", "high", "low", "volume", "amount"])
    status = _cached(root, "status", code, dates, ["close", "preclose", "tradestatus", "isST"])
    events = _distributions(root, code, column, dates)
    membership, mature, exits = _membership(intervals, dates)
    raw_trade = raw["close"].gt(0) & raw["volume"].gt(0)
    status_trade = status["tradestatus"].eq(1) & status["close"].gt(0)
    raw_on_suspension = raw_trade & status["tradestatus"].eq(0)
    close = raw["close"].where(raw_trade & ~raw_on_suspension).combine_first(
        status["close"].where(status_trade)
    )
    observed = close.dropna()
    prior = observed.shift().reindex(dates)
    reference = status["preclose"].where(status["preclose"].gt(0))
    risk = status["isST"].eq(1)
    price_conflict = raw_trade & status_trade & raw["close"].sub(status["close"]).abs().gt(0.01001)
    both = raw_trade & status_trade & ~price_conflict
    verified = both & status["isST"].isin([0, 1]) & reference.notna()
    cash = np.zeros(len(dates))
    shares = cash.copy()
    for event in events:
        cash[event.ex_date] += event.cash
        shares[event.ex_date] += event.shares
    expected_reference = (prior - cash) / (1 + shares)
    has_action = (cash != 0) | (shares != 0)
    unresolved = close.notna() & prior.notna() & (
        (reference.notna() & reference.sub(expected_reference).abs().gt(0.0151))
        | (has_action & reference.isna())
    )
    action_conflicts = set(np.flatnonzero(unresolved))
    events = [event for event in events if event.ex_date not in action_conflicts]
    signal_returns = ((close * (1 + shares) + cash) / prior - 1).where(verified & ~unresolved)
    buy, sell = execution_masks(
        close.to_numpy(), raw["high"].to_numpy(), raw["low"].to_numpy(), reference.to_numpy(),
        risk.to_numpy(), dates, (verified & ~unresolved).to_numpy(),
    )
    eligible = mature & (verified & ~unresolved & ~risk).to_numpy()
    amount = raw["amount"].where(raw_trade).where(~status["tradestatus"].eq(0), 0)
    in_sample = dates >= pd.Timestamp(START)
    audit = {
        "code": code,
        "exit_intervals": len(exits),
        "missing_raw_file": int(not (root / "raw" / f"{code}.parquet").is_file()),
        "missing_status_file": int(not (root / "status" / f"{code}.parquet").is_file()),
        "missing_actions_file": int(not (root / "actions" / f"{code}.parquet").is_file()),
        "akshare_traded": int((raw_trade & membership & in_sample).sum()),
        "status_traded": int((status_trade & membership & in_sample).sum()),
        "missing_status": int((raw_trade & status["tradestatus"].isna() & membership & in_sample).sum()),
        "close_disagreements": int(((price_conflict | raw_on_suspension) & membership & in_sample).sum()),
        "unresolved_actions": int((unresolved & membership & in_sample).sum()),
        "unverified_exits": 0,
    }
    exits_verified = {}
    for day in exits:
        terminal = close.iloc[:day].where(verified.iloc[:day]).dropna()
        last_quote = close.iloc[:day].last_valid_index()
        terminal_price = float(terminal.iloc[-1]) if len(terminal) else 0.0
        if last_quote is not None and (terminal.empty or terminal.index[-1] != last_quote):
            audit["unverified_exits"] += 1
            terminal_price = 0.0
        exits_verified[day] = terminal_price
    values = {
        "close": close.to_numpy(), "amount": amount.to_numpy(),
        "returns": signal_returns.to_numpy(), "eligible": eligible,
        "buy": buy & membership, "sell": sell & membership, "risk": risk.to_numpy(),
        "unresolved": (unresolved | price_conflict | raw_on_suspension).to_numpy(),
    }
    return values, exits_verified, events, audit


def load_market(root=DEFAULT_CACHE, *, end=END, survivors_only=False):
    root = Path(root)
    listings = pd.read_parquet(root / "listings.parquet")
    listings = listings.loc[listings["listed"].le(pd.Timestamp(end))]
    if survivors_only:
        listings = listings.loc[~listings["exit_registry"]]
    calendar = pd.DatetimeIndex(pd.read_parquet(root / "calendar.parquet")["date"])
    month_ends = calendar[~calendar.to_period("M").duplicated(keep="last")]
    dates = calendar[calendar <= pd.Timestamp(end)]
    groups = listings.groupby("code", sort=True)
    codes = pd.Index(list(groups.groups), name="code")
    shape = (len(dates), len(codes))
    arrays = {field: np.full(shape, np.nan, order="F") for field in ("close", "amount", "returns")}
    arrays.update({
        field: np.zeros(shape, dtype=bool, order="F")
        for field in ("eligible", "buy", "sell", "risk", "unresolved")
    })
    exits = {}
    distributions = []
    audits = []
    for column, (code, intervals) in enumerate(groups):
        values, terminal, events, audit = _load_symbol(root, code, intervals, dates, column)
        for field, matrix in arrays.items():
            matrix[:, column] = values[field]
        for day, price in terminal.items():
            exits.setdefault(day, []).append((column, price))
        distributions.extend(events)
        audits.append(audit)
    return Market(
        dates, codes, **arrays, month_ends=month_ends.intersection(dates), exits=exits,
        distributions=distributions, audit=pd.DataFrame(audits),
    )
