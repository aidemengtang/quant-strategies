"""An independent share and cash ledger for executable A-share orders."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOT = 100
MAX_WEIGHT = 0.08
PARTICIPATION = 0.05
INITIAL_CASH = 5_000_000.0
STAMP_CHANGE = pd.Timestamp("2023-08-28")
TRANSFER_CHANGE = pd.Timestamp("2022-04-29")
TRANSFER_UNIFICATION = pd.Timestamp("2015-08-01")


@dataclass
class Simulation:
    nav: pd.Series
    turnover: pd.Series
    friction: pd.Series
    max_weight: float
    unresolved_exposure: int
    trades: pd.DataFrame
    weights: pd.DataFrame | None
    returns: pd.DataFrame | None


def _lot_down(shares):
    return int(max(0, shares)) // LOT * LOT


def _distribution_schedule(events):
    record, ex_date = defaultdict(list), defaultdict(list)
    for event in events:
        record[event.record].append(event)
        ex_date[event.ex_date].append(event)
    return record, ex_date


def _rates(date, slip, levies):
    transfer = 0.0
    if levies:
        if date < TRANSFER_UNIFICATION:
            # Conservative common rate before the exchange fee bases were unified.
            transfer = 0.0003
        else:
            transfer = 0.00002 if date < TRANSFER_CHANGE else 0.00001
    stamp = (0.001 if date < STAMP_CHANGE else 0.0005) if levies else 0
    return slip + transfer, slip + transfer + stamp


def _fees(notional, commission, variable):
    return (max(5.0, commission * notional) if commission else 0) + variable * notional


def simulate(market, signals, adv, *, start, initial_cash=INITIAL_CASH, commission=0.0005,
             slip=0.001, levies=True, record=False, zero_recovery=False, write_off_unknown=False):
    """Generate closing positions from orders frozen at the preceding signal close."""
    first = market.dates.searchsorted(start)
    dates = market.dates[first:]
    count = len(market.codes)
    shares = np.zeros(count, dtype=np.int64)
    pending_shares = shares.copy()
    desired = shares.copy()
    marks = np.zeros(count)
    if first:
        for day in range(first):
            np.copyto(marks, market.close[day], where=np.isfinite(market.close[day]))
    cash = initial_cash
    receivable = 0.0
    retirement_value = 0.0
    cash_due, shares_due = defaultdict(list), defaultdict(list)
    recorded = {}
    record_schedule, ex_schedule = _distribution_schedule(market.distributions)
    buy_expiry = -1
    max_weight = 0.0
    unresolved_exposure = 0
    nav_values = np.empty(len(dates))
    turnover = np.zeros(len(dates))
    friction = np.zeros(len(dates))
    trades = []
    weight_rows = np.zeros((len(dates), count)) if record else None
    return_rows = np.zeros_like(weight_rows) if record else None

    for row, date in enumerate(dates):
        day = first + row
        previous_assets = (shares + pending_shares) * marks
        affected = (previous_assets > 0) & market.unresolved[day]
        unresolved_exposure += np.count_nonzero(affected)
        if write_off_unknown:
            shares[affected] = pending_shares[affected] = desired[affected] = 0
        np.copyto(marks, market.close[day], where=np.isfinite(market.close[day]))
        distributions = np.zeros(count)
        for event in ex_schedule[day]:
            quantity = recorded.pop(event, 0)
            gross_dividend = quantity * event.cash
            dividend = gross_dividend * (0.8 if levies else 1.0)
            friction[row] += gross_dividend - dividend
            bonus = int(quantity * event.shares + 1e-8)
            if dividend:
                receivable += dividend
                cash_due[event.pay].append(dividend)
                distributions[event.column] += dividend
            if bonus:
                pending_shares[event.column] += bonus
                shares_due[event.share_pay].append((event.column, bonus))
        paid = sum(cash_due.pop(day, []))
        cash += paid
        receivable -= paid
        for column, quantity in shares_due.pop(day, []):
            # Delisting may terminate a pending share entitlement before payment.
            settled = min(quantity, pending_shares[column])
            shares[column] += settled
            pending_shares[column] -= settled
        for column, terminal in market.exits.get(day, ()):
            proceeds = (shares[column] + pending_shares[column]) * (0 if zero_recovery else terminal)
            retirement_value += proceeds
            distributions[column] += proceeds
            shares[column] = pending_shares[column] = desired[column] = 0

        assets = (shares + pending_shares) * marks
        if record:
            np.divide(assets + distributions - previous_assets, previous_assets,
                      out=return_rows[row], where=previous_assets > 0)
        equity = cash + receivable + retirement_value + assets.sum()
        buy_rate, sell_rate = _rates(date, slip, levies)
        # Risk-warning orders may liquidate, but they cannot add exposure.
        desired[market.risk[day]] = 0
        sold = np.flatnonzero((shares + pending_shares > desired) & market.sell[day])
        bought = np.flatnonzero((shares + pending_shares < desired) & market.buy[day]) if day <= buy_expiry else ()
        for direction, columns, rate in ((-1, sold, sell_rate), (1, bought, buy_rate)):
            for column in columns:
                price = market.close[day, column]
                amount = adv[day, column]
                if not np.isfinite(amount) or amount <= 0:
                    continue
                raw_size = abs(desired[column] - shares[column] - pending_shares[column])
                limit = min(amount, market.amount[day, column]) * PARTICIPATION / price
                if not np.isfinite(limit) or limit < 1:
                    continue
                if direction == -1:
                    size = min(raw_size, shares[column])
                    # An entire existing odd-lot position can be liquidated.
                    quantity = size if size == shares[column] and size <= limit else _lot_down(min(size, limit))
                else:
                    minimum_fee = 5.0 if commission else 0.0
                    affordable = min(cash / (1 + commission + rate), (cash - minimum_fee) / (1 + rate))
                    headroom = (MAX_WEIGHT * (equity - minimum_fee) - assets[column])
                    headroom /= 1 + MAX_WEIGHT * (commission + rate)
                    quantity = _lot_down(min(raw_size, limit, affordable / price, headroom / price))
                if not quantity:
                    continue
                notional = quantity * price
                fee = _fees(notional, commission, rate)
                shares[column] += direction * quantity
                cash -= direction * notional + fee
                assets[column] += direction * notional
                turnover[row] += notional
                friction[row] += fee
                capacity = PARTICIPATION * amount * equity / (raw_size * price)
                trades.append((date, market.codes[column], direction, quantity, notional, fee, capacity))
                equity -= fee

        for event in record_schedule[day]:
            recorded[event] = shares[event.column] + pending_shares[event.column]
        nav_values[row] = equity
        exposure = assets / equity
        max_weight = max(max_weight, exposure.max(initial=0))
        if record:
            weight_rows[row] = exposure
        if day in signals:
            desired.fill(0)
            for column, weight in signals[day].items():
                desired[column] = _lot_down(equity * weight / market.close[day, column])
            buy_expiry = day + 3

    holdings, returns = None, None
    if record:
        used = (weight_rows > 0).any(axis=0)
        holdings = pd.DataFrame(weight_rows[:, used], index=dates, columns=market.codes[used])
        returns = pd.DataFrame(return_rows[:, used], index=dates, columns=market.codes[used])
    previous_nav = np.r_[initial_cash, nav_values[:-1]]
    return Simulation(
        pd.Series(nav_values / initial_cash, index=dates, name="nav"),
        pd.Series(turnover / previous_nav, index=dates, name="turnover"),
        pd.Series(friction / previous_nav, index=dates, name="friction"),
        max_weight,
        unresolved_exposure,
        pd.DataFrame(trades, columns=["date", "code", "side", "shares", "notional", "fee", "capacity"]),
        holdings,
        returns,
    )
