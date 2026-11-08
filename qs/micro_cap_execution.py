"""Close execution with adjusted valuation and exchange lot sizes."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qs.micro_cap_data import START, lot_rules

CAPITAL = 5_000_000
PARTICIPATION = .05
TRADE_COLUMNS = ["date", "code", "side", "planned", "filled", "shares", "cost", "reason", "adjustment_cash"]
CAPACITY_COLUMNS = ["date", "hold_bottleneck_code", "hold_adv", "hold_weight", "hold_kind",
                    "order_bottleneck_code", "order_adv", "order_kind"]


def fee_rates(dates):
    stamp = np.where(dates >= "2023-08-28", .0005, .001)
    transfer = np.select([dates >= "2022-04-29", dates >= "2015-08-01"], [.00001, .00002], .0003)
    return stamp, transfer


def fees(notional, rate, other):
    return np.where(notional > 0, np.maximum(notional * rate, 5) + notional * other, 0.)


def _bottleneck(codes, adv, code):
    if code is None:
        return None, np.nan, None
    value = adv[code]
    kind = "unavailable" if not np.isfinite(value) else "zero" if value == 0 else None
    return codes[code], value, kind


@dataclass
class Ledger:
    nav: pd.Series
    gross: pd.Series
    turnover: pd.Series
    count: pd.Series
    max_weight: pd.Series
    cash: pd.Series
    holding_capacity: pd.Series
    order_capacity: pd.Series
    trades: pd.DataFrame
    weights: pd.DataFrame
    capacity_diag: pd.DataFrame


def simulate(market, selections, n, *, rate=.0005, slip=.001, recovery=1., record=False, adv_floor=0.):
    start = market.dates.searchsorted(START)
    stamp, transfer = fee_rates(market.dates)
    lots = [lot_rules(code) for code in market.codes]
    cash, previous = float(CAPITAL), float(CAPITAL)
    held, pending, fills, weights, rows = {}, {}, [], [], []
    capacity_diag = []
    expiry = -1

    def log(code, side, planned, filled, shares, cost, reason, adjustment_cash=0.):
        if record:
            fills.append((market.dates[day], market.codes[code], side, planned, filled,
                          shares, cost, reason, adjustment_cash))

    for day in range(start, len(market.dates)):
        marks = market.adjusted[day]
        adv = market.adv[day - 1] if day else market.adv[day]
        turnover = costs = 0.
        for code in market.exits.get(day, ()):
            pending.pop(code, None)
            value = held.pop(code, 0.) * marks[code]
            if value:
                cash += value * recovery
                log(code, "delist", value, value * recovery, 0, 0., "verified_mark" if recovery else "zero_recovery")
        equity = cash + sum(units * marks[code] for code, units in held.items())
        bottleneck = np.inf
        order_code = None
        for code, target in list(pending.items()):
            delta = (target - held.get(code, 0.)) * marks[code]
            if abs(delta) < .01:
                del pending[code]
                continue
            buying = delta > 0
            if buying and day > expiry:
                log(code, "buy", delta, 0., 0, 0., "expired")
                del pending[code]
                continue
            # Process all sales before purchases, with locked holdings occupying slots.
            if buying:
                continue
            if equity > 0:
                capacity = PARTICIPATION * adv[code] * equity / -delta if np.isfinite(adv[code]) else 0.
                if record and capacity < bottleneck:
                    order_code = code
                bottleneck = min(bottleneck, capacity)
            if not market.sell[day, code]:
                log(code, "sell", -delta, 0., 0, 0., "down_limit" if market.known[day, code] else "suspended_or_unknown")
                continue
            p = float(market.close[day, code])
            cap = PARTICIPATION * adv[code] if np.isfinite(adv[code]) else 0.
            minimum, step = lots[code]
            full_exit = target == 0 and -delta <= cap
            if full_exit:
                # The remaining economic claim can include a fractional equivalent share.
                shares = int(-delta / p + 1e-10)
                value = shares * p
                residual = -delta - value
            else:
                shares = int(min(-delta, cap) / p) // step * step
                value = shares * p
                residual = 0.
                if shares < minimum:
                    log(code, "sell", -delta, 0., 0, 0., "adv_or_lot")
                    if target > 0 and -delta < minimum * p:
                        del pending[code]
                    continue
            cost = max(rate * value, 5) + value * (slip + transfer[day] + stamp[day]) if shares else 0.
            if cash + value + residual < cost:
                log(code, "sell", -delta, 0., 0, 0., "cash_for_fee")
                continue
            if full_exit:
                del held[code]
                del pending[code]
            else:
                held[code] -= value / marks[code]
                if target > 0 and -delta - value < minimum * p:
                    del pending[code]
            cash += value + residual - cost
            costs += cost
            turnover += value
            log(code, "sell", -delta, value, shares, cost,
                "filled" if code not in pending else "partial", residual)
        for code, target in list(pending.items()):
            delta = (target - held.get(code, 0.)) * marks[code]
            if delta <= 0:
                continue
            if equity > 0:
                capacity = PARTICIPATION * adv[code] * equity / delta if np.isfinite(adv[code]) else 0.
                if record and capacity < bottleneck:
                    order_code = code
                bottleneck = min(bottleneck, capacity)
            if not market.buy[day, code]:
                reason = ("risk" if market.risk[day, code] else "up_limit") if market.known[day, code] else "suspended_or_unknown"
            elif not np.isfinite(adv[code]) or adv[code] < adv_floor:
                reason = "adv_floor"
            elif code not in held and len(held) >= n:
                reason = "slots"
            else:
                p = float(market.close[day, code])
                minimum, step = lots[code]
                cap = PARTICIPATION * adv[code] if np.isfinite(adv[code]) else 0.
                other = slip + transfer[day]
                budget = min((cash - 5) / (1 + other), cash / (1 + rate + other))
                shares = int(max(0., min(delta, cap, budget)) / p) // step * step
                if shares >= minimum:
                    value = shares * p
                    cost = max(rate * value, 5) + value * other
                    held[code] = held.get(code, 0.) + value / marks[code]
                    cash -= value + cost
                    costs += cost
                    turnover += value
                    if delta - value < minimum * p:
                        del pending[code]
                    log(code, "buy", delta, value, shares, cost, "filled" if code not in pending else "partial")
                    continue
                if delta < minimum * p:
                    del pending[code]
                reason = "cash_adv_or_lot"
            log(code, "buy", delta, 0., 0, 0., reason)
        nav = equity - costs
        if day in selections:
            selection = selections[day]
            allocation = selection.weights if selection.weights is not None else np.full(len(selection.names), 1 / n)
            pending = {code: nav * weight / marks[code] for code, weight in zip(selection.names, allocation)}
            pending.update({code: 0. for code in held if code not in pending})
            expiry = day + 3
        actual = {code: units * marks[code] / nav for code, units in held.items()} if nav > 0 else {}
        hold_code = None
        if actual:
            lowest = sorted(actual, key=lambda code: adv[code] if np.isfinite(adv[code]) else -1)[:25]
            hold_code, capacity = min(
                ((code, PARTICIPATION * adv[code] / actual[code] if np.isfinite(adv[code]) else 0.)
                 for code in lowest), key=lambda item: item[1])
        else:
            capacity = 0.
        if record:
            weights.extend((market.dates[day], market.codes[code], weight) for code, weight in actual.items())
            hold_name, hold_adv, hold_kind = _bottleneck(market.codes, adv, hold_code)
            capacity_diag.append((market.dates[day], hold_name, hold_adv, actual.get(hold_code, np.nan), hold_kind,
                                  *_bottleneck(market.codes, adv, order_code)))
        rows.append((nav / CAPITAL, (nav + costs) / previous, turnover / equity if equity else 0.,
                     len(actual), max(actual.values(), default=0.), cash / nav if nav else 1.,
                     capacity, bottleneck if np.isfinite(bottleneck) else np.nan))
        previous = nav
    frame = pd.DataFrame(rows, index=market.dates[start:], columns=(
        "nav", "gross", "turnover", "count", "max_weight", "cash", "holding_capacity", "order_capacity"))
    return Ledger(*(frame[c] for c in frame),
                  pd.DataFrame(fills, columns=TRADE_COLUMNS),
                  pd.DataFrame(weights, columns=["date", "code", "weight"]),
                  pd.DataFrame(capacity_diag, columns=CAPACITY_COLUMNS))
