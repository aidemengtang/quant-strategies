"""Read-only daily capacity diagnostics using the micro-cap ledger convention."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qs.micro_cap_execution import CAPACITY_COLUMNS, PARTICIPATION, _bottleneck


@dataclass
class CapacityLedger:
    count: pd.Series
    holding_capacity: pd.Series
    order_capacity: pd.Series
    capacity_diag: pd.DataFrame


def snapshot(date, codes, adv, weights, order_weights):
    """Inspect actual closing holdings and pre-participation absolute order weights.

    ADV must already be lagged by the caller. Missing ADV gives zero capacity,
    while its diagnostic kind stays distinct from a verified zero ADV.
    """
    held = np.flatnonzero(weights > 0)
    orders = np.flatnonzero(order_weights > 0)

    def minimum(columns, allocation):
        if not len(columns):
            return None, np.nan
        capacities = PARTICIPATION * np.where(np.isfinite(adv[columns]), adv[columns], 0.)
        capacities /= allocation[columns]
        lowest = int(np.argmin(capacities))
        return columns[lowest], capacities[lowest]

    lowest_adv = sorted(held, key=lambda code: adv[code] if np.isfinite(adv[code]) else -1)[:25]
    hold_code, hold_capacity = minimum(np.array(lowest_adv, dtype=int), weights)
    order_code, order_capacity = minimum(orders, order_weights)
    name, value, kind = _bottleneck(codes, adv, hold_code)
    row = (date, name, value, weights[hold_code] if hold_code is not None else np.nan, kind,
           *_bottleneck(codes, adv, order_code))
    return (len(held), hold_capacity if len(held) else 0., order_capacity), row


def ledger_from_rows(dates, rows, diagnostics):
    frame = pd.DataFrame(rows, index=dates, columns=["count", "holding_capacity", "order_capacity"])
    return CapacityLedger(*(frame[c] for c in frame), pd.DataFrame(diagnostics, columns=CAPACITY_COLUMNS))
