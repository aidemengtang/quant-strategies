from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qs.micro_cap_data import Market, adjusted_prices, limit_masks, rolling_adv
from qs.micro_cap_execution import CAPITAL, PARTICIPATION, fee_rates, fees, simulate
from research.micro_cap_study import capacity_audit, coarse, perturb
from strategies.micro_cap import Parameters, Selection, build_weights, rank, trailing


def market_for(returns):
    returns = np.array(returns, dtype=float)
    dates = pd.bdate_range("2015-01-05", periods=len(returns))
    codes = pd.Index([f"{600000 + i}.SH" for i in range(returns.shape[1])])
    shape = returns.shape
    close = np.full(shape, 10.)
    adv = np.full(shape, 100_000_000.)
    cap = np.broadcast_to(np.arange(len(codes)) + 1., shape)
    known = np.isfinite(returns)
    return Market(dates, codes, returns, adjusted_prices(returns), close, adv, adv.copy(), cap,
                  known, known.copy(), np.zeros(shape, dtype=bool), known.copy(), {}, {})


def selected(*names):
    return Selection(np.array(names, dtype=int), (), np.array([], dtype=int))


def test_independent_bucket_width_ranks_globally():
    scores = np.array([0., 0., 0., 10., 10., 10., 10., 10., 10., 20.])
    result = rank(np.arange(len(scores)), scores, width=5)
    assert result.names[:3].tolist() == [0, 1, 2]
    assert [len(bucket) for bucket in result.buckets] == [5, 5]
    assert all(result.bucket_ids[:3] == 0)


def test_split_does_not_create_a_raw_close_reversal():
    market = market_for([[0., 0.], [.02, -.05], [.01, -.01], [.01, -.03]])
    market.close[2:, 0] /= 2
    growth = trailing(market, 3, window=3, skip=0)
    assert growth[0] == pytest.approx(1.02 * 1.01 * 1.01 - 1)
    assert rank(np.arange(2), growth, width=20).names.tolist() == [1]
    ledger = simulate(market, {0: selected(0)}, 1)
    assert ledger.nav.iloc[2] > ledger.nav.iloc[1]


def test_missing_signal_and_low_adv_are_excluded_and_weight_stays_in_cash():
    market = market_for([[0, 0, 0], [0, 0, 0], [-.1, -.02, -.5], [0, 0, 0]])
    market.adv[:, 2] = 29_000_000
    params = Parameters(2, 0, 3, 2, 30_000_000, 20)
    _, weights = build_weights(market, params)
    assert weights.iloc[2].to_numpy() == pytest.approx([1 / 3, 0, 0])
    market.returns[1, 0] = np.nan
    assert build_weights(market, params)[1].to_numpy().sum() == 0


def test_signal_executes_next_close_and_returns_start_after_fill():
    market = market_for([[0.], [.2], [.1], [.3]])
    ledger = simulate(market, {0: selected(0)}, 1, record=True)
    assert ledger.nav.iloc[0] == 1.
    assert ledger.nav.iloc[1] < 1.
    fill = ledger.trades.iloc[0]
    assert fill.date == market.dates[1]
    assert ledger.nav.iloc[2] / ledger.nav.iloc[1] - 1 == pytest.approx(
        fill.filled * .1 / (ledger.nav.iloc[1] * CAPITAL))
    stamp, transfer = fee_rates(market.dates)
    assert coarse(market, [0], [np.array([0])], 1, stamp, transfer)[0] == pytest.approx(
        ((1 - .0018) * 1.1 * 1.3) ** 63 - 1)


def test_direction_limits_and_unknown_state():
    allowed = np.array([True, True, False, True])
    buy, sell, known = limit_masks(
        np.full(4, 10.), np.array([11., 10., 10., 12.]), np.array([10., 9., 10., 10.]),
        np.full(4, 11.), np.full(4, 9.), allowed)
    assert buy.tolist() == [False, True, False, False]
    assert sell.tolist() == [True, False, False, False]
    assert known.tolist() == [True, True, False, False]


def test_locked_holding_occupies_slot_and_buy_expires():
    market = market_for(np.zeros((8, 2)))
    market.sell[3:6, 0] = False
    ledger = simulate(market, {0: selected(0), 2: selected(1)}, 1, record=True)
    assert ledger.count.tolist() == [0, 1, 1, 1, 1, 1, 0, 0]
    assert ledger.trades.reason.eq("down_limit").sum() == 3
    assert ledger.trades.reason.eq("expired").sum() == 1
    assert (ledger.trades.loc[ledger.trades.reason.isin(["slots", "down_limit"]), "filled"] == 0).all()


def test_adv_is_lagged_and_floor_is_never_relaxed():
    market = market_for(np.zeros((6, 1)))
    market.adv[:3] = 29_000_000
    ledger = simulate(market, {0: selected(0)}, 1, adv_floor=30_000_000, record=True)
    assert ledger.count.eq(0).all()
    assert ledger.trades.reason.tolist() == ["adv_floor"] * 3 + ["expired"]


def test_delist_mark_and_zero_recovery():
    market = replace(market_for([[0.], [0.], [-.2], [np.nan], [0.]]), exits={3: [0]})
    plans = {0: selected(0)}
    marked = simulate(market, plans, 1, record=True)
    lost = simulate(market, plans, 1, recovery=0.)
    assert marked.nav.iloc[3] == pytest.approx(marked.nav.iloc[2])
    assert marked.count.iloc[3] == 0
    assert lost.nav.iloc[3] == pytest.approx(marked.nav.iloc[2] * marked.cash.iloc[2])
    assert marked.trades.iloc[-1].reason == "verified_mark"


def test_weight_conservation_minimum_commission_and_future_perturbation():
    market = market_for([[0.], [0.], [.2], [-.2], [.02]])
    ledger = simulate(market, {0: selected(0)}, 1, record=True)
    assert ledger.cash.min() >= 0
    assert ledger.weights.weight.between(0, 1).all()
    assert fees(np.array([0, 200, 20_000]), .0005, 0).tolist() == [0., 5., 10.]
    when = market.dates[3]
    changed = simulate(perturb(market, when), {0: selected(0)}, 1)
    assert changed.nav.iloc[:3].equals(ledger.nav.iloc[:3])
    assert changed.nav.iloc[3] != ledger.nav.iloc[3]


def test_capacity_audit_counts_empty_day_only_on_holding_side():
    market = market_for(np.zeros((2, 1)))
    market.adv[0, 0] = 0
    ledger = simulate(market, {0: selected(0)}, 1, record=True)
    assert ledger.capacity_diag.order_kind.iloc[1] == "zero"
    yearly, _ = capacity_audit(ledger)
    assert yearly.iloc[0].to_dict() == dict(year=2015, hold_zero=0, hold_unavailable=0,
                                           hold_empty=2, order_zero=0, order_unavailable=0)


def test_full_suspension_window_reports_real_zero_for_hold_and_order():
    market = market_for(np.zeros((22, 1)))
    market.amount[1:] = 0
    market = replace(market, adv=rolling_adv(market.amount))
    ledger = simulate(market, {19: selected(0), 20: selected()}, 1, record=True)
    last = ledger.capacity_diag.iloc[-1]
    assert last.hold_bottleneck_code == last.order_bottleneck_code == market.codes[0]
    assert last.hold_adv == last.order_adv == 0
    assert last.hold_kind == last.order_kind == "zero"
    yearly, _ = capacity_audit(ledger)
    assert yearly.loc[0, ["hold_zero", "order_zero"]].tolist() == [1, 1]


def test_unavailable_adv_is_distinct_from_zero_capacity():
    market = market_for(np.zeros((3, 1)))
    market.adv[1, 0] = np.nan
    ledger = simulate(market, {0: selected(0), 1: selected()}, 1, record=True)
    assert ledger.holding_capacity.iloc[-1] == ledger.order_capacity.iloc[-1] == 0
    yearly, _ = capacity_audit(ledger)
    assert yearly.loc[0, ["hold_unavailable", "order_unavailable"]].tolist() == [1, 1]
    assert yearly.loc[0, ["hold_zero", "order_zero"]].tolist() == [0, 0]


def test_capacity_sensitivity_uses_only_2019_onward_without_changing_full_sample():
    dates = pd.to_datetime(["2018-12-27", "2018-12-28", "2018-12-31",
                            "2019-01-02", "2019-01-03", "2019-01-04"])
    market = replace(market_for(np.zeros((len(dates), 1))), dates=dates)
    market.adv[:, 0] = [100e6, 0, 20e6, 40e6, 80e6, 80e6]
    market.sell[3:, 0] = False
    ledger = simulate(market, {0: selected(0), 2: selected()}, 1, record=True)
    yearly, recent = capacity_audit(ledger)
    weight = ledger.capacity_diag.hold_weight.iloc[-1]
    expected_median = round(PARTICIPATION * 40e6 / weight / 10_000, 1)
    expected_p10 = round(PARTICIPATION * 24e6 / weight / 10_000, 1)
    assert yearly.hold_zero.tolist() == [1, 0]
    assert recent["days"].tolist() == [3, 3]
    assert recent["median"].tolist() == [expected_median, expected_median]
    assert recent["p10"].tolist() == [expected_p10, expected_p10]
    assert ledger.holding_capacity.quantile(.1) == 0


def test_capacity_audit_reports_dates_with_mismatched_ledger_values(capsys):
    market = market_for(np.zeros((2, 1)))
    ledger = simulate(market, {0: selected(0)}, 1, record=True)
    ledger.holding_capacity.iloc[1] *= 1.01
    date = str(market.dates[1].date())
    with pytest.raises(ValueError, match=date):
        capacity_audit(ledger)
    assert date in capsys.readouterr().out
