from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qs.engine import run_backtest
from qs.lowvol_momentum_data import Distribution, Market, _membership, execution_masks
from qs.lowvol_momentum_execution import simulate
from research.lowvol_momentum_study import HOLDINGS, WINDOWS, plateau, same_turnover
from strategies import lowvol_momentum as strategy


def market_fixture(days=8, assets=2):
    dates = pd.bdate_range("2020-01-01", periods=days)
    codes = pd.Index([f"S{i}" for i in range(assets)])
    close = np.full((days, assets), 10.0)
    amount = np.full_like(close, 100_000_000)
    yes = np.ones_like(close, dtype=bool)
    no = ~yes
    return Market(
        dates, codes, close, amount, np.zeros_like(close), yes, yes.copy(), yes.copy(),
        no, no.copy(), dates[~dates.to_period("M").duplicated(keep="last")], {}, [],
        pd.DataFrame(),
    )


def run(market, orders, commission=0):
    return simulate(
        market, orders, market.amount, start=market.dates[0],
        commission=commission, slip=0, levies=False, record=True,
    )


def test_future_data_do_not_change_past_orders_or_positions():
    market = market_fixture(320, 28)
    market.amount[:] = np.linspace(30_000_000, 90_000_000, 28)
    market.returns[:] = np.linspace(0.0001, 0.0028, 28)
    market.close[:] *= np.cumprod(1 + market.returns, axis=0)
    volume = strategy.average_volume(market)
    cutoff = market.month_ends[-3]
    day = market.dates.get_loc(cutoff)
    original = strategy.monthly_ranks(market, 126, volume, market.dates[0])
    assert original[day].size > 0
    changed = replace(market, close=market.close.copy(), returns=market.returns.copy())
    changed.close[day + 1:] *= 1.5
    changed.returns[day + 1:] = -0.08
    shocked = strategy.monthly_ranks(changed, 126, volume, market.dates[0])
    for signal in original:
        if signal <= day:
            np.testing.assert_array_equal(original[signal], shocked[signal])
    before = run(market, strategy.target_orders(original, 25)).weights
    after = run(changed, strategy.target_orders(shocked, 25)).weights
    pd.testing.assert_frame_equal(
        before.loc[:cutoff].reindex(columns=market.codes, fill_value=0),
        after.loc[:cutoff].reindex(columns=market.codes, fill_value=0),
    )


def test_order_executes_next_day_and_position_earns_after_fill():
    market = market_fixture()
    market.close[:, 0] = [10, 11, 12, 12, 12, 12, 12, 12]
    result = run(market, {0: {0: 0.04}})
    assert result.trades.iloc[0]["date"] == market.dates[1]
    assert result.weights.iloc[0, 0] == 0
    assert result.nav.iloc[:3].tolist() == [1.0, 1.0, 1.004]


def test_allocation_is_nonnegative_cash_backed_and_capped_per_name():
    market = market_fixture(assets=28)
    result = run(market, {0: dict.fromkeys(range(28), 0.04)})
    assert result.weights.ge(0).all().all()
    assert result.weights.sum(axis=1).le(1 + 1e-9).all()
    assert result.max_weight <= 0.08
    assert result.weights.iloc[-1].gt(0).any()


def test_transaction_costs_reduce_terminal_nav_monotonically():
    market = market_fixture()
    orders = {0: {0: 0.04}, 3: {}}
    low, medium, high = (run(market, orders, rate).nav.iloc[-1] for rate in (0.0005, 0.001, 0.002))
    assert low > medium > high
    trades = run(market, orders, 0.0005).trades
    assert trades["fee"].sum() == pytest.approx(200_000 * 0.001)


def test_vectorized_and_engine_agree_with_cash_ledger_without_fees():
    market = market_fixture()
    market.close[:, 0] = np.linspace(10, 14, len(market.dates))
    market.close[:, 1] = np.linspace(10, 8, len(market.dates))
    result = run(market, {0: {0: 0.04}, 3: {1: 0.04}})
    gross = 1 + (result.weights.shift(1, fill_value=0) * result.returns).sum(axis=1)
    nav, _ = run_backtest(result.returns, result.weights, cost_rate=0)
    np.testing.assert_allclose(result.nav, gross.cumprod(), rtol=1e-9)
    np.testing.assert_allclose(result.nav, nav, rtol=1e-9)


def test_delisting_uses_last_verifiable_price_at_exit_and_zero_recovery_is_lower():
    market = market_fixture()
    market.close[4:, 0] = np.nan
    market.exits[5] = [(0, 7.5)]
    market.buy[4:, 0] = market.sell[4:, 0] = False
    result = run(market, {0: {0: 0.04}})
    assert result.nav.iloc[4] == 1
    assert result.nav.iloc[5] == pytest.approx(0.99)
    assert result.weights.iloc[5, 0] == 0
    assert result.trades["side"].tolist() == [1]
    zero = simulate(
        market, {0: {0: 0.04}}, market.amount, start=market.dates[0],
        commission=0, slip=0, levies=False, zero_recovery=True,
    )
    assert zero.nav.iloc[5] == pytest.approx(0.96)


def test_directional_price_limits_and_unknown_status_block_fills():
    reference = np.full(5, 10.0)
    high = np.array([11.0, 10.2, 10.4, 10.4, 10.4])
    low = np.array([9.8, 9.0, 9.5, 9.5, 9.8])
    risk = np.array([False, False, True, True, False])
    verified = np.array([True, True, True, True, False])
    dates = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-01", "2026-07-06", "2020-01-01"])
    buy, sell = execution_masks(reference, high, low, reference, risk, dates, verified)
    assert buy.tolist() == [False, True, False, False, False]
    assert sell.tolist() == [True, False, False, True, False]


def test_suspended_positions_cannot_trade_and_buy_orders_expire():
    market = market_fixture()
    market.close[2:6, 0] = np.nan
    market.buy[2:6, 0] = market.sell[2:6, 0] = False
    market.buy[1:5, 1] = False
    result = run(market, {0: {0: 0.04, 1: 0.04}, 2: {}})
    assert result.trades["code"].tolist() == ["S0", "S0"]
    assert result.trades["date"].tolist() == [market.dates[1], market.dates[6]]
    assert result.weights.loc[market.dates[5], "S0"] == 0.04
    assert result.weights.loc[market.dates[6], "S0"] == 0


def test_ipo_age_and_sme_effective_membership():
    dates = pd.bdate_range("2020-01-01", periods=540)
    rows = pd.DataFrame({"code": ["002001"], "listed": [dates[0]], "delisted": [pd.NaT]})
    listed, mature, exits = _membership(rows, dates)
    assert not listed[dates < pd.Timestamp("2021-04-06")].any()
    assert mature[dates.get_loc(pd.Timestamp("2021-04-06"))]
    assert exits == []
    rows.loc[0, "code"] = "000001"
    _, ordinary_mature, _ = _membership(rows, dates)
    assert not ordinary_mature[:249].any()
    assert ordinary_mature[249]


def test_empty_and_single_asset_keep_unused_slots_in_cash():
    market = market_fixture(assets=1)
    empty = run(market, strategy.target_orders({0: np.array([], dtype=int)}, 25))
    assert empty.nav.eq(1).all()
    assert empty.weights.empty
    single = run(market, strategy.target_orders({0: np.array([0])}, 25))
    assert single.weights.iloc[-1, 0] == pytest.approx(0.04)


@pytest.mark.parametrize("lookback,top_n", [(83, 25), (169, 25), (126, 14), (126, 36), (126.5, 25)])
def test_parameter_bounds(lookback, top_n):
    with pytest.raises(ValueError):
        strategy.validate_parameters(lookback, top_n)


def test_dividend_claim_is_marked_at_ex_date_without_anticipating_it():
    market = market_fixture()
    market.close[3:, 0] = 9
    market.distributions.append(Distribution(0, 2, 3, 5, 5, 1.0, 0.0))
    result = run(market, {0: {0: 0.04}})
    assert result.nav.eq(1).all()
    assert result.weights.iloc[3, 0] == pytest.approx(0.036)
    assert result.returns.iloc[3, 0] == pytest.approx(0.0)


def test_plateau_rejects_isolated_winner_and_turnover_must_match():
    grid = pd.DataFrame(
        [(window, count, 0.13 if (window, count) == (126, 25) else 0.08, 0.07)
         for window in WINDOWS for count in HOLDINGS],
        columns=["lookback", "top_n", "cagr", "max_weight"],
    )
    assert plateau(grid) == ((126, 25), 1 / 25, False)
    assert same_turnover(pd.Series([0.0, 0.20]), pd.Series([0.0, 0.20009]))
    assert not same_turnover(pd.Series([0.0, 0.20]), pd.Series([0.0, 0.20011]))
