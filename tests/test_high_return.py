import numpy as np
import pandas as pd
import pytest

from qs.engine import run_backtest
from research.placebo import randomized_weights
from research.study import turnover
from strategies import _rotation, candidate_a, candidate_b, high_return


@pytest.fixture
def panel():
    dates = pd.bdate_range("2015-01-05", periods=330)
    days = np.arange(len(dates))
    return pd.DataFrame({
        "weak": 100 * np.exp(0.0003 * days + 0.001 * np.sin(days)),
        "strong": 100 * np.exp(0.0020 * days + 0.003 * np.sin(days)),
        "medium": 100 * np.exp(0.0010 * days + 0.002 * np.sin(days)),
        "late": np.where(days < 300, np.nan, 100 + days),
        "missing": np.nan,
    }, index=dates)


@pytest.fixture(params=[high_return, candidate_b], ids=["high", "candidate_b"])
def strategy(request):
    return request.param


def use_prices(monkeypatch, strategy, panel, **kwargs):
    monkeypatch.setattr(_rotation, "load_panel", lambda symbols, start, end: panel)
    return strategy.build_weights(symbols=list(panel.columns), **kwargs)


def test_future_prices_do_not_change_prior_weights(monkeypatch, strategy, panel):
    _, original = use_prices(monkeypatch, strategy, panel)
    cutoff = original.index[-2]
    assert original.loc[cutoff].gt(0).any()
    changed = panel.copy()
    changed.loc[changed.index > cutoff] *= 1.5
    _, shocked = use_prices(monkeypatch, strategy, changed)
    pd.testing.assert_frame_equal(original.loc[:cutoff], shocked.loc[:cutoff])


def test_weights_have_no_leverage_or_prelisting_positions(monkeypatch, strategy, panel):
    _, weights = use_prices(monkeypatch, strategy, panel)
    assert weights.gt(0).any().any()
    assert weights.ge(0).all().all()
    assert weights.sum(axis=1).le(1 + 1e-9).all()
    assert weights[["late", "missing"]].eq(0).all().all()


def test_cost_monotonicity(monkeypatch, strategy, panel):
    returns, weights = use_prices(monkeypatch, strategy, panel)
    terminal = [
        run_backtest(returns, weights, cost_rate=cost)[0].iloc[-1]
        for cost in (0.0005, 0.001, 0.002)
    ]
    assert terminal[0] > terminal[1] > terminal[2]


def test_nav_matches_vectorized(monkeypatch, strategy, panel):
    returns, weights = use_prices(monkeypatch, strategy, panel)
    nav, _ = run_backtest(returns, weights, cost_rate=0)
    held = weights.reindex(returns.index).ffill().shift(1).fillna(0.0)
    expected = (1 + (held * returns.fillna(0)).sum(axis=1)).prod()
    assert abs(nav.iloc[-1] - expected) / abs(expected) < 1e-9


def test_short_history_single_asset_and_nan_columns(monkeypatch, strategy, panel):
    short = panel[["strong", "missing"]].loc["2016-01-01":].iloc[:30]
    returns, weights = use_prices(monkeypatch, strategy, short)
    assert not weights.empty
    assert weights.eq(0).all().all()
    assert run_backtest(returns, weights)[0].eq(1.0).all()

    returns, weights = use_prices(monkeypatch, strategy, panel[["strong"]])
    assert weights["strong"].iloc[-1] == pytest.approx(1.0)
    assert not run_backtest(returns, weights)[0].empty


def test_selects_strongest_trailing_momentum(monkeypatch, strategy, panel):
    _, weights = use_prices(monkeypatch, strategy, panel, top_n=1)
    active = weights.loc[weights.sum(axis=1).gt(0)]
    assert not active.empty
    assert active["strong"].eq(1).all()


def test_candidate_a_uses_inverse_volatility(monkeypatch, panel):
    _, weights = use_prices(monkeypatch, candidate_a, panel)
    selected = ["weak", "medium", "strong"]
    volatility = panel[selected].pct_change().rolling(60).std().loc[weights.index[-1]]
    inverse = 1 / volatility
    pd.testing.assert_series_equal(
        weights.iloc[-1][selected], inverse / inverse.sum(), check_names=False
    )


def test_candidate_b_risk_off_uses_bond_sleeve(monkeypatch, panel):
    risk_off = pd.DataFrame({
        "strong": panel["strong"],
        "511010": panel["weak"],
        "down1": 1 / panel["weak"],
        "down2": 1 / panel["medium"],
    })
    _, weights = use_prices(monkeypatch, candidate_b, risk_off)
    assert weights.iloc[-1].to_dict() == {
        "strong": 0.7, "511010": 0.3, "down1": 0.0, "down2": 0.0
    }


def test_candidate_b_rank_buffer_retains_nearby_incumbent():
    scores = pd.Series({"leader": 0.3, "challenger": 0.2, "incumbent": 0.1})
    assert candidate_b.buffered_picks(scores, {"incumbent"}, 2) == [
        "incumbent", "leader"
    ]


def test_placebos_match_turnover_and_respect_listing_dates():
    dates = pd.bdate_range("2020-01-02", periods=3)
    weights = pd.DataFrame(
        [[1.0, 0.0, 0.0], [0.4, 0.6, 0.0], [0.4, 0.0, 0.6]],
        index=dates, columns=["A", "B", "C"],
    )
    available = pd.DataFrame(True, index=dates, columns=weights.columns)
    available.loc[dates[0], "C"] = False
    trials = list(randomized_weights(weights, available, trials=20))
    assert any(not trial.equals(weights) for trial in trials)
    for trial in trials:
        assert (trial.eq(0) | available).all().all()
        pd.testing.assert_series_equal(trial.gt(0).sum(axis=1), weights.gt(0).sum(axis=1))
        pd.testing.assert_series_equal(turnover(trial), turnover(weights))


def test_capacity_audit_classifies_zero_unavailable_empty_and_2019_sample():
    from research.high_return_capacity import audit, capacity_audit

    dates = pd.bdate_range("2018-11-01", periods=70)
    amount = pd.DataFrame(100e6, index=dates, columns=["ETF"])
    amount.iloc[25:45] = 0.
    amount.iloc[50] = np.nan
    # The non-trading cash anchor must not become an extra empty capacity day.
    returns = pd.DataFrame(0., index=dates.insert(0, dates[0] - pd.Timedelta(days=1)), columns=amount.columns)
    targets = pd.DataFrame({"ETF": [1., .5, .25, 0.]}, index=dates[[20, 45, 55, 67]])
    nav, gross, ledger = audit(returns, targets, amount)
    yearly, recent = capacity_audit(ledger)
    totals = yearly.iloc[:, 1:].sum().to_dict()
    assert totals == dict(hold_zero=1, hold_unavailable=16, hold_empty=23,
                          order_zero=1, order_unavailable=1)
    assert ledger.capacity_diag.hold_adv.iloc[45] == 0
    assert np.isnan(ledger.capacity_diag.hold_adv.iloc[51])
    valid = (dates >= "2019-01-01") & ledger.count.gt(0) & ledger.capacity_diag.hold_kind.ne(
        "unavailable").to_numpy()
    assert recent.loc["holding", "days"] == valid.sum()
    assert recent.loc["holding", "median"] == round(ledger.holding_capacity[valid].median() / 10_000, 1)
    expected, _ = run_backtest(returns, targets)
    pd.testing.assert_series_equal(nav, expected)
    assert gross.eq(1.).all()


def test_three_factor_attribution_matches_micro_cap_formula_and_loading_signs():
    from types import SimpleNamespace

    from qs.micro_cap_data import adjusted_prices
    from research.capacity_report import three_factors
    from research.high_return_capacity import attribution
    from research.micro_cap_study import attribution as micro_attribution

    dates = pd.bdate_range("2014-11-03", periods=100)
    rng = np.random.default_rng(17)
    returns = rng.normal(.0003, .02, (len(dates), 30))
    market = SimpleNamespace(dates=dates, returns=returns, adjusted=adjusted_prices(returns),
                             cap=np.broadcast_to(np.arange(1, 31), returns.shape).astype(float))
    x = three_factors(market)
    expected_beta = np.array([.8, -.4, .3])
    gross = pd.Series(np.exp(x.to_numpy() @ expected_beta), index=x.index)
    nav = (gross * np.exp(-.0001)).cumprod()
    result = attribution(nav, gross, x, is_end="2020-12-31")
    reference = micro_attribution(market, SimpleNamespace(nav=nav, gross=gross))
    np.testing.assert_allclose(result.betas, expected_beta, atol=1e-12)
    np.testing.assert_allclose(result.components, reference.components, atol=1e-12)
    np.testing.assert_allclose(result.factor_sums, reference.factor_sums, atol=1e-12)
    assert np.sign(result.betas).tolist() == [1, -1, 1]
    assert result.components.sum() == pytest.approx(np.log(nav.iloc[-1]))


def test_high_return_factor_loadings_are_frozen_at_is_boundary():
    from research.high_return_capacity import attribution

    dates = pd.bdate_range("2021-12-01", periods=60)
    x = pd.DataFrame(np.random.default_rng(9).normal(0, .01, (60, 3)), index=dates)
    log_gross = x.to_numpy() @ np.array([1., -.5, .25])
    log_gross[dates > "2021-12-31"] = x.loc["2022-01-01":].to_numpy() @ np.array([-2., 3., -4.])
    gross = pd.Series(np.exp(log_gross), index=dates)
    result = attribution(gross.cumprod(), gross, x, is_end="2021-12-31")
    np.testing.assert_allclose(result.betas, [1., -.5, .25], atol=1e-12)
    np.testing.assert_allclose(result.factor_sums, x.sum().to_numpy())


def test_capacity_replay_checks_nav_values_independently_of_csv_date_metadata(tmp_path):
    from research.capacity_report import verify_nav

    nav = pd.Series([1., 1.1], index=pd.bdate_range("2020-01-01", periods=2), name="nav")
    path = tmp_path / "nav.csv"
    nav.to_csv(path, index_label="date")
    verify_nav(nav, path)
    changed = nav.copy()
    changed.iloc[-1] += .01
    with pytest.raises(AssertionError):
        verify_nav(changed, path)
