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
