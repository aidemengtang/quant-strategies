import numpy as np
import pandas as pd
import pytest

from qs.engine import run_backtest
from strategies import etf_momentum


def test_month_end_dates():
    index = pd.bdate_range("2026-01-05", "2026-03-20")

    month_ends = etf_momentum.month_end_dates(index)

    assert len(month_ends) == 3
    expected = pd.DatetimeIndex(["2026-01-30", "2026-02-27", "2026-03-20"])
    pd.testing.assert_index_equal(pd.DatetimeIndex(month_ends), expected)


def test_build_weights_picks_top_n(monkeypatch):
    dates = pd.bdate_range("2025-01-01", periods=300)
    gains = {"D": 0.05, "B": 0.50, "E": 0.05, "A": 1.00, "F": 0.05, "C": 0.20}
    panel = pd.DataFrame(
        {symbol: np.linspace(100, 100 * (1 + gain), len(dates))
         for symbol, gain in gains.items()},
        index=dates,
    )
    monkeypatch.setattr(etf_momentum, "load_panel", lambda symbols, start, end: panel)

    _, weights = etf_momentum.build_weights(
        symbols=list(panel.columns), lookback=250, top_n=3, start="2025-01-01"
    )

    last_weights = weights.iloc[-1]
    assert set(last_weights[last_weights.ne(0)].index) == {"A", "B", "C"}
    np.testing.assert_allclose(
        last_weights[["A", "B", "C"]], 1 / 3, atol=1e-9, rtol=0
    )
    assert last_weights[["D", "E", "F"]].eq(0).all()


def test_absolute_momentum_filter(monkeypatch):
    dates = pd.bdate_range("2025-01-01", periods=300)
    panel = pd.DataFrame(
        {symbol: np.linspace(100, end_price, len(dates))
         for symbol, end_price in zip("ABCDEF", [95, 90, 85, 80, 75, 70])},
        index=dates,
    )
    monkeypatch.setattr(etf_momentum, "load_panel", lambda symbols, start, end: panel)

    _, weights = etf_momentum.build_weights(
        symbols=list(panel.columns),
        lookback=250,
        top_n=3,
        start="2025-01-01",
        min_score=0.0,
    )

    assert not weights.empty
    assert weights.eq(0).to_numpy().all()


def test_engine_no_future_leak():
    dates = pd.bdate_range("2026-01-05", periods=5)
    returns = pd.DataFrame({"A": [0.05, -0.02, 0.30, 0.08, -0.01]}, index=dates)
    weights = pd.DataFrame({"A": [1.0]}, index=dates[[2]])

    nav, _ = run_backtest(returns, weights, cost_rate=0, init_nav=1.0)

    assert nav.iloc[:3].tolist() == [1.0, 1.0, 1.0]
    assert nav.iloc[3] == pytest.approx(1 + returns.loc[dates[3], "A"])


def test_engine_turnover_cost():
    dates = pd.bdate_range("2026-01-05", periods=3)
    returns = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
    weights = pd.DataFrame({"A": [1.0]}, index=dates[[1]])

    nav, _ = run_backtest(returns, weights, cost_rate=0.001, init_nav=1.0)

    assert nav.iloc[1] == pytest.approx(0.999, abs=1e-12, rel=0)


def test_engine_nav_matches_vectorized():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2026-01-05", periods=200)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, size=(200, 3)), index=dates, columns=list("ABC")
    )
    daily_weights = pd.DataFrame(0.0, index=dates, columns=returns.columns)
    daily_weights.iloc[1:] = 1 / 3

    nav, _ = run_backtest(returns, daily_weights.iloc[[1]], cost_rate=0, init_nav=1.0)

    daily_returns = (daily_weights.shift(1, fill_value=0) * returns).sum(axis=1)
    expected_nav = (1 + daily_returns).prod()
    assert abs(nav.iloc[-1] - expected_nav) / abs(expected_nav) < 1e-10
