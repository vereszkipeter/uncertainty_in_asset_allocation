"""Unit tests for the portfolio analytics."""

import numpy as np
import pandas as pd
import pytest

from portfolio import (
    annualized_mean,
    annualized_mean_covariance,
    cvar_risk_contributions,
    draw_bootstrap_sample,
    efficient_frontier,
    optimize_special_portfolios,
    portfolio_cvar,
    portfolio_return,
    risk_parity_cvar_objective,
)


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """Correlated monthly returns with a known covariance structure."""
    rng = np.random.default_rng(123)
    means = np.array([0.002, 0.003, 0.004, 0.005])
    vols = np.array([0.010, 0.020, 0.025, 0.040])
    corr = np.array(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.2, 1.0, 0.6, 0.1],
            [0.1, 0.6, 1.0, 0.2],
            [0.0, 0.1, 0.2, 1.0],
        ]
    )
    covariance = np.outer(vols, vols) * corr
    values = rng.multivariate_normal(means, covariance, size=600)
    return pd.DataFrame(
        values,
        columns=["A", "B", "C", "D"],
        index=pd.date_range("2015-01-31", periods=600, freq="ME"),
    )


def test_portfolio_return() -> None:
    weights = np.array([0.5, 0.5, 0.0, 0.0])
    expected = np.array([0.10, 0.20, 0.30, 0.40])
    assert portfolio_return(weights, expected) == pytest.approx(0.15)


def test_portfolio_cvar_is_positive_and_at_least_var(synthetic_returns) -> None:
    weights = np.full(4, 0.25)
    cvar = portfolio_cvar(weights, synthetic_returns, confidence=0.95)
    portfolio_returns = synthetic_returns.to_numpy() @ weights
    var_loss = -np.quantile(portfolio_returns, 0.05)
    assert cvar > 0
    assert cvar >= var_loss - 1e-12


def test_cvar_risk_contributions_sum_to_cvar(synthetic_returns) -> None:
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    cvar = portfolio_cvar(weights, synthetic_returns, confidence=0.95)
    contributions = cvar_risk_contributions(
        weights, synthetic_returns, confidence=0.95
    )
    assert contributions.sum() == pytest.approx(cvar, rel=1e-10)


def test_risk_parity_equalizes_contributions(synthetic_returns) -> None:
    result = optimize_special_portfolios(
        annualized_mean(synthetic_returns),
        synthetic_returns,
        risk_free_rate=0.02,
        confidence=0.95,
    )
    weights = result["risk_parity"].weights
    contributions = cvar_risk_contributions(
        weights, synthetic_returns, confidence=0.95
    )
    relative = contributions / contributions.sum()
    assert np.max(np.abs(relative - 0.25)) < 0.02
    assert risk_parity_cvar_objective(
        weights, synthetic_returns, confidence=0.95
    ) < 1e-4


def test_special_portfolios_are_feasible(synthetic_returns) -> None:
    expected = annualized_mean(synthetic_returns)
    result = optimize_special_portfolios(
        expected, synthetic_returns, risk_free_rate=0.02, confidence=0.95
    )
    assert set(result) == {"min_risk", "tangency", "risk_parity"}
    for item in result.values():
        assert item.weights.shape == (4,)
        assert np.all(item.weights >= -1e-12)
        assert np.all(item.weights <= 1.0 + 1e-12)
        assert item.weights.sum() == pytest.approx(1.0)


def test_min_risk_beats_equal_weights(synthetic_returns) -> None:
    expected = annualized_mean(synthetic_returns)
    result = optimize_special_portfolios(
        expected, synthetic_returns, risk_free_rate=0.02, confidence=0.95
    )
    equal_weights = np.full(4, 0.25)
    assert portfolio_cvar(
        result["min_risk"].weights, synthetic_returns, confidence=0.95
    ) <= portfolio_cvar(equal_weights, synthetic_returns, confidence=0.95) + 1e-10


def test_efficient_frontier_is_monotone(synthetic_returns) -> None:
    expected = annualized_mean(synthetic_returns)
    frontier = efficient_frontier(
        expected, synthetic_returns, n_points=10, confidence=0.95
    )
    assert not frontier.empty
    assert list(frontier.columns) == ["cvar", "ret"]
    assert frontier["ret"].between(expected.min(), expected.max()).all()
    assert np.all(np.diff(frontier["cvar"]) >= -1e-10)
    assert np.all(np.diff(frontier["ret"]) >= -1e-8)


def test_annualized_mean_covariance_formula(synthetic_returns) -> None:
    values = synthetic_returns.to_numpy()
    expected = np.cov(values, rowvar=False, ddof=1) / len(values) * 144.0
    actual = annualized_mean_covariance(synthetic_returns)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


def test_bootstrap_mean_covariance_matches_theory(synthetic_returns) -> None:
    """The joint bootstrap must recover the 144 * Sigma / n scaling.

    The exact finite-sample bootstrap covariance is (n - 1) / n times the
    usual sample-mean covariance, so we compare against that reference.
    """
    n_simulations = 3000
    rng = np.random.default_rng(7)
    draws = np.empty((n_simulations, synthetic_returns.shape[1]))
    for index in range(n_simulations):
        sample = draw_bootstrap_sample(synthetic_returns, rng)
        draws[index] = annualized_mean(sample)

    n_obs = len(synthetic_returns)
    theory = annualized_mean_covariance(synthetic_returns) * (n_obs - 1) / n_obs
    empirical = np.cov(draws, rowvar=False, ddof=1)

    # Variances are the stable part of the covariance matrix. Off-diagonal
    # entries are much smaller than the diagonal and therefore have a large
    # relative sampling error even with 3,000 bootstrap draws.
    np.testing.assert_allclose(
        np.diag(empirical), np.diag(theory), rtol=0.10, atol=1e-8
    )

    # A diversified portfolio aggregates the full covariance matrix and is a
    # more stable check of the joint scaling.
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    empirical_variance = weights @ empirical @ weights
    theory_variance = weights @ theory @ weights
    assert empirical_variance == pytest.approx(theory_variance, rel=0.10)


def test_bootstrap_sample_has_range_index(synthetic_returns) -> None:
    rng = np.random.default_rng(0)
    sample = draw_bootstrap_sample(synthetic_returns, rng)
    assert len(sample) == len(synthetic_returns)
    assert isinstance(sample.index, pd.RangeIndex)
    assert list(sample.columns) == list(synthetic_returns.columns)


def test_invalid_weight_shape_raises(synthetic_returns) -> None:
    with pytest.raises(ValueError):
        portfolio_cvar(np.array([0.5, 0.5]), synthetic_returns)
    with pytest.raises(ValueError):
        portfolio_return(np.array([0.5, 0.5]), np.array([0.1, 0.2, 0.3]))
