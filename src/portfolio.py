"""Portfolio analytics shared by the data pipeline and the Streamlit app.

Conventions
-----------
* ``monthly_returns`` is a ``(n_obs, n_assets)`` array or DataFrame of
  monthly simple returns.
* ``expected_returns`` is an annualized vector, i.e. ``12 * monthly mean``.
* CVaR is the empirical Expected Shortfall at ``confidence`` (default 95%)
  and is returned as a positive loss number.
* Risk parity is defined on portfolio variance, not on CVaR. CVaR risk
  contributions can be negative for tail-hedging assets (notably cash), which
  makes the CVaR risk-parity problem prone to corner solutions.
* All optimizations are long-only and fully invested (weights in [0, 1],
  sum to one).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class OptimizationResult:
    """Weights plus convergence information for one special portfolio."""

    weights: np.ndarray
    success: bool
    message: str


def _as_returns(monthly_returns) -> np.ndarray:
    values = np.asarray(monthly_returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("monthly_returns must be a 2-D array or DataFrame")
    return values


def _check_weights(weights, n_assets: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.shape != (n_assets,):
        raise ValueError(
            f"weights must have shape ({n_assets},), got {values.shape}"
        )
    return values


def portfolio_return(weights, expected_returns) -> float:
    """Annualized expected return of a weighted portfolio."""
    er = np.asarray(expected_returns, dtype=float)
    if er.ndim != 1:
        raise ValueError("expected_returns must be a 1-D array or Series")
    w = _check_weights(weights, er.size)
    return float(w @ er)


def portfolio_cvar(weights, monthly_returns, confidence: float = 0.95) -> float:
    """Empirical CVaR (Expected Shortfall) of a long-only portfolio.

    The value is the average loss in the worst ``1 - confidence`` tail of
    monthly portfolio returns, returned as a positive number.
    """
    returns = _as_returns(monthly_returns)
    w = _check_weights(weights, returns.shape[1])
    portfolio_returns = returns @ w
    var = np.quantile(portfolio_returns, 1.0 - confidence)
    tail = portfolio_returns[portfolio_returns <= var]
    if tail.size == 0:
        return 0.0
    return float(-tail.mean())


def portfolio_metrics(
    weights, expected_returns, monthly_returns, confidence: float = 0.95
) -> tuple[float, float]:
    """Return ``(cvar, annualized_expected_return)`` for a portfolio."""
    return (
        portfolio_cvar(weights, monthly_returns, confidence),
        portfolio_return(weights, expected_returns),
    )


def cvar_risk_contributions(
    weights, monthly_returns, confidence: float = 0.95
) -> np.ndarray:
    """Euler allocation of CVaR to the individual assets.

    The contributions sum to :func:`portfolio_cvar` up to numerical error.
    They can be negative for assets that hedge the portfolio tail (for example
    cash). For that reason CVaR risk contributions are kept as a diagnostic and
    are not used to define the primary risk-parity portfolio; see
    :func:`risk_parity_volatility_weights`.
    """
    returns = _as_returns(monthly_returns)
    w = _check_weights(weights, returns.shape[1])
    portfolio_returns = returns @ w
    var = np.quantile(portfolio_returns, 1.0 - confidence)
    tail = returns[portfolio_returns <= var]
    if tail.shape[0] == 0:
        return np.zeros_like(w)
    return w * (-tail.mean(axis=0))


def variance_risk_contributions(weights, monthly_returns) -> np.ndarray:
    """Euler allocation of portfolio variance to the individual assets.

    The contributions ``w_i * (Sigma w)_i`` sum to ``w' Sigma w``. Their
    shares are the standard risk-parity contributions and are always positive
    for a positive-definite covariance matrix with positive weights.
    """
    returns = _as_returns(monthly_returns)
    w = _check_weights(weights, returns.shape[1])
    covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
    return w * (covariance @ w)


def risk_parity_variance_objective(weights, monthly_returns) -> float:
    """Squared deviation of variance risk contributions from equal shares."""
    contributions = variance_risk_contributions(weights, monthly_returns)
    total = contributions.sum()
    if not np.isfinite(total) or total <= 1e-18:
        return 1e6
    target = 1.0 / len(contributions)
    return float(np.sum((contributions / total - target) ** 2))


def risk_parity_volatility_weights(
    monthly_returns,
    *,
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> OptimizationResult:
    """Long-only equal-variance-contribution (volatility) risk parity.

    The weights solve the convex log-barrier problem

        minimize 0.5 * w' Sigma w - (1 / N) * sum(log(w_i)),  w_i > 0.

    The first-order condition is ``w_i * (Sigma w)_i = 1 / N``, i.e. equal
    variance contributions. The solution is unique for a positive-definite
    covariance matrix. The raw solution is normalized to sum to one; this
    leaves the contribution shares unchanged. Unlike CVaR risk parity, every
    asset receives a strictly positive weight.
    """
    returns = _as_returns(monthly_returns)
    n_assets = returns.shape[1]
    if returns.shape[0] < 2:
        raise ValueError("at least two observations are required")
    covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
    scale = float(np.mean(np.diag(covariance)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("covariance matrix must have positive variance")
    covariance = covariance / scale

    target = 1.0 / n_assets

    def objective(weights: np.ndarray) -> float:
        return float(
            0.5 * weights @ covariance @ weights
            - target * np.sum(np.log(weights))
        )

    def gradient(weights: np.ndarray) -> np.ndarray:
        return covariance @ weights - target / weights

    initial = 1.0 / np.sqrt(np.diag(covariance))
    initial = initial / initial.sum()
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=tuple((1e-12, None) for _ in range(n_assets)),
        options={"maxiter": max_iter, "ftol": tol},
    )

    raw = np.asarray(getattr(result, "x", []), dtype=float)
    if (
        raw.shape != (n_assets,)
        or not np.all(np.isfinite(raw))
        or np.any(raw <= 0.0)
        or raw.sum() <= 0.0
    ):
        return OptimizationResult(
            weights=np.full(n_assets, 1.0 / n_assets),
            success=False,
            message=f"risk-parity fallback: {result.message}",
        )

    weights = raw / raw.sum()
    return OptimizationResult(
        weights=weights,
        success=bool(result.success),
        message=str(result.message),
    )


def negative_cvar_sharpe(
    weights,
    expected_returns,
    monthly_returns,
    risk_free_rate: float,
    confidence: float = 0.95,
) -> float:
    """Negative of the CVaR-adjusted Sharpe ratio (for minimization)."""
    cvar, ret = portfolio_metrics(weights, expected_returns, monthly_returns, confidence)
    if cvar <= 1e-12:
        return np.inf
    return float(-(ret - risk_free_rate) / cvar)


def _fallback_weights(result, n_assets: int) -> np.ndarray:
    """Return feasible weights even if the optimizer did not converge."""
    raw = getattr(result, "x", None)
    if raw is None:
        return np.full(n_assets, 1.0 / n_assets)
    weights = np.asarray(raw, dtype=float)
    if weights.shape != (n_assets,) or not np.all(np.isfinite(weights)):
        return np.full(n_assets, 1.0 / n_assets)
    weights = np.clip(weights, 0.0, 1.0)
    total = weights.sum()
    if total <= 0.0:
        return np.full(n_assets, 1.0 / n_assets)
    return weights / total


def _run_slsqp(objective, args, x0, bounds, constraints, options):
    return minimize(
        objective,
        x0,
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=options,
    )


def _select_best_result(results, n_assets: int) -> OptimizationResult:
    """Pick the best objective value and report whether any start converged."""
    valid = [
        result
        for result in results
        if getattr(result, "x", None) is not None
        and np.all(np.isfinite(result.x))
        and np.isfinite(result.fun)
    ]
    best = min(valid, key=lambda result: result.fun) if valid else results[0]
    return OptimizationResult(
        weights=_fallback_weights(best, n_assets),
        success=any(bool(result.success) for result in results),
        message=str(best.message),
    )


def optimize_special_portfolios(
    expected_returns,
    monthly_returns,
    risk_free_rate: float,
    confidence: float = 0.95,
) -> dict[str, OptimizationResult]:
    """Compute the min-risk, CVaR-tangency and risk-parity portfolios.

    The minimum-risk portfolio and the tangency portfolio use CVaR. The
    risk-parity portfolio equalizes **variance** contributions (volatility
    risk parity), because CVaR contributions can be negative for tail-hedging
    assets and produce unstable corner solutions. SLSQP is used with
    long-only bounds. The tangency objective is non-smooth (CVaR is
    piecewise linear), so it is started from several feasible points; this
    removes the occasional "positive directional derivative" failure seen
    with a single equal-weight start. A failed optimization is reported in
    ``OptimizationResult.success`` and a feasible fallback is still returned,
    so callers never receive NaNs or negative weights.
    """
    er = np.asarray(expected_returns, dtype=float)
    returns = _as_returns(monthly_returns)
    n_assets = er.size
    if returns.shape[1] != n_assets:
        raise ValueError("expected_returns and monthly_returns disagree on assets")

    initial_weights = np.full(n_assets, 1.0 / n_assets)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    options = {"maxiter": 500, "ftol": 1e-10}

    min_risk_result = _run_slsqp(
        portfolio_cvar,
        (returns, confidence),
        initial_weights,
        bounds,
        constraints,
        options,
    )
    min_risk = OptimizationResult(
        weights=_fallback_weights(min_risk_result, n_assets),
        success=bool(min_risk_result.success),
        message=str(min_risk_result.message),
    )

    risk_parity = risk_parity_volatility_weights(returns)

    max_return_weights = np.zeros(n_assets)
    max_return_weights[int(np.argmax(er))] = 1.0
    tangency_starts = [
        initial_weights,
        min_risk.weights,
        risk_parity.weights,
        max_return_weights,
    ]
    tangency_results = [
        _run_slsqp(
            negative_cvar_sharpe,
            (er, returns, risk_free_rate, confidence),
            start,
            bounds,
            constraints,
            options,
        )
        for start in tangency_starts
    ]
    tangency = _select_best_result(tangency_results, n_assets)

    return {
        "min_risk": min_risk,
        "tangency": tangency,
        "risk_parity": risk_parity,
    }


def efficient_frontier(
    expected_returns,
    monthly_returns,
    n_points: int = 20,
    confidence: float = 0.95,
    min_return: float | None = None,
) -> pd.DataFrame:
    """CVaR-return efficient frontier for a long-only portfolio.

    For each target annualized return the portfolio CVaR is minimized subject
    to full investment and the return target. Targets below the minimum-risk
    portfolio's return are deliberately excluded: they are dominated by the
    minimum-risk portfolio and would make the frontier bend backwards.

    Failed points are skipped. The previous solution is used as a warm start
    for the next target.
    """
    er = np.asarray(expected_returns, dtype=float)
    returns = _as_returns(monthly_returns)
    n_assets = er.size
    if returns.shape[1] != n_assets:
        raise ValueError("expected_returns and monthly_returns disagree on assets")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    x0 = np.full(n_assets, 1.0 / n_assets)
    options = {"maxiter": 500, "ftol": 1e-10}

    if min_return is None:
        min_result = minimize(
            portfolio_cvar,
            x0,
            args=(returns, confidence),
            method="SLSQP",
            bounds=bounds,
            constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},),
            options=options,
        )
        if min_result.success and np.isfinite(min_result.fun):
            min_weights = _fallback_weights(min_result, n_assets)
            min_return = portfolio_return(min_weights, er)
        else:
            min_return = float(er.min())

    lower = max(float(min_return), float(er.min()))
    upper = float(er.max())
    if lower >= upper - 1e-12:
        return pd.DataFrame(columns=["cvar", "ret"])
    target_returns = np.linspace(lower, upper, n_points)

    results: list[dict[str, float]] = []
    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {
                "type": "eq",
                "fun": lambda w, t=target: portfolio_return(w, er) - t,
            },
        )
        result = minimize(
            portfolio_cvar,
            x0,
            args=(returns, confidence),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
        if result.success and np.isfinite(result.fun):
            results.append({"cvar": float(result.fun), "ret": float(target)})
            x0 = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
            total = x0.sum()
            if total > 0.0:
                x0 = x0 / total

    if not results:
        return pd.DataFrame(columns=["cvar", "ret"])

    frontier = (
        pd.DataFrame(results)
        .sort_values("ret")
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Remove dominated points. A point is dominated if a higher-return point
    # has lower or equal CVaR; keeping it would make the plotted frontier bend
    # backwards. The filter keeps the lower envelope of CVaR over return.
    keep: list[int] = []
    min_cvar_to_the_right = np.inf
    for index in range(len(frontier) - 1, -1, -1):
        cvar = float(frontier.loc[index, "cvar"])
        if cvar < min_cvar_to_the_right - 1e-12:
            keep.append(index)
            min_cvar_to_the_right = cvar
    frontier = frontier.loc[sorted(keep)].reset_index(drop=True)
    return frontier.sort_values("cvar").reset_index(drop=True)


def annualized_mean(monthly_returns) -> np.ndarray:
    """Annualized sample mean (x12) of monthly returns."""
    values = np.asarray(monthly_returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("monthly_returns must be a 2-D array or DataFrame")
    return values.mean(axis=0) * 12.0


def annualized_mean_covariance(monthly_returns) -> np.ndarray:
    """Sampling covariance of the annualized sample mean.

    For monthly returns with sample covariance ``S`` and ``n`` observations,

        Cov(12 * mean) = 144 * S / n.

    This is the parametric reference used in tests and documentation.
    """
    values = _as_returns(monthly_returns)
    n_obs = values.shape[0]
    if n_obs < 2:
        raise ValueError("at least two observations are required")
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    return covariance / n_obs * (12.0 ** 2)


def draw_bootstrap_sample(monthly_returns, rng: np.random.Generator) -> pd.DataFrame:
    """Draw one i.i.d. row bootstrap sample with a fresh RangeIndex.

    A RangeIndex avoids duplicate-label pitfalls in later boolean indexing.
    The time ordering carries no information in an i.i.d. bootstrap, so the
    original dates are intentionally not preserved.
    """
    frame = pd.DataFrame(monthly_returns)
    n_obs = len(frame)
    seed = int(rng.integers(0, 2**32 - 1))
    return frame.sample(n=n_obs, replace=True, random_state=seed).reset_index(drop=True)
