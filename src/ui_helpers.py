"""Small presentation helpers shared by the Streamlit app and its tests.

Keeping these functions out of ``app.py`` makes the table construction,
tooltip formatting and chart annotations testable without starting Streamlit.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from portfolio import portfolio_metrics

PORTFOLIO_LABELS = {
    "user": "Az Ön portfóliója",
    "min_risk": "Min Risk",
    "tangency": "Tangency",
    "risk_parity": "Risk Parity",
}
PORTFOLIO_ORDER = ("user", "min_risk", "tangency", "risk_parity")

_TICKER_PATTERN = re.compile(r"\(([^)]+)\)\s*$")


def ticker_label(asset_name: str) -> str:
    """Return the ticker in an asset label, e.g. ``USD Készpénz (BIL)`` -> BIL."""
    match = _TICKER_PATTERN.search(asset_name)
    if match:
        return match.group(1)
    return asset_name


def format_weights_for_tooltip(weights, asset_names) -> str:
    """Format portfolio weights as an HTML block for Plotly tooltips."""
    header = "<br><b>Portfólió súlyok:</b><br>"
    lines = [f"{name}: {weight:.2%}" for name, weight in zip(asset_names, weights)]
    return header + "<br>".join(lines)


def to_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#RRGGBB`` to a Plotly ``rgba(...)`` string."""
    return (
        f"rgba({int(hex_color[1:3], 16)}, {int(hex_color[3:5], 16)}, "
        f"{int(hex_color[5:7], 16)}, {alpha})"
    )


def cvar_at_return(frontier: pd.DataFrame, target_return: float) -> float:
    """Linearly interpolate the frontier CVaR at a target annualized return.

    ``numpy.interp`` clamps outside the frontier's return range. Callers that
    need to know whether the target is inside the range should compare it with
    the frontier's minimum and maximum return first.
    """
    if frontier is None or len(frontier) == 0:
        return float("nan")
    if not {"cvar", "ret"}.issubset(frontier.columns):
        raise ValueError("frontier must contain 'cvar' and 'ret' columns")

    ordered = frontier.sort_values("ret")
    returns = ordered["ret"].to_numpy(dtype=float)
    cvar = ordered["cvar"].to_numpy(dtype=float)
    return float(np.interp(float(target_return), returns, cvar))


def cvar_gap_pp(
    user_cvar: float,
    user_return: float,
    frontier: pd.DataFrame,
) -> float:
    """CVaR gap in percentage points between the user and the frontier.

    A positive value means that the user's portfolio carries more monthly CVaR
    than the frontier portfolio with the same expected return.
    """
    frontier_cvar = cvar_at_return(frontier, user_return)
    return float((user_cvar - frontier_cvar) * 100.0)


def frontier_insight(
    user_cvar: float,
    user_return: float,
    frontier: pd.DataFrame,
) -> str:
    """Human-readable comparison of the user's point with the frontier."""
    if frontier is None or len(frontier) == 0:
        return "A hatékony front nem számítható ki a jelenlegi adatokból."

    minimum_return = float(frontier["ret"].min())
    maximum_return = float(frontier["ret"].max())
    if user_return < minimum_return or user_return > maximum_return:
        return (
            "Az Ön portfóliója a hatékony front hozamtartományán kívül esik, "
            "ezért az azonos hozamhoz tartozó CVaR-rés nem értelmezhető."
        )

    gap = cvar_gap_pp(user_cvar, user_return, frontier)
    if gap > 0.05:
        return (
            f"Az Ön portfóliója {gap:.2f} százalékponttal magasabb havi CVaR-t "
            "vállal, mint a hatékony front azonos évesített hozam mellett."
        )
    if gap > -0.05:
        return (
            "Az Ön portfóliója gyakorlatilag a hatékony fronton van: az azonos "
            "hozamhoz tartozó CVaR-rés a numerikus pontosság határán belül van."
        )
    return (
        f"Az Ön portfóliója {abs(gap):.2f} százalékponttal alacsonyabb havi "
        "CVaR-t mutat, mint a front azonos hozamú pontja; ez a numerikus "
        "optimalizálás és a front diszkretizálásának következménye lehet."
    )


def build_weights_table(
    asset_classes,
    weights_by_portfolio: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Transposed weights table: one row per portfolio, one column per ticker."""
    tickers = [ticker_label(asset) for asset in asset_classes]
    rows = []
    for key in PORTFOLIO_ORDER:
        weights = np.asarray(weights_by_portfolio[key], dtype=float)
        if weights.shape != (len(asset_classes),):
            raise ValueError(
                f"weights for {key!r} must have shape ({len(asset_classes)},)"
            )
        row = {"Portfólió": PORTFOLIO_LABELS[key]}
        row.update({ticker: float(weight) for ticker, weight in zip(tickers, weights)})
        rows.append(row)
    return pd.DataFrame(rows)


def build_metrics_table(
    asset_classes,
    weights_by_portfolio: dict[str, np.ndarray],
    expected_returns,
    monthly_returns,
    confidence: float,
) -> pd.DataFrame:
    """Annualized return and monthly CVaR for every displayed portfolio."""
    rows = []
    for key in PORTFOLIO_ORDER:
        weights = np.asarray(weights_by_portfolio[key], dtype=float)
        cvar, annual_return = portfolio_metrics(
            weights, expected_returns, monthly_returns, confidence
        )
        rows.append(
            {
                "Portfólió": PORTFOLIO_LABELS[key],
                "Évesített hozam": annual_return,
                "Havi CVaR": cvar,
            }
        )
    return pd.DataFrame(rows)
