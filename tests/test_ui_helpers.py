"""Tests for the presentation helpers used by the Streamlit dashboard."""

import numpy as np
import pandas as pd
import pytest

from ui_helpers import (
    build_metrics_table,
    build_weights_table,
    cvar_at_return,
    cvar_gap_pp,
    format_weights_for_tooltip,
    frontier_insight,
    ticker_label,
    to_rgba,
)


@pytest.fixture
def frontier() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cvar": [0.005, 0.010, 0.020, 0.040],
            "ret": [0.01, 0.02, 0.04, 0.08],
        }
    )


def test_ticker_label_extracts_parenthetical_ticker() -> None:
    assert ticker_label("USD Készpénz (BIL)") == "BIL"
    assert ticker_label("Arany (GLD)") == "GLD"
    assert ticker_label("No ticker") == "No ticker"


def test_to_rgba_converts_hex_color() -> None:
    assert to_rgba("#0072B2", 0.25) == "rgba(0, 114, 178, 0.25)"


def test_format_weights_for_tooltip_lists_every_asset() -> None:
    text = format_weights_for_tooltip([0.5, 0.5], ["BIL", "IEF"])
    assert "Portfólió súlyok" in text
    assert "BIL: 50.00%" in text
    assert "IEF: 50.00%" in text


def test_cvar_at_return_interpolates(frontier) -> None:
    assert cvar_at_return(frontier, 0.03) == pytest.approx(0.015)
    assert cvar_at_return(frontier, 0.0) == pytest.approx(0.005)
    assert cvar_at_return(frontier, 1.0) == pytest.approx(0.040)
    assert np.isnan(cvar_at_return(pd.DataFrame(), 0.03))


def test_cvar_gap_pp_is_positive_when_user_is_riskier(frontier) -> None:
    gap = cvar_gap_pp(user_cvar=0.025, user_return=0.03, frontier=frontier)
    assert gap == pytest.approx(1.0)


def test_frontier_insight_reports_positive_gap(frontier) -> None:
    text = frontier_insight(user_cvar=0.025, user_return=0.03, frontier=frontier)
    assert "1.00 százalékponttal magasabb" in text


def test_frontier_insight_handles_outside_return_range(frontier) -> None:
    text = frontier_insight(user_cvar=0.025, user_return=0.20, frontier=frontier)
    assert "hozamtartományán kívül" in text


def test_weights_table_is_transposed_and_ordered() -> None:
    assets = ["USD Készpénz (BIL)", "USA Állampapír (IEF)"]
    weights = {
        "user": np.array([0.4, 0.6]),
        "min_risk": np.array([0.9, 0.1]),
        "tangency": np.array([0.2, 0.8]),
        "risk_parity": np.array([0.7, 0.3]),
    }
    table = build_weights_table(assets, weights)
    assert list(table.columns) == ["Portfólió", "BIL", "IEF"]
    assert table["Portfólió"].tolist() == [
        "Az Ön portfóliója",
        "Min Risk",
        "Tangency",
        "Risk Parity",
    ]
    assert table.loc[0, "BIL"] == pytest.approx(0.4)
    assert table.loc[1, "IEF"] == pytest.approx(0.1)


def test_metrics_table_reports_return_and_cvar() -> None:
    assets = ["A (A)", "B (B)"]
    weights = {
        "user": np.array([0.5, 0.5]),
        "min_risk": np.array([1.0, 0.0]),
        "tangency": np.array([0.0, 1.0]),
        "risk_parity": np.array([0.5, 0.5]),
    }
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, -0.04],
            "B": [0.02, -0.01, 0.04, -0.03],
        }
    )
    expected = np.array([0.10, 0.20])
    table = build_metrics_table(assets, weights, expected, returns, confidence=0.95)
    assert list(table.columns) == ["Portfólió", "Évesített hozam", "Havi CVaR"]
    assert len(table) == 4
    assert table.loc[0, "Évesített hozam"] == pytest.approx(0.15)
    assert table.loc[0, "Havi CVaR"] > 0
