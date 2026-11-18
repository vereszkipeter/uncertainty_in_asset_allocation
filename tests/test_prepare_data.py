"""Tests for the data pipeline."""

import numpy as np
import pandas as pd
import pytest

from config import ASSET_CLASSES, SCHEMA_VERSION, SPECIAL_KEYS, TICKER_LIST
from portfolio import annualized_mean
from prepare_data import get_returns_data, run_simulations, validate_returns


def _synthetic_returns(n_obs: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n_assets = len(ASSET_CLASSES)
    returns = rng.normal(0.004, 0.02, size=(n_obs, n_assets))
    frame = pd.DataFrame(
        returns,
        columns=ASSET_CLASSES,
        index=pd.date_range("2015-01-31", periods=n_obs, freq="ME"),
    )
    frame.index.name = "date"
    return frame


def test_get_returns_data_reads_cache(tmp_path) -> None:
    returns = _synthetic_returns()
    csv_path = tmp_path / "returns.csv"
    returns.to_csv(csv_path)

    loaded = get_returns_data(
        csv_path,
        TICKER_LIST,
        ASSET_CLASSES,
        start_date="2015-01-01",
        end_date="2025-01-01",
        force_download=False,
    )
    pd.testing.assert_frame_equal(loaded, returns, check_freq=False)


def test_validate_returns_rejects_missing_values() -> None:
    returns = _synthetic_returns()
    returns.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_returns(returns, ASSET_CLASSES)


def test_validate_returns_rejects_wrong_columns() -> None:
    returns = _synthetic_returns()
    returns.columns = ["A", "B", "C", "D"]
    with pytest.raises(ValueError, match="asset classes"):
        validate_returns(returns, ASSET_CLASSES)


def test_run_simulations_shapes_and_consistency() -> None:
    returns = _synthetic_returns()
    result = run_simulations(
        returns,
        n_simulations=3,
        seed=42,
        risk_free_rate=0.02,
        confidence=0.95,
        n_frontier_points=6,
    )

    assert result.expected_returns.shape == (3, len(ASSET_CLASSES))
    assert result.bootstrap_returns.shape == (3, len(returns), len(ASSET_CLASSES))
    assert len(result.frontiers) == 3
    assert set(result.failures) == set(SPECIAL_KEYS)

    for index in range(3):
        np.testing.assert_allclose(
            result.expected_returns[index],
            annualized_mean(result.bootstrap_returns[index]),
        )
        for key in SPECIAL_KEYS:
            assert result.special_points[key].shape == (3, 2)
            assert result.special_weights[key].shape == (3, len(ASSET_CLASSES))
            assert result.special_weights[key][index].sum() == pytest.approx(1.0)
            assert np.all(result.special_weights[key][index] >= -1e-12)

    for frontier in result.frontiers:
        assert list(frontier.columns) == ["cvar", "ret"]


def test_run_simulations_is_reproducible() -> None:
    returns = _synthetic_returns()
    kwargs = dict(
        n_simulations=2,
        seed=123,
        risk_free_rate=0.02,
        confidence=0.95,
        n_frontier_points=5,
    )
    first = run_simulations(returns, **kwargs)
    second = run_simulations(returns, **kwargs)
    np.testing.assert_array_equal(
        first.expected_returns, second.expected_returns
    )
    np.testing.assert_array_equal(
        first.bootstrap_returns, second.bootstrap_returns
    )
    for key in SPECIAL_KEYS:
        np.testing.assert_array_equal(
            first.special_weights[key], second.special_weights[key]
        )


def test_schema_version_is_supported() -> None:
    # Guard against accidental schema changes without regenerating the bundle.
    assert SCHEMA_VERSION == 2
