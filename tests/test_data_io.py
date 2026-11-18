"""Tests for the versioned HDF5 persistence layer."""

import numpy as np
import pandas as pd
import pytest

from config import SCHEMA_VERSION, SPECIAL_KEYS
from data_io import load_results, save_results


def _bundle():
    rng = np.random.default_rng(42)
    asset_names = ["Asset A", "Asset B", "Asset C"]
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    returns_data = pd.DataFrame(
        rng.normal(0.005, 0.02, size=(24, 3)), index=dates, columns=asset_names
    )
    n_simulations = 4
    expected_returns = rng.normal(0.05, 0.01, size=(n_simulations, 3))
    bootstrap_returns = rng.normal(
        0.005, 0.02, size=(n_simulations, 24, 3)
    )
    special_points = {
        key: rng.normal(0.05, 0.01, size=(n_simulations, 2))
        for key in SPECIAL_KEYS
    }
    special_weights = {
        key: rng.dirichlet(np.ones(3), size=n_simulations)
        for key in SPECIAL_KEYS
    }
    frontiers = [
        pd.DataFrame(
            {
                "cvar": np.linspace(0.01, 0.05, 5),
                "ret": np.linspace(0.02, 0.08, 5),
            }
        )
        for _ in range(n_simulations)
    ]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": 42,
        "n_simulations": n_simulations,
        "confidence_level": 0.95,
        "risk_free_rate": 0.02,
    }
    return {
        "title": "Test bundle",
        "returns_data": returns_data,
        "expected_returns": expected_returns,
        "bootstrap_returns": bootstrap_returns,
        "special_points": special_points,
        "special_weights": special_weights,
        "frontiers": frontiers,
        "metadata": metadata,
    }


def test_roundtrip(tmp_path) -> None:
    bundle = _bundle()
    path = tmp_path / "bundle.h5"
    save_results(path, **bundle)
    loaded = load_results(path)

    assert loaded["title"] == "Test bundle"
    assert loaded["asset_classes"] == ["Asset A", "Asset B", "Asset C"]
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["seed"] == 42
    assert loaded["n_simulations"] == 4
    assert loaded["confidence_level"] == pytest.approx(0.95)
    assert loaded["risk_free_rate"] == pytest.approx(0.02)

    pd.testing.assert_frame_equal(
        loaded["base_returns"],
        bundle["returns_data"],
        check_freq=False,
        check_index_type=False,
    )
    np.testing.assert_allclose(
        loaded["sim_expected_returns"], bundle["expected_returns"]
    )
    assert len(loaded["bootstrap_samples_list"]) == 4
    np.testing.assert_allclose(
        loaded["bootstrap_samples_list"][0].to_numpy(),
        bundle["bootstrap_returns"][0],
    )
    for key in SPECIAL_KEYS:
        np.testing.assert_allclose(
            loaded[f"sim_{key}_points"], bundle["special_points"][key]
        )
        np.testing.assert_allclose(
            loaded[f"sim_{key}_weights"], bundle["special_weights"][key]
        )
    assert len(loaded["sim_frontiers"]) == 4
    pd.testing.assert_frame_equal(
        loaded["sim_frontiers"][0], bundle["frontiers"][0]
    )


def test_schema_version_mismatch_raises(tmp_path) -> None:
    bundle = _bundle()
    bundle["metadata"] = {**bundle["metadata"], "schema_version": 1}
    path = tmp_path / "old.h5"
    save_results(path, **bundle)
    with pytest.raises(ValueError, match="Unsupported HDF5 schema version"):
        load_results(path)
