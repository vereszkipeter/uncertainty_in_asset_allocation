"""HDF5 persistence for the simulation bundle.

The on-disk layout is intentionally simple and versioned:

``base/``
    Historical monthly returns, dates and asset names.
``simulation_inputs/``
    One annualized expected-return vector and one bootstrap return matrix per
    simulation.
``simulation_outputs/``
    Special-portfolio points/weights and the CVaR efficient frontiers.

Storing bootstrap returns as a 3-D array avoids the previous mixed-type
DataFrame flattening (integer dates + floats) and the fragile ``decode`` step.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from config import SCHEMA_VERSION, SPECIAL_KEYS


def _decode(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _decode_array(values) -> list[str]:
    return [_decode(value) for value in values]


def save_results(
    path,
    *,
    title: str,
    returns_data: pd.DataFrame,
    expected_returns: np.ndarray,
    bootstrap_returns: np.ndarray,
    special_points: dict[str, np.ndarray],
    special_weights: dict[str, np.ndarray],
    frontiers: list[pd.DataFrame],
    metadata: dict,
) -> None:
    """Write a complete simulation bundle to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as handle:
        handle.attrs["title"] = title
        for key, value in metadata.items():
            handle.attrs[key] = value

        base = handle.create_group("base")
        base.create_dataset("returns", data=returns_data.to_numpy(dtype=float))
        base.create_dataset(
            "dates",
            data=np.asarray(
                [date.strftime("%Y-%m-%d") for date in returns_data.index],
                dtype=object,
            ),
            dtype=h5py.string_dtype("utf-8"),
        )
        base.create_dataset(
            "asset_names",
            data=np.asarray(list(returns_data.columns), dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )

        inputs = handle.create_group("simulation_inputs")
        inputs.create_dataset(
            "expected_returns", data=np.asarray(expected_returns, dtype=float)
        )
        inputs.create_dataset(
            "bootstrap_returns", data=np.asarray(bootstrap_returns, dtype=float)
        )

        outputs = handle.create_group("simulation_outputs")
        for key in SPECIAL_KEYS:
            outputs.create_dataset(
                f"{key}_points", data=np.asarray(special_points[key], dtype=float)
            )
            outputs.create_dataset(
                f"{key}_weights", data=np.asarray(special_weights[key], dtype=float)
            )

        frontier_group = outputs.create_group("efficient_frontiers")
        for index, frontier in enumerate(frontiers):
            if frontier is None or frontier.empty:
                continue
            frontier_group.create_dataset(
                f"frontier_{index}",
                data=frontier[["cvar", "ret"]].to_numpy(dtype=float),
            )


def load_results(path) -> dict:
    """Load a simulation bundle produced by :func:`save_results`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as handle:
        schema_version = int(handle.attrs.get("schema_version", 0))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported HDF5 schema version {schema_version}; "
                f"expected {SCHEMA_VERSION}. Run 'python src/prepare_data.py' "
                "to regenerate the data bundle."
            )

        asset_names = _decode_array(handle["base/asset_names"][:])
        dates = pd.to_datetime(_decode_array(handle["base/dates"][:]))
        base_returns = pd.DataFrame(
            handle["base/returns"][:], index=dates, columns=asset_names
        )

        outputs = handle["simulation_outputs"]
        data: dict = {
            "title": _decode(handle.attrs.get("title", "")),
            "asset_classes": asset_names,
            "base_returns": base_returns,
            "schema_version": schema_version,
            "seed": int(handle.attrs.get("seed", -1)),
            "n_simulations": int(handle.attrs.get("n_simulations", 0)),
            "confidence_level": float(handle.attrs.get("confidence_level", 0.95)),
            "risk_free_rate": float(handle.attrs.get("risk_free_rate", 0.02)),
        }

        for key in SPECIAL_KEYS:
            data[f"sim_{key}_points"] = pd.DataFrame(
                outputs[f"{key}_points"][:], columns=["cvar", "ret"]
            )
            data[f"sim_{key}_weights"] = outputs[f"{key}_weights"][:]

        frontier_group = outputs["efficient_frontiers"]
        frontier_names = sorted(
            frontier_group.keys(),
            key=lambda name: int(name.rsplit("_", 1)[-1]),
        )
        data["sim_frontiers"] = [
            pd.DataFrame(frontier_group[name][:], columns=["cvar", "ret"])
            for name in frontier_names
        ]

        inputs = handle["simulation_inputs"]
        data["sim_expected_returns"] = inputs["expected_returns"][:]
        bootstrap_returns = inputs["bootstrap_returns"][:]
        data["bootstrap_samples_list"] = [
            pd.DataFrame(bootstrap_returns[index], columns=asset_names)
            for index in range(bootstrap_returns.shape[0])
        ]

    return data
