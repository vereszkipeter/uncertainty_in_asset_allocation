"""Build the simulation bundle consumed by the Streamlit app.

The pipeline uses a joint row bootstrap:

1. resample monthly return rows with replacement;
2. use the bootstrap sample mean (annualized by 12) as the expected-return
   draw;
3. use the same bootstrap sample to estimate CVaR and the efficient frontier.

This keeps expected-return uncertainty and distributional uncertainty
coherent. In particular, the covariance of the bootstrap mean is of order
``144 * Sigma / n``, which is the correct annualized sampling covariance of
the expected-return estimator.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance

# Make sibling modules importable when this file is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    ASSET_CLASSES,
    CONFIDENCE_LEVEL,
    END_DATE,
    N_FRONTIER_POINTS,
    N_SIMULATIONS,
    RETURNS_CSV_FILE,
    RISK_FREE_RATE,
    SCHEMA_VERSION,
    SEED,
    SPECIAL_KEYS,
    START_DATE,
    TICKER_LIST,
    TITLE,
    HDF5_FILE,
)
from data_io import save_results  # noqa: E402
from portfolio import (  # noqa: E402
    annualized_mean,
    draw_bootstrap_sample,
    efficient_frontier,
    optimize_special_portfolios,
    portfolio_metrics,
    portfolio_return,
)


@dataclass
class SimulationResult:
    """Arrays produced by :func:`run_simulations`."""

    expected_returns: np.ndarray
    bootstrap_returns: np.ndarray
    special_points: dict[str, np.ndarray]
    special_weights: dict[str, np.ndarray]
    frontiers: list[pd.DataFrame]
    failures: dict[str, int]


def validate_returns(returns_data: pd.DataFrame, asset_classes: list[str]) -> None:
    """Fail fast on malformed return data."""
    if returns_data.empty:
        raise ValueError("return data is empty")
    if list(returns_data.columns) != list(asset_classes):
        raise ValueError(
            "return columns do not match the configured asset classes: "
            f"{list(returns_data.columns)} != {list(asset_classes)}"
        )
    if returns_data.isna().any().any():
        raise ValueError("return data contains missing values")
    if not np.isfinite(returns_data.to_numpy(dtype=float)).all():
        raise ValueError("return data contains non-finite values")
    if not isinstance(returns_data.index, pd.DatetimeIndex):
        raise ValueError("return data must have a DatetimeIndex")
    if not returns_data.index.is_monotonic_increasing:
        raise ValueError("return data must be sorted by date")
    if len(returns_data) < 2:
        raise ValueError("at least two monthly observations are required")


def get_returns_data(
    csv_path,
    tickers: list[str],
    asset_classes: list[str],
    start_date: str,
    end_date: str,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load monthly returns from the CSV cache, or download them.

    The cache is deliberately used by default so the pipeline is reproducible
    and does not depend on the network. Pass ``force_download=True`` to refresh
    it from Yahoo Finance.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not force_download:
        print(f"Loading monthly returns from cache: '{csv_path}'")
        returns_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        returns_data.index.name = "date"
        missing = [name for name in asset_classes if name not in returns_data.columns]
        if missing:
            raise ValueError(f"cached CSV is missing columns: {missing}")
        returns_data = returns_data.loc[:, asset_classes]
        validate_returns(returns_data, asset_classes)
        return returns_data

    print("Downloading monthly returns from Yahoo Finance...")
    # Yahoo Finance treats `end` as exclusive; the CLI date is inclusive.
    download_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    full_data = yfinance.download(
        tickers,
        start=start_date,
        end=download_end,
        progress=False,
        auto_adjust=True,
    )
    if full_data is None or full_data.empty:
        raise RuntimeError("Yahoo Finance returned no data")

    close = full_data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    else:
        missing = [ticker for ticker in tickers if ticker not in close.columns]
        if missing:
            raise RuntimeError(f"downloaded data is missing tickers: {missing}")
        close = close.loc[:, tickers]

    returns_data = (
        close.resample("ME").last().pct_change(fill_method=None).dropna()
    )
    returns_data.columns = asset_classes
    returns_data.index.name = "date"
    validate_returns(returns_data, asset_classes)
    returns_data.to_csv(csv_path)
    print(f"Saved monthly returns to cache: '{csv_path}'")
    return returns_data


def run_simulations(
    returns_data: pd.DataFrame,
    *,
    n_simulations: int = N_SIMULATIONS,
    seed: int = SEED,
    risk_free_rate: float = RISK_FREE_RATE,
    confidence: float = CONFIDENCE_LEVEL,
    n_frontier_points: int = N_FRONTIER_POINTS,
    progress=None,
) -> SimulationResult:
    """Run the joint bootstrap simulation and return all arrays."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")

    n_obs, n_assets = returns_data.shape
    rng = np.random.default_rng(seed)

    expected_returns = np.empty((n_simulations, n_assets), dtype=float)
    bootstrap_returns = np.empty((n_simulations, n_obs, n_assets), dtype=float)
    special_points = {
        key: np.empty((n_simulations, 2), dtype=float) for key in SPECIAL_KEYS
    }
    special_weights = {
        key: np.empty((n_simulations, n_assets), dtype=float)
        for key in SPECIAL_KEYS
    }
    frontiers: list[pd.DataFrame] = []
    failures = {key: 0 for key in SPECIAL_KEYS}

    for index in range(n_simulations):
        sample = draw_bootstrap_sample(returns_data, rng)
        sim_expected_returns = annualized_mean(sample)

        expected_returns[index] = sim_expected_returns
        bootstrap_returns[index] = sample.to_numpy(dtype=float)

        special = optimize_special_portfolios(
            sim_expected_returns,
            sample,
            risk_free_rate,
            confidence=confidence,
        )
        min_risk_return = portfolio_return(
            special["min_risk"].weights, sim_expected_returns
        )
        frontiers.append(
            efficient_frontier(
                sim_expected_returns,
                sample,
                n_points=n_frontier_points,
                confidence=confidence,
                min_return=min_risk_return,
            )
        )

        for key in SPECIAL_KEYS:
            result = special[key]
            if not result.success:
                failures[key] += 1
            special_weights[key][index] = result.weights
            special_points[key][index] = portfolio_metrics(
                result.weights,
                sim_expected_returns,
                sample,
                confidence=confidence,
            )

        if progress is not None:
            progress(index + 1, n_simulations)

    return SimulationResult(
        expected_returns=expected_returns,
        bootstrap_returns=bootstrap_returns,
        special_points=special_points,
        special_weights=special_weights,
        frontiers=frontiers,
        failures=failures,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the uncertainty-in-asset-allocation HDF5 bundle."
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=N_SIMULATIONS,
        help=f"number of bootstrap simulations (default: {N_SIMULATIONS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"random seed (default: {SEED})",
    )
    parser.add_argument(
        "--start",
        default=START_DATE,
        help=f"start date for the download (default: {START_DATE})",
    )
    parser.add_argument(
        "--end",
        default=END_DATE,
        help=f"end date for the download (default: {END_DATE})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_LEVEL,
        help=f"CVaR confidence level (default: {CONFIDENCE_LEVEL})",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=RISK_FREE_RATE,
        help=f"annual risk-free rate (default: {RISK_FREE_RATE})",
    )
    parser.add_argument(
        "--frontier-points",
        type=int,
        default=N_FRONTIER_POINTS,
        help=f"points per efficient frontier (default: {N_FRONTIER_POINTS})",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="ignore the CSV cache and download fresh data",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start_time = time.time()

    returns_data = get_returns_data(
        RETURNS_CSV_FILE,
        TICKER_LIST,
        ASSET_CLASSES,
        args.start,
        args.end,
        force_download=args.force_download,
    )
    validate_returns(returns_data, ASSET_CLASSES)

    print(
        f"\nRunning {args.simulations} joint bootstrap simulations "
        f"(seed={args.seed}, confidence={args.confidence:.0%})..."
    )

    def progress(done: int, total: int) -> None:
        print(f"  - {done}/{total}")

    result = run_simulations(
        returns_data,
        n_simulations=args.simulations,
        seed=args.seed,
        risk_free_rate=args.risk_free_rate,
        confidence=args.confidence,
        n_frontier_points=args.frontier_points,
        progress=progress,
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "n_simulations": args.simulations,
        "confidence_level": args.confidence,
        "risk_free_rate": args.risk_free_rate,
    }
    save_results(
        HDF5_FILE,
        title=TITLE,
        returns_data=returns_data,
        expected_returns=result.expected_returns,
        bootstrap_returns=result.bootstrap_returns,
        special_points=result.special_points,
        special_weights=result.special_weights,
        frontiers=result.frontiers,
        metadata=metadata,
    )

    failed = {key: count for key, count in result.failures.items() if count}
    if failed:
        print(f"Warning: optimizer failures (fallback used): {failed}")
    else:
        print("All special-portfolio optimizations converged.")

    print(f"\nSaved simulation bundle to '{HDF5_FILE}'.")
    print(f"Total runtime: {time.time() - start_time:.2f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
