# uncertainty_in_asset_allocation

A small research dashboard for studying how uncertainty in **input estimates**
(expected returns and the return distribution) propagates into a CVaR-based
asset-allocation decision.

The app visualizes the point-estimate efficient frontier, three reference
portfolios (minimum risk, CVaR tangency and CVaR risk parity), and the
bootstrap uncertainty cloud around them.

## What is in the repository

| Path | Purpose |
| --- | --- |
| `src/prepare_data.py` | Downloads/caches monthly returns, runs the joint bootstrap, and writes the HDF5 bundle. |
| `src/app.py` | Streamlit dashboard (Hungarian UI). |
| `src/portfolio.py` | Shared portfolio math: CVaR, risk contributions, optimizers, frontier, bootstrap helpers. |
| `src/data_io.py` | Versioned HDF5 persistence. |
| `src/config.py` | Asset universe, paths, seeds, and simulation defaults. |
| `data/monthly_returns.csv` | Cached monthly simple returns, 2010-01 to 2024-12. |
| `data/asset_allocation_uncertainty.h5` | Pre-computed simulation bundle used by the app. |
| `tests/` | Pytest suite for the math, pipeline, and HDF5 round-trip. |

## Assets and data

The universe contains four USD-denominated ETFs:

- `USD Készpénz (BIL)` — 1-3 month Treasury bill ETF (cash proxy)
- `USA Állampapír (IEF)` — 7-10 year Treasury bond ETF
- `USA Váll. Kötvény (LQD)` — investment-grade corporate bond ETF
- `Arany (GLD)` — gold ETF

Monthly simple returns are computed from Yahoo Finance adjusted close prices
with `resample("ME").last().pct_change()`. The CSV cache is used by default so
the committed results are reproducible without network access. Pass
`--force-download` to refresh the cache.

## Methodology

### Point estimates

- Expected returns are annualized as `12 * monthly mean`.
- Risk is the empirical **CVaR (Expected Shortfall)** at the 95% level on
  monthly returns, reported as a positive loss number.
- Portfolios are long-only and fully invested. The optimizers use SLSQP with
  weights in `[0, 1]` and a sum-to-one constraint.
- The efficient frontier minimizes CVaR for a grid of target annualized
  returns. The grid starts at the minimum-risk portfolio's return, so
  dominated, backward-bending points are not plotted. Any remaining dominated
  points found by the numerical optimizer are removed by a lower-envelope
  filter.

### Input uncertainty: joint bootstrap

The previous implementation drew expected returns from a normal distribution
whose covariance was scaled by `12` instead of `144`. Because the annualized
mean is `12 * monthly mean`, its sampling covariance is

```
Cov(12 * mean) = 144 * Sigma / n,
```

not `12 * diag(Sigma) / n`. The old code therefore understated the standard
deviation of the expected-return estimate by a factor of `sqrt(12) ≈ 3.46`.

The pipeline now uses a **joint row bootstrap** instead:

1. Resample the monthly return rows with replacement.
2. Use the bootstrap sample mean × 12 as the expected-return draw.
3. Use the **same** bootstrap sample to estimate CVaR and the efficient
   frontier.

This keeps expected-return uncertainty and distributional uncertainty
coherent, captures the full covariance structure, and avoids a separate
normality assumption for the mean estimator.

### Reproducibility

The default bundle uses `seed=42` and 100 simulations. The seed and simulation
count are stored as HDF5 attributes and shown in the app caption. Re-running
`prepare_data.py` with the same seed reproduces the bundle; the test suite
checks this explicitly.

## Installation

Python 3.12 is recommended. The versions below are the ones used for the
committed bundle.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

## Build the data bundle

```bash
python src/prepare_data.py
```

Useful options:

```bash
python src/prepare_data.py --simulations 200 --seed 7
python src/prepare_data.py --force-download
python src/prepare_data.py --start 2015-01-01 --end 2024-12-31
```

Dates passed to `--start` and `--end` are inclusive. Run
`python src/prepare_data.py --help` for all options.

## Run the app

```bash
streamlit run src/app.py
```

The app reads `data/asset_allocation_uncertainty.h5`. If the file is missing or
was written with an older schema, the app shows an error asking you to run
`python src/prepare_data.py`.

## Tests

```bash
python -m pytest
```

The suite covers:

- CVaR, risk contributions, and weight validation;
- feasibility and optimality of the three reference portfolios;
- monotonicity of the efficient frontier;
- the `144 * Sigma / n` covariance scaling and the joint-bootstrap covariance;
- pipeline reproducibility and shape/consistency checks;
- HDF5 round-trip and schema-version rejection.

## Validation of the committed bundle

With the committed data (`n=180`, `100` simulations, `seed=42`):

- The equal-weight portfolio's empirical bootstrap variance of the
  annualized expected return is within **1.5%** of the theoretical
  `(n-1)/n * 144 * Sigma / n` reference.
- The diagonal standard-deviation ratios are between **0.89 and 1.10** of the
  theoretical bootstrap reference (sampling noise with 100 draws).
- All 100 special-portfolio optimizations converged after the tangency
  multi-start fix.
- All 100 efficient frontiers are non-empty and monotone in both CVaR and
  expected return; the average number of retained points is 19.5 out of 20.
- The full test suite passes: **19 tests**.

These are statistical checks, not a guarantee that the point estimates are
unbiased. The bootstrap is conditional on the historical sample.

## Limitations

- The bootstrap is i.i.d. across months and ignores serial dependence,
  volatility clustering, and regime shifts.
- Estimation is in-sample; there is no out-of-sample or transaction-cost
  analysis.
- The risk-free rate is a constant 2% assumption, while the asset universe
  includes a cash proxy (BIL).
- CVaR at 95% with 180 monthly observations uses roughly nine tail
  observations, so the risk estimate itself is noisy.
- The asset universe is small and defensive; the minimum-risk portfolio can
  legitimately concentrate in cash.
- Expected returns are annualized by ×12 rather than by compounding; this is a
  deliberate convention for the risk-return plot, not a forecast.

## HDF5 schema (version 2)

```
attrs: title, schema_version, seed, n_simulations,
       confidence_level, risk_free_rate
base/
  returns        [n_obs, n_assets] float64
  dates          [n_obs] ISO-8601 strings
  asset_names    [n_assets] UTF-8 strings
simulation_inputs/
  expected_returns   [n_sim, n_assets] float64
  bootstrap_returns  [n_sim, n_obs, n_assets] float64
simulation_outputs/
  min_risk_points / min_risk_weights
  tangency_points / tangency_weights
  risk_parity_points / risk_parity_weights
  efficient_frontiers/frontier_{i}  [n_points, 2] (cvar, ret)
```

Bootstrap samples are stored as a 3-D array; their row order is the resampled
order and carries no date meaning. Dates are stored as ISO strings to avoid
platform-dependent integer units.
